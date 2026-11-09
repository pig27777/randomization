"""Batched v1.3 top grasp on MJWarp: heterogeneous-kappa batches and a vectorized protocol-v2.2 executor.

Physics runs on the GPU: native MJWarp step (float32), face forces with per-environment material
(warp_v13.face_forces_env, float64), compaction welds (warp_effects.compaction, unchanged since S2) and
the v1.3 point-connection pinch kernels. The protocol state machine mirrors top_grasp.evaluate line by
line in numpy over all environments; every step downloads only a small readout array.

Usage: results = run_batch(specs) with specs = [dict(kappa=Kappa, grasp=dict, seed=int, pull=False), ...];
specs are grouped into structural buckets (same segment counts and options) and run bucket by bucket.
"""
from __future__ import annotations

import dataclasses
import copy
import math
import time

import mujoco
import mujoco_warp as mjw
import mujoco_warp._src.types as mjw_types
import mujoco_warp._src.warp_util as mjw_util
import numpy as np
import warp as wp

import mj_adapter as A
import pinch
import top_grasp as T
import warp_effects as E
import warp_v13 as K
from model import G, Kappa
from mjwarp_backend import promote_constraints, promote_state, write_equalities, write_forces

wp.set_module_options({"enable_backward": False, "enable_fp_fma": False})

BATCHABLE = {f.name for f in dataclasses.fields(mjw_types.Model)
             if mjw_util.is_array_spec(f.type) and getattr(f.type, 'shape', ())[:1] == ('*',)}
MODEL_FIELDS = {f.name for f in dataclasses.fields(mjw_types.Model)}
SHARED_FIELDS = {'bvh_aabb', 'dof_M0', 'dof_length'}
SETTLE, APPROACH, PRESS, CLOSE, LIFT, CARRY, LOWER, OPEN, PULL = range(9)
PHASE_NAMES = ('settle', 'approach', 'press', 'close', 'lift', 'carry', 'lower', 'open', 'pull')
CONTROLLED = (CLOSE, LIFT, CARRY, LOWER, PULL)


@wp.kernel
def apply_controls(controls: wp.array2d(dtype=wp.float64), pad_qadr: wp.array2d(dtype=int),
                   pad_vadr: wp.array2d(dtype=int), pad_body: wp.array(dtype=int),
                   qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                   mocap: wp.array2d(dtype=wp.vec3), external: wp.array3d(dtype=wp.float64)):
    """controls[env] = pad A/B joint positions (6), velocities (6), pinch frame (3), spring force (1)."""
    env = wp.tid()
    for p in range(2):
        for c in range(3):
            qpos[env, pad_qadr[p, c]] = float(controls[env, 3*p+c])
            qvel[env, pad_vadr[p, c]] = float(controls[env, 6+3*p+c])
    mocap[env, 0] = wp.vec3(float(controls[env, 12]), float(controls[env, 13]), float(controls[env, 14]))
    external[env, pad_body[0], 1] = -controls[env, 15]
    external[env, pad_body[1], 1] = controls[env, 15]


def signature(sim):
    m = sim.m
    shape = tuple(getattr(m, n) for n in ('nq', 'nv', 'nbody', 'ngeom', 'nsite', 'ntendon', 'nwrap', 'neq', 'npair',
                                         'nexclude', 'nmocap', 'njnt'))
    return (sim.desc.rigid, sim.desc.kappa.segments, sim.desc.mobile is not None, shape)


def union_aabbs(values):
    """Conservative union of MuJoCo's local AABBs (center, half extent)."""
    values=np.asarray(values)
    if values.ndim!=3 or values.shape[-1]!=6 or np.any(values[...,3:]<0):
        raise ValueError('expected [environment, node, 6] AABBs with nonnegative half extents')
    lower=np.min(values[...,:3]-values[...,3:],axis=0)
    upper=np.max(values[...,:3]+values[...,3:],axis=0)
    return np.concatenate([(lower+upper)/2,(upper-lower)/2],axis=-1)


def stacked_model(sims, shared_audit=None):
    """One MJWarp model; fields that differ between environments become batched arrays."""
    base = sims[0].m
    audit={}
    for name in sorted(SHARED_FIELDS):
        values=np.stack([getattr(s.m,name) for s in sims])
        span=np.ptp(values,axis=0)
        audit[name]=dict(policy='conservative union' if name=='bvh_aabb' else 'first model representative',
            maximum_elementwise_range=float(np.max(span,initial=0.)),
            minimum_value=float(np.min(values,initial=np.inf)),maximum_value=float(np.max(values,initial=-np.inf)),
            consumed_by_mjwarp=name in MODEL_FIELDS,representative_index=0)
    inertia=np.array([s.m.stat.meaninertia for s in sims])
    audit['stat.meaninertia']=dict(policy='first model representative',minimum_value=float(inertia.min()),
        maximum_value=float(inertia.max()),maximum_elementwise_range=float(np.ptp(inertia)),representative_index=0)
    if any(not np.array_equal(base.bvh_aabb,s.m.bvh_aabb) for s in sims[1:]):
        # Do not mutate the CPU reference models retained for comparison.
        base=copy.copy(base)
        base.bvh_aabb[:]=union_aabbs(np.stack([s.m.bvh_aabb for s in sims]))
    differs, fixed_mismatch = ['eq_data'], []
    for name in dir(base):
        value = getattr(base, name)
        if name.startswith('_') or not isinstance(value, np.ndarray):
            continue
        if all(np.array_equal(value, getattr(s.m, name)) for s in sims[1:]):
            continue
        if name in SHARED_FIELDS:
            continue
        (differs if name in BATCHABLE else fixed_mismatch).append(name)
    if fixed_mismatch:
        raise ValueError(f'non-batchable model fields differ inside a bucket: {sorted(set(fixed_mismatch))}')
    differs = sorted(set(differs))
    model = mjw.put_model(base, batch_sizes={n: len(sims) for n in differs})
    for name in differs:
        target = getattr(model, name)
        values = np.stack([getattr(s.m, name) for s in sims])
        target.assign(wp.array(values, dtype=target.dtype))
    if shared_audit is not None:shared_audit.update(audit)
    return model, differs


class Batch:
    """One structural bucket: MJWarp model/data, per-environment material, pinch tables and readouts."""

    def __init__(self, specs, nconmax=400, njmax=3000):
        self.specs = specs
        self.nworld = n = len(specs)
        built = []
        for s in specs:
            k, grasp = s['kappa'], T.TopGrasp(**s['grasp'])
            grasp.validate()
            yaw = 0. if s.get('pull') else math.pi / 2 - grasp.psi
            sim, starts, original_top = T.task_scene(k, grasp, yaw, bottom_fixed=bool(s.get('pull')))
            built.append((sim, starts, original_top, grasp))
        self.sims = sims = [b[0] for b in built]
        if len({signature(x) for x in sims}) != 1:
            raise ValueError('a Batch needs one structural bucket')
        sim = self.sim = sims[0]
        self.rigid = sim.desc.rigid
        self.grasp = [b[3] for b in built]
        self.starts = np.array([[b[1][p] for p in ('pA', 'pB')] for b in built])          # (n, 2, 3)
        self.original_top = np.array([b[2] for b in built])
        self.pad_p0 = np.array([[x.pad_p0[p] for p in ('pA', 'pB')] for x in sims])
        self.mass = np.array([x.mass for x in sims])
        self.dt = sim.m.opt.timestep
        with wp.ScopedDevice('cuda:0'):
            self.shared_model_fields={}
            self.model, self.batched_fields = stacked_model(sims,self.shared_model_fields)
            self.model.opt.warn_overflow = 0
            self.data = mjw.put_data(sim.m, sim.d, nworld=n, nconmax=nconmax, njmax=njmax)
            self.data.qpos.assign(np.stack([x.d.qpos for x in sims]).astype(np.float32))
            self.data.qvel.zero_()
            self.data.mocap_pos.assign(np.stack([x.d.mocap_pos for x in sims]).astype(np.float32))
            mjw.forward(self.model, self.data)
            m = sim.m
            f = lambda v, t=int: wp.array(np.asarray(v), dtype=t)
            self.xpos = wp.zeros((n, m.nbody), dtype=wp.vec3d)
            self.xmat = wp.zeros((n, m.nbody), dtype=wp.mat33d)
            self.xquat = wp.zeros((n, m.nbody), dtype=wp.vec4d)
            self.xipos = wp.zeros((n, m.nbody), dtype=wp.vec3d)
            self.qpos = wp.zeros((n, m.nq), dtype=wp.float64)
            self.external = wp.zeros((n, m.nbody, 6), dtype=wp.float64)
            self.forces = wp.zeros_like(self.external)
            self.eq_data = wp.array(np.stack([x.m.eq_data for x in sims]), dtype=wp.float64)
            self.efc_force = wp.zeros((n, njmax), dtype=wp.float64)
            self.count = wp.zeros(n, dtype=int)
            self.overflow = wp.zeros(1, dtype=int)
            self.contact_geom = wp.zeros((n, nconmax, 2), dtype=int)
            self.contact_frame = wp.zeros((n, nconmax, 9), dtype=wp.float64)
            self.contact_address = wp.full((n, nconmax, 3), -1, dtype=int)
            self.contact_dist = wp.zeros((n, nconmax), dtype=wp.float64)
            pad = np.full(m.ngeom, -1)
            for i, p in enumerate(('pA', 'pB')):
                pad[sorted(sim.pad_geoms[p])] = i
            self.pad_of_geom = f(pad)
            self.content = f(sim._obj_geom_mask, bool)
            self.bodies = f(sim.obj_bodies)
            self.body_mass = wp.array(np.stack([x.m.body_mass[x.obj_bodies] for x in sims]), dtype=wp.float64)
            self.pad_qadr = f([sim.pad_q[p] for p in ('pA', 'pB')])
            self.pad_vadr = f([sim.pad_v[p] for p in ('pA', 'pB')])
            self.pad_body = f([m.body(p).id for p in ('pA', 'pB')])
            self.controls = wp.zeros((n, 16), dtype=wp.float64)
            self.out = wp.zeros((n, 21), dtype=wp.float64)
            self.reaction_out = wp.zeros((n, 4), dtype=wp.float64)
            self.failure_out = wp.zeros((n, 4), dtype=wp.float64)
            if not self.rigid:
                b = sim.bushings
                for x in sims[1:]:
                    for name in ('a', 'b', 'node_a', 'node_b', 'axis', 'weight_a', 'weight_b'):
                        if not np.array_equal(getattr(b, name), getattr(x.bushings, name)):
                            raise ValueError(f'bushing topology {name} differs inside a bucket')
                stack = lambda name: np.stack([getattr(x.bushings, name) for x in sims])
                self.topology = [f(b.a), f(b.b), f(b.node_a), f(b.node_b), f(b.axis),
                                 f(b.weight_a, wp.float64), f(b.weight_b, wp.float64)]
                self.material = [wp.array(stack('ra'), dtype=wp.vec3d), wp.array(stack('kt'), dtype=wp.vec3d),
                                 wp.array(stack('kr'), dtype=wp.vec3d), wp.array(stack('tension'), dtype=wp.float64),
                                 wp.array(stack('compression_travel'), dtype=wp.float64)]
                limit = stack('limit')
                self.limits = [f(b.a), f(b.b), f(b.axis), self.material[0],
                               wp.array(stack('compression_limit'), dtype=wp.float64),
                               wp.array(np.where(np.isfinite(limit), limit, 1e300), dtype=wp.vec3d)]
                self.scratch = [wp.zeros((n, b.n), dtype=wp.vec3d) for _ in range(3)] + [
                    wp.zeros((n, b.n), dtype=wp.float64), wp.zeros((n, b.node_count), dtype=wp.vec3d),
                    wp.zeros((n, b.n), dtype=wp.float64)]
                self.compaction_inputs = [f(sim.seg_bodies), f(sim._obj_geom_mask, bool), f(sim.welds),
                                          f(b.node_a), f(b.node_b), f(sim._seg_qadr)]
                self.compaction_filtered = wp.zeros(n, dtype=wp.float64)
                self.compaction_raw = wp.zeros(n, dtype=wp.float64)
                probe = pinch.PinchAttachment(sim, self.original_top[0], T.WIDTH)
                self.top = probe.top
                self.top_arr, self.seg_bodies = f(self.top), f(sim.seg_bodies)
                self.geom_body = f(m.geom_bodyid)
                self.half = wp.array(np.stack([[s.half for s in x.desc.segments] for x in sims]), dtype=wp.vec3d)
                self.connects = f([probe.equations[i] for i in self.top])
                is_connect = np.zeros(m.neq, int)
                is_connect[list(probe.equations.values())] = 1
                self.is_connect = f(is_connect)
                self.top_contacts = wp.zeros((n, 2, len(self.top)), dtype=int)
                self.covered = wp.zeros((n, len(self.top)), dtype=int)
                self.eligible = wp.zeros(n, dtype=int)
                self.request = wp.zeros(n, dtype=int)
                self.tips = wp.zeros(n, dtype=wp.vec3d)
                self.tips2 = wp.zeros((n, 2), dtype=wp.vec3d)
                self.original_top_arr = wp.array(self.original_top, dtype=wp.float64)
            self.nconmax, self.njmax = nconmax, njmax

    # ---------------------------------------------------------------- physics
    def step(self, controls):
        d, m, n = self.data, self.sim.m, self.nworld
        self.controls.assign(controls)
        wp.launch(apply_controls, dim=n, inputs=[self.controls, self.pad_qadr, self.pad_vadr, self.pad_body,
                  d.qpos, d.qvel, d.mocap_pos, self.external])
        wp.copy(self.forces, self.external)
        if self.rigid:
            wp.launch(write_forces, dim=(n, m.nbody), inputs=[self.forces, d.xfrc_applied])
            mjw.step(self.model, d)
            self.promote()
        else:
            mjw.step1(self.model, d)
            self.promote()
            wp.launch(K.face_forces_env, dim=n, inputs=[*self.topology, *self.material, self.sim.bushings.centered,
                      self.xpos, self.xmat, self.xquat, self.xipos, self.forces, *self.scratch])
            wp.launch(write_forces, dim=(n, m.nbody), inputs=[self.forces, d.xfrc_applied])
            mjw.step2(self.model, d)
            self.promote()
        self.count.zero_()
        wp.launch(K.pack_contacts, dim=d.naconmax, inputs=[d.nacon, d.contact.worldid, d.contact.geom,
                  d.contact.frame, d.contact.efc_address, d.contact.dim, d.contact.dist, self.count, self.overflow,
                  self.contact_geom, self.contact_frame, self.contact_address, self.contact_dist])
        wp.launch(promote_constraints, dim=self.efc_force.shape, inputs=[d.efc.force, self.efc_force])
        if not self.rigid:
            wp.launch(E.compaction, dim=n, inputs=[*self.compaction_inputs, self.external, self.contact_geom,
                      self.contact_frame, self.contact_address[:, :, 0], self.efc_force, self.scratch[-1], self.qpos,
                      self.dt, A.COMPACTION_LOAD_TAU, A.WELD_ON, A.WELD_RELEASE, A.COMPACTION_LOAD_MIN,
                      self.compaction_filtered, self.compaction_raw, d.eq_active, self.eq_data])
            wp.launch(write_equalities, dim=(n, m.neq), inputs=[self.eq_data, self.model.eq_data])
        wp.launch(K.readouts, dim=n, inputs=[self.count, self.contact_geom, self.contact_frame, self.contact_address,
                  self.contact_dist, self.efc_force, self.pad_of_geom, self.content, self.sim.table_geom,
                  self.bodies, self.body_mass, self.xipos, self.qpos, self.pad_qadr, self.out])
        if not self.rigid:
            wp.launch(K.pinch_reaction, dim=n, inputs=[d.nefc, int(mujoco.mjtConstraint.mjCNSTR_EQUALITY),
                      d.efc.type, d.efc.id, self.efc_force, self.is_connect, d.eq_active, self.reaction_out])
        return self.out.numpy(), (self.reaction_out.numpy() if not self.rigid else None)

    def promote(self):
        d, m = self.data, self.sim.m
        wp.launch(promote_state, dim=(self.nworld, max(m.nbody, m.nq)), inputs=[d.xpos, d.xmat, d.xquat, d.xipos,
                  d.qpos, self.xpos, self.xmat, self.xquat, self.xipos, self.qpos])

    def failures(self):
        if self.rigid:
            q, v = self.data.qpos.numpy(), self.data.qvel.numpy()
            return np.c_[np.zeros((self.nworld, 3)), (np.isfinite(q).all(1) & np.isfinite(v).all(1)).astype(float)]
        wp.launch(K.failure_metrics, dim=self.nworld, inputs=[*self.limits, self.xpos, self.xmat, self.xquat,
                  self.data.qpos, self.data.qvel, self.failure_out])
        return self.failure_out.numpy()

    def overflowed(self):
        nefc = self.data.nefc.numpy()
        return bool(self.overflow.numpy()[0]) or int(self.data.nacon.numpy()[0]) > self.data.naconmax or \
            bool((nefc > self.njmax).any())

    # ---------------------------------------------------------------- pinch
    def candidates(self, request, tips):
        """Eligibility and covered top segments for the requested environments (PinchAttachment.candidate)."""
        if self.rigid:
            return np.zeros(self.nworld, bool), None
        self.request.assign(request.astype(np.int32))
        self.tips2.assign(tips)
        wp.launch(K.pinch_top_contacts, dim=self.nworld, inputs=[self.top_arr, self.seg_bodies, self.geom_body,
                  self.pad_of_geom, self.count, self.contact_geom, self.contact_frame, self.contact_address,
                  self.efc_force, self.xmat, self.top_contacts])
        wp.launch(K.pinch_candidate, dim=self.nworld, inputs=[self.request, self.top_arr, self.seg_bodies, self.half,
                  self.top_contacts, self.xpos, self.xmat, self.tips2, self.original_top_arr, pinch.DEPTH_MIN,
                  T.WIDTH, self.covered, self.eligible])
        return self.eligible.numpy().astype(bool) & request, self.covered.numpy()

    def bind(self, mask, points):
        self.request.assign(mask.astype(np.int32))
        self.tips.assign(points)
        wp.launch(K.pinch_bind, dim=self.nworld, inputs=[self.request, self.top_arr, self.seg_bodies, self.connects,
                  self.covered, self.xpos, self.xmat, self.tips, self.eq_data, self.data.eq_active])
        wp.launch(write_equalities, dim=(self.nworld, self.sim.m.neq), inputs=[self.eq_data, self.model.eq_data])

    def release(self, mask):
        if self.rigid or not mask.any():
            return
        self.request.assign(mask.astype(np.int32))
        wp.launch(K.pinch_release, dim=self.nworld, inputs=[self.request, self.connects, self.data.eq_active])


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]]).transpose(2, 0, 1) if np.ndim(a) else np.array([[c, -s], [s, c]])


def run(batch, max_time=90., progress=None):
    """Vectorized top_grasp.evaluate over a Batch; returns one result dict per environment."""
    n, dt = batch.nworld, batch.dt
    g = batch.grasp
    W, Dd, F = (np.array([getattr(x, a) for x in g]) for a in ('w', 'd', 'F'))
    v_lift, a_max = np.array([x.v_lift for x in g]), np.array([x.a_max for x in g])
    pull = np.array([bool(s.get('pull')) for s in batch.specs])
    category = [s['kappa'].category or ('hard' if s['kappa'].rigid else 'bendable') for s in batch.specs]
    threshold = np.array([T.THRESHOLD[c] for c in category])
    coefficient = np.array([s['kappa'].pinch_coeff for s in batch.specs], float)
    angle = np.array([np.random.default_rng(s['seed']).uniform(0, 2 * math.pi) for s in batch.specs])
    carry = .5 * np.c_[np.cos(angle), np.sin(angle), np.zeros(n)]
    carry_T = np.sqrt(2 * math.pi * .5 / a_max)
    sign = np.array([1., -1.])  # pA closes towards +y, pB towards -y
    starts, p0, top = batch.starts, batch.pad_p0, batch.original_top
    check_every = max(1, round(.010 / dt))
    phase = np.full(n, SETTLE); phase_t = np.zeros(n)
    arm = np.zeros((n, 3)); previous = starts.copy()
    squeeze = np.zeros(n); ctrl = np.zeros(n); measured = np.zeros(n); force_max = np.zeros(n)
    press_start = np.zeros(n); lift_start = np.zeros(n); effective_d = Dd.copy()
    com_rest = np.zeros((n, 3)); lift_height = np.full(n, np.nan); release_squeeze = np.zeros(n)
    spring_ref = np.full(n, np.nan); active = np.zeros(n, bool); broken = np.zeros(n, bool)
    filt = K.LoadFilter(n, pinch.LOAD_FILTER_TAU)
    lost = np.zeros(n); peak_pull = np.zeros(n); peak_weld = np.zeros(n)
    real_max = np.zeros(n); spring_max = np.zeros(n); current = np.zeros(n)
    outcome = [None] * n; reason = [None] * n; done = np.zeros(n, bool)
    events = [[] for _ in range(n)]; phases = [[(0., 'settle')] for _ in range(n)]
    out = np.zeros((n, 21)); t = 0.; steps = 0; wall = time.perf_counter()
    failure_state = [None] * n; overflow_steps = 0

    def go(mask, code):
        phase[mask] = code
        phase_t[mask] = t
        for i in np.flatnonzero(mask):
            phases[i].append((t, PHASE_NAMES[code]))

    def finish(mask, what, why):
        for i in np.flatnonzero(mask):
            if outcome[i] is None:
                outcome[i], reason[i] = what, why

    def tips_now():
        tips = p0 + out[:, 15:21].reshape(n, 2, 3)
        tips[:, :, 2] -= T.HEIGHT / 2
        return tips

    def gap_now():
        pads = p0 + out[:, 15:21].reshape(n, 2, 3)
        return pads[:, 1, 1] - pads[:, 0, 1] - 2 * T.RADIUS

    def spring(gap):
        return np.where(np.isnan(spring_ref), 0., pinch.GRIP_STIFFNESS *
                        np.maximum(np.nan_to_num(spring_ref) - pinch.GATHER_TRAVEL - gap, 0.))

    def release(mask, why):
        mask = mask & active
        batch.release(mask)
        for i in np.flatnonzero(mask):
            events[i].append(dict(t=t, event=why))
        active[mask] = False
        if why == 'overload':
            broken[mask] = True

    out[:, 15:21] = (starts - p0).reshape(n, 6)
    while not done.all():
        live = ~done & np.array([o is None for o in outcome])
        tau = t - phase_t
        start = phase.copy()       # top_grasp.evaluate runs one if/elif branch per iteration
        pad_f = out[:, 0:6].reshape(n, 2, 3)
        table_f = out[:, 6:9]
        com = out[:, 12:15]
        tips = tips_now()
        # ---- phase logic
        m = live & (start == SETTLE) & (tau >= .3)
        com_rest[m] = com[m]
        go(m, APPROACH)
        finish_press = np.zeros(n, bool); press_force = np.zeros(n)
        m = live & (start == APPROACH)
        if m.any():
            force = np.maximum(0., pad_f[:, :, 2].sum(1))
            touched = m & (np.linalg.norm(pad_f, axis=2) > .05).any(1)
            contact_height = tips[:, :, 2].mean(1)
            effective_d[touched] = np.minimum(Dd, .6 * np.maximum(contact_height, 0.))[touched]
            press_start[touched] = arm[touched, 2]
            go(touched, PRESS)
            finish_press |= touched & (force >= T.PRESS_FORCE_LIMIT)
            press_force = np.where(touched, force, press_force)
            missed = m & ~touched & (arm[:, 2] < -top - .010)
            finish(missed, 'grasp_fail', 'neither fingertip contacted the top')
            moving = m & ~touched & ~missed
            arm[moving, 2] = -T.PRESS_V * tau[moving]
        m = live & (start == PRESS)
        if m.any():
            force = np.maximum(0., pad_f[:, :, 2].sum(1))
            stop = m & ((force >= T.PRESS_FORCE_LIMIT) | (tau >= effective_d / T.PRESS_V))
            finish_press |= stop
            press_force = np.where(stop, force, press_force)
            moving = m & ~stop
            arm[moving, 2] = press_start[moving] - np.minimum(T.PRESS_V * tau[moving], effective_d[moving])
        close_end = live & (start == CLOSE) & ((ctrl >= .95 * F) | (squeeze >= W / 2) | (tau >= 8.))
        want = (finish_press | (close_end & (current > 0))) & ~active & ~broken
        eligible = np.zeros(n, bool)
        if want.any() and not batch.rigid:
            eligible, covered = batch.candidates(want, tips)
        if finish_press.any():
            gap = gap_now()
            spring_ref[finish_press] = np.where(eligible, gap, np.nan)[finish_press]
            go(finish_press, CLOSE)
        if close_end.any():
            bind = close_end & (current > 0) & eligible
            if bind.any():
                batch.bind(bind, tips.mean(1))
                active[bind] = True
                filt.reset(bind)
                for i in np.flatnonzero(bind):
                    events[i].append(dict(t=t, event='bind', segments=int(covered[i].sum())))
                lift_start[bind] = arm[bind, 2]
                go(bind & pull, PULL)
                go(bind & ~pull, LIFT)
            fail = close_end & ~bind
            release(fail, 'not_gripped')
            finish(fail, 'grasp_fail', 'top pinch conditions or current grip force absent')
        m = live & (start == LIFT)
        if m.any():
            arm[m, 2] = lift_start[m] + np.minimum(v_lift[m] * tau[m], T.LIFT_HEIGHT)
            hold = m & (tau >= T.LIFT_HEIGHT / v_lift + .2)
            lift_height[hold] = (com[hold, 2] - com_rest[hold, 2]) * 1000
            on_table = hold & (out[:, 9] > 0)
            finish(on_table, 'grasp_fail', 'object still contacts table after lift hold')
            go(hold & ~on_table, CARRY)
        m = live & (start == CARRY)
        if m.any():
            s = np.minimum(tau, carry_T)
            z = tau / carry_T - np.sin(2 * math.pi * s / carry_T) / (2 * math.pi)
            arm[m] = (np.c_[np.zeros(n), np.zeros(n), lift_start + T.LIFT_HEIGHT] + np.minimum(z, 1.)[:, None] * carry)[m]
            end = m & (tau >= carry_T + .3)
            low = end & (com[:, 2] - com_rest[:, 2] < .030)
            finish(low, 'drop', 'COM returned near the table after carry')
            go(end & ~low, LOWER)
        m = live & (start == LOWER)
        if m.any():
            arm[m] = (carry + np.c_[np.zeros(n), np.zeros(n),
                                    np.maximum(lift_start + T.LIFT_HEIGHT - .2 * tau, lift_start + .005)])[m]
            down = m & ((table_f[:, 2] >= .3 * batch.mass * G) | (arm[:, 2] <= lift_start + .005))
            release(down, 'open')
            spring_ref[down] = np.nan
            go(down, OPEN)
        m = live & (start == OPEN)
        if m.any():
            squeeze[m] = (release_squeeze - np.minimum(.05 * tau, .030))[m]
            end = m & (tau >= .030 / .05 + .5)
            for i in np.flatnonzero(end):
                offset = rot(-angle[i]) @ (com[i, :2] - com_rest[i, :2] - carry[i, :2])
                ok = bool(np.all(np.abs(offset) <= .100))
                outcome[i], reason[i] = ('success', None) if ok else ('drop', 'COM outside target square')
            done |= end
        m = live & (start == PULL)
        if m.any():
            arm[m, 2] = lift_start[m] + .005 * tau[m]
            finish(m & (tau > 60), 'grasp_fail', 'no pull-off within 60 s measurement window')
        stepping = ~done
        # ---- control and targets
        c = stepping & np.isin(phase, CONTROLLED)
        velocity = np.clip(.0025 * (F - ctrl), -T.CLOSE_V, T.CLOSE_V)
        squeeze[c] = np.clip(squeeze + velocity * dt, 0., W / 2)[c]
        release_squeeze = np.where(stepping & (phase != OPEN), squeeze, release_squeeze)
        targets = starts + np.stack([np.zeros(n), squeeze, np.zeros(n)], 1)[:, None, :] * \
            np.array([[0, 1, 0], [0, -1, 0]])[None] + arm[:, None, :]
        targets[done] = previous[done]
        gap_before = previous[:, 1, 1] - previous[:, 0, 1] - 2 * T.RADIUS
        controls = np.zeros((n, 16))
        controls[:, 0:6] = (previous - p0).reshape(n, 6)
        controls[:, 6:12] = ((targets - previous) / dt).reshape(n, 6)
        controls[:, 12:15] = targets.mean(1) - [0, 0, T.HEIGHT / 2]
        controls[:, 15] = spring(gap_before)
        previous = targets
        out, reaction = batch.step(controls)
        t += dt; steps += 1
        gap = gap_now()
        pad_f = out[:, 0:6].reshape(n, 2, 3)
        spring_force = spring(gap)
        real = (np.maximum(0., -pad_f[:, 0, 1]) + np.maximum(0., pad_f[:, 1, 1])) / 2
        upd = stepping
        current = np.where(upd, real + spring_force, current)
        real_max = np.where(upd, np.maximum(real_max, real), real_max)
        spring_max = np.where(upd, np.maximum(spring_max, spring_force), spring_max)
        ctrl = np.where(upd, ctrl + dt / (.005 + dt) * (current - ctrl), ctrl)
        measured = np.where(upd, measured + dt / (.020 + dt) * (current - measured), measured)
        force_max = np.where(upd, np.maximum(force_max, measured), force_max)
        if reaction is not None:
            weld = np.where((active & (reaction[:, 3] > 0))[:, None], reaction[:, :3], 0.)
            peak_weld = np.maximum(peak_weld, np.linalg.norm(weld, axis=1))
            pulling = upd & pull & (phase == PULL)
            peak_pull = np.where(pulling, np.maximum(peak_pull, np.maximum(-(weld[:, 2] + pad_f[:, :, 2].sum(1)), 0.)),
                                 peak_pull)
            broke = filt.update(dt, np.linalg.norm(weld, axis=1), coefficient * current, active & upd)
            if broke.any():
                release(broke, 'overload')
                spring_ref[broke] = np.nan
                finish(broke & pull, 'pull_off', '5 ms low-passed pinch load exceeded the capacity')
        open_ = upd & np.array([o is None for o in outcome])
        finish(open_ & (measured > threshold), 'over_squeeze', 'measured grip force above class threshold')
        if steps % check_every == 0:
            fm = batch.failures()
            open_ = upd & np.array([o is None for o in outcome])
            bad = open_ & ((fm[:, 3] < 1) | (fm[:, 0] > .5) | (fm[:, 1] > 1.2) | (fm[:, 2] > 1.2))
            for i in np.flatnonzero(bad):
                failure_state[i] = dict(t=t, phase=PHASE_NAMES[phase[i]], metrics=fm[i].tolist())
            finish(bad, 'sim_fail', 'non-finite state or Bushings.check over limit')
            watch = upd & np.isin(phase, (LIFT, CARRY)) & np.array([o is None for o in outcome])
            contact = np.maximum(np.where(active, current, 0.), np.maximum(out[:, 10], out[:, 11]))
            lost = np.where(watch, np.where(contact < .05, lost + check_every * dt, 0.), lost)
            gone = watch & (lost > .050)
            finish(gone & (phase == CARRY), 'drop', 'both contacts lost for >50 ms')
            finish(gone & (phase == LIFT), 'grasp_fail', 'both contacts lost for >50 ms')
            if batch.overflowed():
                overflow_steps += 1
            if progress:
                progress(t, done, outcome)
        finish(upd & (t > max_time), 'grasp_fail', f'execution did not complete within {max_time:g} s')
        done |= np.array([o is not None for o in outcome])
    wall = time.perf_counter() - wall
    return [dict(outcome=outcome[i], reason=reason[i], cls=category[i], max_grip_force_N=float(force_max[i]),
                 lift_height_mm=None if np.isnan(lift_height[i]) else float(lift_height[i]),
                 measured_pinch_coeff=float(peak_pull[i] / 10) if pull[i] else None, pull_force_N=float(peak_pull[i]),
                 peak_weld_force_N=float(peak_weld[i]), pinch_events=events[i], phases=phases[i],
                 max_real_grip_force_N=float(real_max[i]), max_spring_force_N=float(spring_max[i]),
                 failure_state=failure_state[i], sim_s=t, batch_wall_s=wall, batch_steps=steps,
                 overflow_checks=overflow_steps, protocol=T.PROTOCOL_VERSION, backend='mjwarp_v13')
            for i in range(n)]


def buckets(specs):
    """Group spec indices by structural signature (segment counts, rigidity, options)."""
    groups = {}
    for i, s in enumerate(specs):
        k = s['kappa']
        groups.setdefault((k.rigid, tuple(k.segments), bool(k.mobile)), []).append(i)
    return groups


def run_specs(specs, nconmax=400, njmax=3000, progress=None):
    results = [None] * len(specs)
    for key, idx in buckets(specs).items():
        batch = Batch([specs[i] for i in idx], nconmax=nconmax, njmax=njmax)
        for i, r in zip(idx, run(batch, progress=progress)):
            r['bucket'] = str(key)
            r['batched_fields'] = batch.batched_fields
            r['shared_model_fields'] = batch.shared_model_fields
            results[i] = r
    return results

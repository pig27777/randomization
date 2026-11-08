"""Newton backend of evaluate(object_spec, grasp_params, seed) (test/policy/README.md) on the proxy-real
test objects X1-X12 (objects.py). The task flow is task.py; this file only simulates and measures.

Example (Newton 1.7 interpreter):
  python -B evaluate.py X1 --params '{"u": 0, "v": 0, "psi": 1.5708, "force_N": 10, "lift_speed": 0.1, "max_accel": 2}'
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task  # noqa: E402
from objects import OBJECTS, PARK, Scene  # noqa: E402
from benchmark import _contact_probe  # noqa: E402  (test/contact_transfer, on the path via objects)
from grasp import _integrate_contact_force  # noqa: E402

PERIOD = .005          # control and measurement period (s)
CHECK_EVERY = 2        # stability checks every 2 periods (10 ms)
BOUNCE_SPEED = 1.0     # m/s, test/content_model/README.md; how it is measured: see the check below
SEED_SHIFT, SEED_YAW = .01, math.radians(10)


class SimFail(Exception):
    pass


@wp.kernel
def _place_pads(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], ids: wp.array[int],
                p_from: wp.array[wp.vec3], p_to: wp.array[wp.vec3], rot: wp.quat, fraction: float, period: float):
    i = wp.tid()
    body_q[ids[i]] = wp.transform(p_from[i] + fraction * (p_to[i] - p_from[i]), rot)
    body_qd[ids[i]] = wp.spatial_vector((p_to[i] - p_from[i]) / period, wp.vec3(0., 0., 0.))


@wp.kernel
def _drag(qd: wp.array[wp.vec3], mass: wp.array[float], rate: float, start: int, f: wp.array[wp.vec3]):
    i = wp.tid() + start
    f[i] = f[i] - mass[i] * rate * qd[i]


class Runner:
    def __init__(self, scene: Scene, rot):
        self.sc = sc = scene
        m = sc.model
        self.n = sc.substeps // 2
        if self.n % 2:
            raise ValueError(f'{sc.substeps} substeps per 10 ms: a {PERIOD * 1e3:g} ms period needs an even count')
        self.rot = rot
        self.ids = wp.array(sc.pads, dtype=int)
        self.p_from = wp.array([PARK, PARK], dtype=wp.vec3)
        self.p_to = wp.array([PARK, PARK], dtype=wp.vec3)
        self.probe_f = wp.zeros(m.particle_count, dtype=wp.vec3)
        self.shape_f = wp.zeros(m.shape_count, dtype=wp.vec3)
        self.kinds = wp.zeros(3, dtype=int)
        self.body_prev = wp.clone(m.body_q)
        self.impulse = wp.zeros(m.shape_count, dtype=wp.vec3d)
        self.graph = None
        self.periods = 0
        self.mass_wp = m.particle_mass

    def substep(self, k):
        sc, m, v, c = self.sc, self.sc.model, self.sc.solver, self.sc.contacts
        wp.launch(_place_pads, dim=2, inputs=[sc.s0.body_q, sc.s0.body_qd, self.ids, self.p_from, self.p_to, self.rot,
                                              (k + 1) / self.n, PERIOD])
        sc.s0.clear_forces()
        if sc.drag:
            wp.launch(_drag, dim=sc.bag_particles.stop - sc.bag_particles.start,
                      inputs=[sc.s0.particle_qd, self.mass_wp, sc.drag, sc.bag_particles.start, sc.s0.particle_f])
        sc.pipeline.collide(sc.s0, c)
        wp.copy(self.body_prev, v.body_q_prev)
        v.step(sc.s0, sc.s1, sc.control, c, sc.dt)
        self.probe_f.zero_(); self.shape_f.zero_(); self.kinds.zero_()
        wp.launch(_contact_probe, dim=c.soft_contact_max, inputs=[c.soft_contact_count, c.soft_contact_indices,
            c.soft_contact_barycentric, sc.s1.particle_q, v.particle_q_prev, m.particle_radius,
            v.body_particle_contact_penalty_k, v.body_particle_contact_material_kd, v.body_particle_contact_material_mu,
            v.friction_epsilon, m.shape_body, sc.s0.body_q, self.body_prev, sc.s0.body_qd, m.body_com,
            c.soft_contact_shape, c.soft_contact_body_pos, c.soft_contact_body_vel, c.soft_contact_normal,
            m.shape_margin, sc.dt, False, self.probe_f, self.shape_f, self.kinds])
        wp.launch(_integrate_contact_force, dim=m.shape_count, inputs=[self.shape_f, sc.dt, self.impulse])
        sc.s0, sc.s1 = sc.s1, sc.s0

    def period(self, targets):
        """Advance one period with the pads moving linearly to targets ((2, 3) world positions)."""
        self.p_from.assign(self.p_to.numpy())
        self.p_to.assign(np.asarray(targets, np.float32))
        if self.graph is None and self.periods > 0:
            with wp.ScopedCapture() as cap:
                for k in range(self.n):
                    self.substep(k)
            self.graph = cap.graph
        if self.graph is None:
            for k in range(self.n):  # first period eagerly, so kernels compile before capture
                self.substep(k)
        else:
            wp.capture_launch(self.graph)
        self.periods += 1
        sc = self.sc
        if sc.bag is not None and sc.bag.history.enabled:
            sc.bag._write_history(sc.s0.particle_q.numpy()[sc.bag_particles].astype(float), PERIOD)


class Package:
    """CPU-side reads of the package state."""

    def __init__(self, sc: Scene):
        self.sc = sc

    def points(self):
        q = self.sc.s0.particle_q.numpy()[self.sc.particles].astype(float)
        if not np.isfinite(q).all() or np.abs(q).max() > 5:
            raise SimFail('non-finite or runaway particle positions')
        return q

    def body_pos(self):
        return self.sc.s0.body_q.numpy()[self.sc.bodies, :3].astype(float) if self.sc.bodies else np.zeros((0, 3))

    def com(self, q=None):
        q = self.points() if q is None else q
        b = self.body_pos()
        return (self.sc.particle_mass @ q + self.sc.body_mass @ b) / self.sc.mass

    def min_jacobian(self, q):
        if self.sc.inv_rest is None:
            return 1.
        p = q[self.sc.tets]
        ds = np.stack([p[:, j] - p[:, 0] for j in (1, 2, 3)], axis=2)
        return float(np.linalg.det(ds @ self.sc.inv_rest).min())


def evaluate(object_spec, grasp_params, seed, trace=False):
    """Run one grasp episode; returns the README dict plus diagnostics."""
    t_wall = time.perf_counter()
    task.check_params(grasp_params)
    p = dict(grasp_params)
    rng = np.random.default_rng(seed)
    sc = Scene(object_spec)
    sc.place(rng.uniform(-SEED_YAW, SEED_YAW), rng.uniform(-SEED_SHIFT, SEED_SHIFT, 2))
    pkg = Package(sc)
    limit = task.SQUEEZE_LIMIT[sc.category]
    events, rows, info = {}, [], dict(object=object_spec, category=sc.category, seed=seed, params=p, mass_kg=sc.mass,
                                      substep_s=sc.dt, particles=sc.particles.stop, tets=int(sc.model.tet_count))
    max_grip, slip, lift_height = 0., 0., 0.
    t = 0.

    # Settle with the pads parked. The bag model owns its first frames (self-contact ramp).
    t_build = time.perf_counter() - t_wall
    run = None
    if sc.bag is not None:
        for _ in range(round(sc.settle_s / .01)):
            sc.bag.step()
    else:
        run = Runner(sc, wp.quat_identity())
        for _ in range(round(sc.settle_s / PERIOD)):
            run.period([PARK, PARK])
    t = sc.settle_s
    q = pkg.points()
    box_c, box_size, box_yaw = task.top_view_box(q[:, :2])
    height = float(q[:, 2].max())
    grasp, d, (s_plus, s_minus), pad_z, transport = task.grasp_frame(box_c, box_size, box_yaw, height, p)
    rot = wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), float(math.atan2(d[1], d[0]) - math.pi / 2))  # pad +y along d
    if run is None:
        run = Runner(sc, rot)
    else:
        run.rot, run.graph = rot, None  # rot is baked into the captured launches
    info.update(box_center=box_c.tolist(), box_size=box_size.tolist(), box_yaw=box_yaw, height=height,
                grasp_point=grasp.tolist(), close_dir=d.tolist())
    target = box_c + task.TRAVEL * transport
    base = np.array([[*grasp, pad_z], [*grasp, pad_z]])

    def pad_pos(s, dx=0., dz=0., opening=0.):
        out = base.copy()
        out[0, :2] += (s_plus - s[0] + opening) * d + dx * transport
        out[1, :2] -= (s_minus - s[1] + opening) * d
        out[1, :2] += dx * transport
        out[:, 2] += dz
        return out

    closer = task.Closer(p['force_N'])
    prev_imp = run.impulse.numpy().copy()
    prev_q, prev_pads, prev_com = None, None, None
    pads = pad_pos(closer.s)
    run.p_to.assign(pads.astype(np.float32))  # place the pads at the start without a jump
    state, t_closed, schedule, track, z_closed = 'close', None, None, None, None
    lost = 0.
    try:
        while True:
            # --- choose pad targets for this period
            if state == 'close':
                pads = pad_pos(closer.s)
            else:
                tc = t - t_closed
                if tc >= total:
                    break
                dx, dz, opening = offset(tc + PERIOD)
                pads = pad_pos(closer.s, dx, dz, opening)
            run.period(pads)
            t += PERIOD
            # --- measure
            imp = run.impulse.numpy()
            force = -(imp - prev_imp) / PERIOD
            prev_imp = imp.copy()
            f0 = float(force[sc.pad_shapes[0], :2] @ d)
            f1 = float(-(force[sc.pad_shapes[1], :2] @ d))
            grip = (f0 + f1) / 2
            max_grip = max(max_grip, grip)
            if grip > limit and 'over_squeeze' not in events:
                events['over_squeeze'] = f'grip force {grip:.1f} N > {limit:g} N at {t:.3f} s ({state})'
                break
            q = pkg.points()
            if run.periods % CHECK_EVERY == 0:
                com = pkg.com(q)
                if prev_q is not None:
                    # Ejection: a node relative to the package, or the package relative to the pads with its
                    # fall removed (a package that slips out from 0.1 m reaches 1.4 m/s; that is a drop).
                    h = CHECK_EVERY * PERIOD
                    com_v = (com - prev_com) / h
                    node = np.linalg.norm((q - prev_q) / h - com_v, axis=1).max()
                    whole = (com_v - (pads.mean(0) - prev_pads.mean(0)) / h) * [1, 1, 1]
                    whole[2] = max(whole[2], 0.)
                    whole = np.linalg.norm(whole)
                    if max(node, whole) > BOUNCE_SPEED:
                        raise SimFail(f'{"node" if node > whole else "package"} {max(node, whole):.2f} m/s '
                                      f'at {t:.3f} s ({state})')
                prev_q, prev_pads, prev_com = q, pads.copy(), com
                j = pkg.min_jacobian(q)
                if j <= 0:
                    raise SimFail(f'inverted tetrahedra (min J {j:.3f}) at {t:.3f} s ({state})')
                count = int(sc.contacts.soft_contact_count.numpy()[0])
                if count > sc.contacts.soft_contact_max:
                    raise SimFail(f'contact buffer overflow {count} at {t:.3f} s')
            com = pkg.com(q)
            if trace and run.periods % 10 == 0:
                rows.append(dict(t=round(t, 3), state=state, f=[round(f0, 3), round(f1, 3)], com=np.round(com, 4).tolist(),
                                 pad=np.round(pads.mean(0), 4).tolist(), minJ=round(pkg.min_jacobian(q), 3)))
            # --- advance the task state
            if state == 'close':
                closer(PERIOD, [f0, f1])
                gap = (s_plus - closer.s[0]) + (s_minus - closer.s[1]) - 2 * task.PAD_HALF[1]
                for i, s0 in ((0, s_plus), (1, s_minus)):
                    if closer.touch[i] is None and closer.s[i] > s0:
                        events['grasp_fail'] = f'pad {i} passed the grasp point without touching'
                if 'grasp_fail' in events:
                    break
                if gap < .002 and None not in closer.touch:
                    closer.done = True
                if t - sc.settle_s > task.CLOSE_TIMEOUT and not closer.done:
                    closer.done = True  # lift with whatever grip there is
                    info['close_timeout'] = True
                if closer.done:
                    state, t_closed = 'carry', t
                    starts, offset, total = task.carry_schedule(p)
                    z_closed = com[2]
                    info.update(close_end_s=round(t, 3), touch_travel_mm=[None if x is None else round(x * 1e3, 2) for x in closer.touch],
                                squeeze_mm=round(closer.squeeze * 1e3, 2), closed_grip_N=round(grip, 3))
                    i = int(np.argmin(np.linalg.norm(q - pads[0], axis=1)))
                    track = (i, q[i] - pads[0])
            else:
                tc = t - t_closed
                if tc >= starts['lift'][0]:
                    slip = max(slip, float(np.linalg.norm(q[track[0]] - pads[0] - track[1])))
                if starts['hold_lift'][1] - PERIOD / 2 <= tc < starts['hold_lift'][1] + PERIOD / 2:
                    lift_height = com[2] - z_closed
                    if lift_height < .5 * task.LIFT_H:
                        events['grasp_fail'] = f'centre of mass rose {lift_height * 1e3:.0f} mm of {task.LIFT_H * 1e3:.0f}'
                        break
                if starts['move'][0] <= tc < starts['lower'][0]:
                    lost = lost + PERIOD if max(f0, f1) < task.F_TOUCH else 0.
                    if lost >= task.LOST_CONTACT_S or com[2] < z_closed + .25 * task.LIFT_H:
                        events['drop'] = f'fell from the pads at {tc:.2f} s after closing ({phase_of(starts, tc)})'
                        break
    except SimFail as err:
        events['sim_fail'] = str(err)
    wp.synchronize()
    final_com = pkg.com() if 'sim_fail' not in events else np.full(3, np.nan)
    outcome, detail = task.classify(events, final_com[:2], target, transport)
    info.update(sim_s=round(t, 3), build_s=round(t_build, 2), final_com=final_com.tolist(), target=target.tolist())
    result = dict(outcome=outcome, max_grip_force_N=round(max_grip, 3), slip_mm=round(slip * 1e3, 2),
                  lift_height_mm=round(lift_height * 1e3, 2), wall_s=round(time.perf_counter() - t_wall, 2),
                  detail=detail, **info)
    if trace:
        result['trace'] = rows
    return result


def phase_of(starts, tc):
    return next((k for k, (a, b) in starts.items() if a <= tc < b), 'end')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('object', choices=sorted(OBJECTS, key=lambda k: int(k[1:])))
    ap.add_argument('--params', required=True, help='JSON grasp parameters (task.PARAM_RANGES)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--trace', action='store_true')
    args = ap.parse_args()
    wp.config.quiet = True
    print(json.dumps(evaluate(args.object, json.loads(args.params), args.seed, args.trace)), flush=True)


if __name__ == '__main__':
    main()

"""v1.3 GPU backend acceptance G1--G5 (REPORT: "v1.3 冻结审阅", "GPU 对照标准的统一修订").

G1 runs the warp_v13 kernels on the CPU device on the exact states of CPU v1.3 top grasps and compares
them with the CPU v1.3 code. Use a new run-id under flock; results go to results/warp_backend_v1_3/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import sys

CONTENT = Path(__file__).resolve().parents[2] / "contents/segment"
sys.path.insert(0, str(CONTENT))
from source_manifest import source_paths, source_name


ROOT = CONTENT
BASE = ROOT / 'results/warp_backend_v1_3'
TOL = 1e-12


def save(path, data):
    path.write_text(json.dumps(data, indent=1, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))


def metadata(out, args):
    import mujoco
    import warp as wp
    import mujoco_warp
    paths = source_paths(ROOT)
    save(out / 'metadata.json', dict(arguments=vars(args), mujoco=mujoco.__version__, warp=wp.__version__,
         mujoco_warp=getattr(mujoco_warp, '__version__', None),
         source_sha256={source_name(p, ROOT): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))


# ---------------------------------------------------------------- G1: same-state operator checks
class CpuKernels:
    """warp_v13 kernels on the CPU device, fed from one MuJoCo CPU state."""

    def __init__(self, sim, binding):
        import warp as wp
        import warp_v13 as K
        self.wp, self.K, self.sim, self.binding = wp, K, sim, binding
        b = sim.bushings
        f = lambda v, t=wp.float64: wp.array(np.asarray(v), dtype=t, device='cpu')
        g = lambda v, t=wp.float64: wp.array(np.asarray(v)[None], dtype=t, device='cpu')
        self.topology = [f(b.a, int), f(b.b, int), f(b.node_a, int), f(b.node_b, int), f(b.axis, int),
                         f(b.weight_a), f(b.weight_b)]
        self.material = [g(b.ra, wp.vec3d), g(b.kt, wp.vec3d), g(b.kr, wp.vec3d), g(b.tension), g(b.compression_travel)]
        self.limits = [f(b.a, int), f(b.b, int), f(b.axis, int), g(b.ra, wp.vec3d), g(b.compression_limit),
                       g(np.where(np.isfinite(b.limit), b.limit, 1e300), wp.vec3d)]
        self.scratch = [wp.zeros((1, b.n), dtype=wp.vec3d, device='cpu') for _ in range(3)] + [
            wp.zeros((1, b.n), dtype=wp.float64, device='cpu'), wp.zeros((1, b.node_count), dtype=wp.vec3d, device='cpu'),
            wp.zeros((1, b.n), dtype=wp.float64, device='cpu')]
        m = sim.m
        pad = np.full(m.ngeom, -1)
        for k, p in enumerate(sim.pad_names):
            pad[sorted(sim.pad_geoms[p])] = k
        self.pad_of_geom = f(pad, int)
        self.content = f(sim._obj_geom_mask, bool)
        self.bodies = f(sim.obj_bodies, int)
        self.mass = g(m.body_mass[sim.obj_bodies])
        self.pad_qadr = wp.array(np.array([sim.pad_q[p] for p in sim.pad_names]), dtype=int, device='cpu')
        top = binding.top
        self.top = f(top, int)
        self.seg_bodies = f(sim.seg_bodies, int)
        self.geom_body = f(m.geom_bodyid, int)
        self.half = g(np.array([s.half for s in sim.desc.segments]), wp.vec3d)
        self.connects = f([binding.equations[i] for i in top], int)
        conn = np.zeros(m.neq, int)
        conn[list(binding.equations.values())] = 1
        self.is_connect = f(conn, int)

    def state(self):
        wp, d = self.wp, self.sim.d
        v = lambda a, t=wp.float64: wp.array(np.ascontiguousarray(a)[None], dtype=t, device='cpu')
        return dict(xpos=v(d.xpos, wp.vec3d), xmat=v(d.xmat.reshape(-1, 3, 3), wp.mat33d), xquat=v(d.xquat, wp.vec4d),
                    xipos=v(d.xipos, wp.vec3d), qpos=v(d.qpos), qpos32=v(d.qpos, float), qvel32=v(d.qvel, float))

    def contacts(self):
        wp, d = self.wp, self.sim.d
        n = d.ncon
        c = d.contact
        address = np.full((max(n, 1), 3), -1)
        for row in range(3):
            ok = (c.efc_address[:n] >= 0) & (row < c.dim[:n])
            address[:n, row] = np.where(ok, c.efc_address[:n] + row, -1)
        geom = np.zeros((max(n, 1), 2), int); geom[:n] = c.geom[:n]
        frame = np.zeros((max(n, 1), 9)); frame[:n] = c.frame[:n]
        dist = np.zeros(max(n, 1)); dist[:n] = c.dist[:n]
        return [wp.array([n], dtype=int, device='cpu'), wp.array(geom[None], dtype=int, device='cpu'),
                wp.array(frame[None], dtype=wp.float64, device='cpu'), wp.array(address[None], dtype=int, device='cpu'),
                wp.array(dist[None], dtype=wp.float64, device='cpu')]

    def face(self, s):
        wp, K = self.wp, self.K
        xfrc = wp.zeros((1, self.sim.m.nbody, 6), dtype=wp.float64, device='cpu')
        wp.launch(K.face_forces_env, dim=1, device='cpu', inputs=[*self.topology, *self.material, self.sim.bushings.centered,
                  s['xpos'], s['xmat'], s['xquat'], s['xipos'], xfrc, *self.scratch])
        return xfrc.numpy()[0], self.scratch[-1].numpy()[0]

    def failure(self, s):
        wp, K = self.wp, self.K
        out = wp.zeros((1, 4), dtype=wp.float64, device='cpu')
        wp.launch(K.failure_metrics, dim=1, device='cpu', inputs=[*self.limits, s['xpos'], s['xmat'], s['xquat'],
                  s['qpos32'], s['qvel32'], out])
        return out.numpy()[0]

    def readouts(self, s, cts):
        wp, K = self.wp, self.K
        out = wp.zeros((1, 21), dtype=wp.float64, device='cpu')
        wp.launch(K.readouts, dim=1, device='cpu', inputs=[cts[0], cts[1], cts[2], cts[3], cts[4],
                  wp.array(self.sim.d.efc_force[None], dtype=wp.float64, device='cpu'), self.pad_of_geom, self.content,
                  self.sim.table_geom, self.bodies, self.mass, s['xipos'], s['qpos'], self.pad_qadr, out])
        return out.numpy()[0]

    def pinch(self, s, cts, tips, original_top):
        wp, K = self.wp, self.K
        efc = wp.array(self.sim.d.efc_force[None], dtype=wp.float64, device='cpu')
        contacts = wp.zeros((1, 2, len(self.binding.top)), dtype=int, device='cpu')
        wp.launch(K.pinch_top_contacts, dim=1, device='cpu', inputs=[self.top, self.seg_bodies, self.geom_body,
                  self.pad_of_geom, cts[0], cts[1], cts[2], cts[3], efc, s['xmat'], contacts])
        covered = wp.zeros((1, len(self.binding.top)), dtype=int, device='cpu')
        eligible = wp.zeros(1, dtype=int, device='cpu')
        wp.launch(K.pinch_candidate, dim=1, device='cpu', inputs=[wp.array([1], dtype=int, device='cpu'), self.top,
                  self.seg_bodies, self.half, contacts, s['xpos'], s['xmat'],
                  wp.array(np.asarray(tips)[None], dtype=wp.vec3d, device='cpu'),
                  wp.array([original_top], dtype=wp.float64, device='cpu'), 0.003, 0.020, covered, eligible])
        return contacts.numpy()[0], covered.numpy()[0], int(eligible.numpy()[0])

    def bind(self, s, covered, point):
        wp, K = self.wp, self.K
        m, d = self.sim.m, self.sim.d
        eq_data = wp.array(m.eq_data.copy()[None], dtype=wp.float64, device='cpu')
        active = wp.array(d.eq_active.astype(bool).copy()[None], dtype=bool, device='cpu')
        wp.launch(K.pinch_bind, dim=1, device='cpu', inputs=[wp.array([1], dtype=int, device='cpu'), self.top,
                  self.seg_bodies, self.connects, wp.array(covered[None], dtype=int, device='cpu'), s['xpos'], s['xmat'],
                  wp.array(np.asarray(point)[None], dtype=wp.vec3d, device='cpu'), eq_data, active])
        return eq_data.numpy()[0], active.numpy()[0]

    def reaction(self):
        wp, K, d = self.wp, self.K, self.sim.d
        out = wp.zeros((1, 4), dtype=wp.float64, device='cpu')
        wp.launch(K.pinch_reaction, dim=1, device='cpu', inputs=[wp.array([d.nefc], dtype=int, device='cpu'), 0,
                  wp.array(d.efc_type[None].astype(np.int32), dtype=int, device='cpu'),
                  wp.array(d.efc_id[None].astype(np.int32), dtype=int, device='cpu'),
                  wp.array(d.efc_force[None], dtype=wp.float64, device='cpu'), self.is_connect,
                  wp.array(d.eq_active.astype(bool)[None], dtype=bool, device='cpu'), out])
        return out.numpy()[0]


def g1_case(job):
    """One CPU v1.3 top grasp; every 10 ms the kernels are compared on the same state."""
    import mujoco
    import model
    import top_grasp as T
    from pinch import PinchAttachment
    from bushing import rotate  # noqa: F401  (Bushings.coordinates)
    worst, counts, failures = {}, {}, []

    def record(name, delta, discrete=True, **where):
        worst[name] = max(worst.get(name, 0.), float(delta))
        counts[name] = counts.get(name, 0) + 1
        if not discrete or not np.isfinite(delta) or delta > TOL:
            failures.append(dict(check=name, delta=float(delta), discrete=bool(discrete), **where))

    state = {}

    def observe(sim, binding, phase, t, force):
        if sim.desc.rigid:
            return
        if 'k' not in state:
            state['k'] = CpuKernels(sim, binding)
            state['probe'] = PinchAttachment(sim, binding.original_top, T.WIDTH)
        k, probe = state['k'], state['probe']
        where = dict(phase=phase, t=round(t, 4))
        s, cts = k.state(), k.contacts()
        d, m = sim.d, sim.m
        saved = d.xfrc_applied.copy()
        d.xfrc_applied[:] = 0.
        sim.bushings.apply(m, d)
        ref_xfrc, ref_rho = d.xfrc_applied.copy(), sim.bushings.orthogonal_compaction.copy()
        d.xfrc_applied[:] = saved
        xfrc, rho = k.face(s)
        record('face_forces', max(np.abs(xfrc - ref_xfrc).max(), np.abs(rho - ref_rho).max()), **where)
        b = sim.bushings
        *_, u, phi = b.coordinates(d)
        strain = -u[b.row, b.axis] / (2 * b.ra[b.row, b.axis])
        ref_fail = np.array([strain.max(), (strain / b.compression_limit).max(), (np.abs(phi) / b.limit).max(),
                             float(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())])
        record('failure_metrics', np.abs(k.failure(s) - ref_fail).max(), **where)
        pads, table = T.contact_reactions(sim, T.reaction_index(sim))
        ref = np.r_[pads['pA'], pads['pB'], table, len(T.table_contacts(sim)),
                    max(sim.pad_normal_force('pA', a) for a in range(3)), max(sim.pad_normal_force('pB', a) for a in range(3)),
                    sim.com(), sim.pad_pos('pA') - sim.pad_p0['pA'], sim.pad_pos('pB') - sim.pad_p0['pB']]
        record('readouts', np.abs(k.readouts(s, cts) - ref).max(), **where)
        tips = np.array([sim.pad_pos(p) - [0, 0, T.HEIGHT / 2] for p in sim.pad_names])
        contacts, covered, eligible = k.pinch(s, cts, tips, binding.original_top)
        ref_contacts = probe.top_contacts()
        mine = tuple({binding.top[i] for i in np.flatnonzero(row)} for row in contacts)
        record('pinch_top_contacts', 0., mine == ref_contacts, **where)
        probe.active, probe.broken = None, False
        ref_candidate = probe.candidate(tips)
        mine_covered = sorted(binding.top[i] for i in np.flatnonzero(covered))
        record('pinch_candidate', 0., (ref_candidate is None and not eligible) or
               (ref_candidate is not None and eligible == 1 and sorted(ref_candidate) == mine_covered), **where)
        if ref_candidate is not None:
            point = tips.mean(0)
            eq_before, active_before, mocap_before = m.eq_data.copy(), d.eq_active.copy(), d.mocap_pos.copy()
            eq, active = k.bind(s, covered, point)
            probe.set_frame(point)
            probe.bind_segments(ref_candidate)
            record('pinch_bind', np.abs(eq - m.eq_data).max(), np.array_equal(active, d.eq_active.astype(bool)), **where)
            m.eq_data[:], d.eq_active[:], d.mocap_pos[:] = eq_before, active_before, mocap_before
            probe.active, probe.segments = None, []
        if binding.active is not None:
            reaction = k.reaction()
            ref_force = binding.force_vector()
            rows = int(reaction[3])
            record('pinch_reaction', np.abs(reaction[:3] - ref_force).max(), rows in (0, 3 * len(binding.active)), **where)
            state['bound_samples'] = state.get('bound_samples', 0) + 1

    t0 = time.perf_counter()
    k = model.ANCHORS[job['category']]
    result = T.evaluate(k, job['grasp'], job['seed'], observer=observe)
    return dict(job=job, outcome=result['outcome'], events=[(e['event'], e['t']) for e in result['pinch_events']],
                worst=worst, counts=counts, failures=failures[:20], n_failures=len(failures),
                bound_samples=state.get('bound_samples', 0), wall_s=time.perf_counter() - t0)


def g1_filter_and_spring():
    """Host float64 low-pass release (warp_v13.LoadFilter) and spring vs pinch.py on scripted sequences."""
    from unittest.mock import patch
    import model
    from mj_adapter import OMEGA_MAX, Sim, scene
    from pinch import PinchAttachment, PinchSpring, fragments, GATHER_TRAVEL, GRIP_STIFFNESS
    from warp_v13 import LoadFilter
    desc = model.build(model.ANCHORS['bendable'], omega_max=OMEGA_MAX)
    body, eq = fragments(desc)
    sim = Sim(scene(desc, (0, 0, .2), table=False, gravity=False, worldbody=body, equality=eq))
    ref = PinchAttachment(sim, .2 + desc.kappa.size[2] / 2)
    ref.bind_segment(ref.top[0])
    dt = sim.m.opt.timestep
    filt = LoadFilter(1, .005)
    rng = np.random.default_rng(20261107)
    rows, worst = 0, 0.
    for step in range(400):
        force = np.array([0., 0., float(rng.choice([5., 9.99, 10., 10.01, 11., 30.]))])
        grip = float(rng.choice([10., 10.5, 12.]))
        with patch.object(ref, 'force_vector', return_value=force):
            released = ref.check_load(dt, grip)
        mine = bool(filt.update(dt, np.array([np.linalg.norm(force)]), np.array([ref.coefficient * grip]), np.array([True]))[0])
        worst = max(worst, abs(filt.load[0] - ref.load))
        rows += 1
        if released != mine:
            return dict(passed=False, step=step, rows=rows)
        if released:
            ref.active, ref.broken = None, False
            ref.bind_segment(ref.top[0]); filt.reset(np.array([True]))
    spring, worst_spring = PinchSpring(), 0.
    for eligible in (False, True):
        for gap0 in (.020, .040, .095):
            spring.arm(gap0, eligible)
            for gap in (gap0, gap0 - .010 + 1e-12, gap0 - .010, gap0 - .010 - 1e-12, gap0 - .015):
                mine = GRIP_STIFFNESS * max(gap0 - GATHER_TRAVEL - gap, 0.) if eligible else 0.
                worst_spring = max(worst_spring, abs(mine - spring.force(gap)))
            spring.release()
    return dict(passed=worst <= TOL and worst_spring <= TOL, filter_steps=rows, filter_max_abs_diff=worst,
                spring_max_abs_diff=worst_spring)


G1_JOBS = [dict(category='bendable', grasp=dict(w=.04, d=.015, F=10.), seed=0),
           dict(category='bendable', grasp=dict(w=.08, d=.02, F=15.), seed=1),
           dict(category='bendable', grasp=dict(u=.25, w=.04, d=.01, F=15.), seed=2),
           dict(category='compressible', grasp=dict(w=.04, d=.015, F=10.), seed=0),
           dict(category='compressible', grasp=dict(v=.3, w=.04, d=.01, F=10.), seed=1)]


def g1(out, workers):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        rows = pool.map(g1_case, G1_JOBS)
    host = g1_filter_and_spring()
    worst = {}
    for r in rows:
        for k, v in r['worst'].items():
            worst[k] = max(worst.get(k, 0.), v)
    passed = all(r['n_failures'] == 0 for r in rows) and host['passed'] and all(r['bound_samples'] for r in rows)
    save(out / 'summary.json', dict(passed=passed, tolerance=TOL, worst=worst, host=host, cases=rows))
    print('G1 passed', passed, json.dumps(worst), host, [(r['job']['category'], r['outcome'], r['n_failures'],
          r['bound_samples']) for r in rows], flush=True)
    return passed


# ---------------------------------------------------------------- G2: responses
KAPPA_METRICS = {'r1y': ('stiffness', 'lock_strain'), 'r1z': ('stiffness', 'lock_strain'), 'r2': ('droop',),
                 'r4': ('measured_pinch_coeff',)}
RESOLUTION = {'lock_strain': .0025, 'com_decay_s': .01}


def compare_kappa(category, test, a, b, tolerance=.02):
    """Layer 1 (soft R1/R2/R4 <= 2 %, discrete readings by resolution); hard objects are recorded only."""
    rows = []
    for key in KAPPA_METRICS[test]:
        av, bv = a.get(key), b.get(key)
        if av is None or bv is None:
            passed, delta, allowed = av is None and bv is None, None, None
        else:
            delta = abs(bv - av)
            allowed = max(tolerance * abs(av), RESOLUTION.get(key, 0.)) if av else max(1e-12, RESOLUTION.get(key, 0.))
            passed = bool(np.isfinite(delta) and (delta <= allowed or np.isclose(delta, allowed, rtol=8 * np.finfo(float).eps, atol=0)))
        record_only = category == 'hard' and test != 'r4'
        rows.append(dict(metric=key, reference=av, candidate=bv, abs_diff=delta, allowed=allowed,
                         relative=(delta / abs(av) if av and delta is not None else None),
                         resolution_used=bool(av and delta is not None and delta > tolerance * abs(av)),
                         record_only=record_only, passed=passed or record_only))
    return rows


def compare_dynamic(a, b):
    """R3 dynamic layer (recorded): slip <= 1 mm, COM swing <= 20 %, hold mean grip <= 5 %, peak-to-peak recorded."""
    rel = lambda x, y: abs(y - x) / abs(x) if x else (0. if y == x else float('inf'))
    rows = [dict(metric='slip_mm', reference=a['slip_mm'], candidate=b['slip_mm'], abs_diff=abs(b['slip_mm'] - a['slip_mm']),
                 limit='<= 1 mm', within=abs(b['slip_mm'] - a['slip_mm']) <= 1.),
            dict(metric='com_swing_mm', reference=a['com_swing_mm'], candidate=b['com_swing_mm'],
                 relative=rel(a['com_swing_mm'], b['com_swing_mm']), limit='<= 20 %',
                 within=rel(a['com_swing_mm'], b['com_swing_mm']) <= .20),
            dict(metric='grip_force', reference=a['grip_force'], candidate=b['grip_force'],
                 relative=rel(a['grip_force'], b['grip_force']), limit='<= 5 %', within=rel(a['grip_force'], b['grip_force']) <= .05),
            dict(metric='force_p2p', reference=a.get('force_p2p'), candidate=b.get('force_p2p'), limit='record only'),
            dict(metric='com_decay_s', reference=a.get('com_decay_s'), candidate=b.get('com_decay_s'), limit='record only')]
    return rows


def g2_response(job):
    import warp as wp
    wp.config.quiet = True
    import model
    import protocols as P
    from mjwarp_backend import DiagnosticSim
    k, test = model.ANCHORS[job['category']], job['test']
    run = lambda: P.r1(k, test[-1]) if test.startswith('r1') else P.r2(k) if test == 'r2' else P.r3(k)
    t0 = time.perf_counter()
    reference = run()
    t1 = time.perf_counter()
    original = P.Sim
    P.Sim = DiagnosticSim
    try:
        candidate = json.loads(json.dumps(run()), parse_constant=lambda v: v)
    finally:
        P.Sim = original
    row = dict(job=job, reference=reference, candidate=candidate, cpu_wall_s=t1 - t0, gpu_wall_s=time.perf_counter() - t1,
               gpu_fail=candidate.get('fail'), cpu_fail=reference.get('fail'))
    if test == 'r3':
        row['dynamic'] = compare_dynamic(reference, candidate)
    else:
        row['kappa'] = compare_kappa(job['category'], test, reference, candidate)
    row['flags'] = {key: dict(reference=reference.get(key), candidate=candidate.get(key))
                    for key in ('fail', 'capped', 'lower_bound', 'slipped', 'dropped', 'suspended', 'table_touch')
                    if key in reference or key in candidate}
    return row


def g2_r4():
    import warp as wp
    wp.config.quiet = True
    import model
    import top_grasp as T
    import mjwarp_v13 as G
    cats = ('hard', 'bendable', 'compressible')
    specs = [dict(kappa=model.ANCHORS[c], grasp=dict(w=.04, d=.015, F=10.), seed=0, pull=True) for c in cats]
    gpu = G.run_specs(specs)
    rows = []
    for c, r in zip(cats, gpu):
        ref = T.measure_r4(model.ANCHORS[c], trace=False)
        rows.append(dict(job=dict(category=c, test='r4'), reference={k: ref[k] for k in ('outcome', 'reason', 'measured_pinch_coeff', 'pull_force_N', 'pinch_events', 'max_grip_force_N')},
                         candidate=r, gpu_fail=r['outcome'] == 'sim_fail', cpu_fail=ref['outcome'] == 'sim_fail',
                         kappa=compare_kappa(c, 'r4', ref, r) if c != 'hard' else
                         [dict(metric='measured_pinch_coeff', reference=ref['measured_pinch_coeff'], candidate=r['measured_pinch_coeff'],
                               outcome=(ref['outcome'], r['outcome']), record_only=False,
                               passed=ref['measured_pinch_coeff'] == r['measured_pinch_coeff'] == 0 and r['outcome'] == ref['outcome'])]))
    return rows


def g2(out, workers):
    from multiprocessing import get_context
    jobs = [dict(category=c, test=t) for c in ('hard', 'bendable', 'compressible') for t in ('r1y', 'r1z', 'r2', 'r3')]
    with get_context('spawn').Pool(workers) as pool:
        r4 = pool.apply_async(g2_r4)
        rows = pool.map(g2_response, jobs)
        rows += r4.get()
    kappa_ok = all(m['passed'] for r in rows for m in r.get('kappa', []))
    stable = not any(r['gpu_fail'] for r in rows)
    passed = kappa_ok and stable
    save(out / 'summary.json', dict(passed=passed, kappa_layer_passed=kappa_ok, stability_passed=stable, rows=rows))
    for r in rows:
        j = r['job']
        detail = r.get('kappa') or r.get('dynamic')
        print(j['category'], j['test'], 'gpu_fail', r['gpu_fail'], [(m['metric'], m.get('reference'), m.get('candidate'),
              m.get('passed', m.get('within'))) for m in detail], flush=True)
    print('G2 passed', passed, 'kappa', kappa_ok, 'stable', stable, flush=True)
    return passed


# ---------------------------------------------------------------- G3/G4/G5: batched executor
V13 = ROOT / 'results/pinch_connect_v1_3'
S2_MANIFEST = ROOT / 'results/cpu_equivalence_v1_2_1/s2_bushing_regression/manifest.json'
E4_SEEDS = (20261116, 20261117, 20261118)


def e4_specs():
    """The 30 S2 E4 combinations (kappa incl. scaled pinch_coeff, w/d/F) x three seeds, with the CPU v1.3 P4 row."""
    from model import Kappa
    combos = [j for j in json.loads(S2_MANIFEST.read_text())['jobs'] if j['test'] == 'e4']
    cpu = {}
    for r in json.loads((V13 / 'r2_p4_e4/summary.json').read_text())['rows']:
        g = r['grasp']
        cpu[(r['category'], g['w'], g['d'], g['F'], r['seed'], r['kappa']['pinch_coeff'])] = r
    specs = []
    for j in combos:
        for seed in E4_SEEDS:
            g = j['grasp']
            ref = cpu[(j['category'], g['w'], g['d'], g['F'], seed, j['kappa']['pinch_coeff'])]
            specs.append(dict(kappa=Kappa.from_dict(j['kappa']), grasp=dict(g), seed=seed,
                              label=dict(set='E4', category=j['category'], original_index=j['original_index'], seed=seed),
                              cpu=ref['result']))
    return specs


def e5_specs(count=50):
    from model import Kappa
    rows = sorted(json.loads((V13 / 'r2_p5_e5/summary.json').read_text())['rows'], key=lambda r: r['index'])[:count]
    return [dict(kappa=Kappa.from_dict(r['kappa']), grasp=dict(r['grasp']), seed=r['seed'],
                 label=dict(set='E5', category=r['category'], index=r['index']), cpu=r['result']) for r in rows]


def same_bucket_specs(count, seed=20261203):
    """count random kappa (sample_kappa 'mixed', new seed) that fall into the most frequent soft bucket, with
    E5-style random grasps: heterogeneous kappa in one batch."""
    from sampling import sample_kappa
    rng = np.random.default_rng(seed)
    pool = {}
    while True:
        k = sample_kappa(rng, 'mixed')
        grasp = dict(u=float(rng.uniform(-.5, .5)), v=float(rng.uniform(-.5, .5)), psi=float(rng.uniform(0, np.pi)),
                     w=float(rng.uniform(.020, .095)), d=float(rng.uniform(0, .040)), F=float(rng.uniform(2, 50)),
                     v_lift=float(rng.uniform(.05, .3)), a_max=float(rng.uniform(.5, 4)))
        if k.rigid:
            continue
        key = (tuple(k.segments), k.mobile)
        pool.setdefault(key, []).append((k, grasp))
        if len(pool[key]) == count:
            return [dict(kappa=k, grasp=g, seed=seed + i, label=dict(set='hetero', bucket=str(key), index=i))
                    for i, (k, g) in enumerate(pool[key])]


def cpu_evaluate(spec):
    import top_grasp as T
    r = T.evaluate(spec['kappa'], spec['grasp'], spec['seed'])
    return {k: r[k] for k in ('outcome', 'reason', 'max_grip_force_N', 'pinch_events', 'wall_s', 'sim_s', 'failure_state')}


def gpu_run(specs, progress_every=None):
    import warp as wp
    wp.config.quiet = True
    import mjwarp_v13 as G
    return G.run_specs([{k: v for k, v in s.items() if k != 'cpu'} for s in specs])


def compare_tasks(specs, gpu):
    rows = []
    for s, g in zip(specs, gpu):
        c = s['cpu']
        rows.append(dict(label=s['label'], kappa=s['kappa'].to_dict(), grasp=s['grasp'], seed=s['seed'],
                         cpu_outcome=c['outcome'], gpu_outcome=g['outcome'], agree=c['outcome'] == g['outcome'],
                         cpu_reason=c['reason'], gpu_reason=g['reason'], cpu_force=c['max_grip_force_N'],
                         gpu_force=g['max_grip_force_N'], gpu_failure_state=g['failure_state'],
                         gpu_events=g['pinch_events'], bucket=g['bucket']))
    agree = sum(r['agree'] for r in rows) / len(rows)
    hard = [r for r in rows if r['label']['category'] == 'hard']
    force = None
    if hard:
        a, b = np.median([r['cpu_force'] for r in hard]), np.median([r['gpu_force'] for r in hard])
        force = dict(cpu_median=float(a), gpu_median=float(b), relative=float(abs(b - a) / a), hard_pairs=len(hard),
                     hard_agreement=sum(r['agree'] for r in hard) / len(hard))
    return dict(rows=rows, agreement=agree, hard_force=force,
                gpu_sim_fail=sum(r['gpu_outcome'] == 'sim_fail' for r in rows),
                cpu_sim_fail=sum(r['cpu_outcome'] == 'sim_fail' for r in rows))


def g3(out, workers):
    from multiprocessing import get_context
    sets = dict(E4=e4_specs(), E5=e5_specs())
    summary = {}
    with get_context('spawn').Pool(workers) as pool:
        for name, specs in sets.items():
            # one process per structural bucket (heterogeneous kappa share a bucket)
            import mjwarp_v13 as G
            groups = list(G.buckets(specs).values())
            parts = pool.map(gpu_run, [[specs[i] for i in idx] for idx in groups])
            gpu = [None] * len(specs)
            for idx, part in zip(groups, parts):
                for i, r in zip(idx, part):
                    gpu[i] = r
            summary[name] = compare_tasks(specs, gpu)
            summary[name]['buckets'] = [len(idx) for idx in groups]
    ok_e4 = summary['E4']['agreement'] >= .95 and summary['E4']['hard_force']['relative'] <= .10 \
        and summary['E4']['hard_force']['hard_agreement'] >= .95
    ok_e5 = summary['E5']['agreement'] >= .95
    hard5 = summary['E5']['hard_force']
    if hard5:
        ok_e5 = ok_e5 and hard5['relative'] <= .10 and hard5['hard_agreement'] >= .95
    gpu_fail = summary['E4']['gpu_sim_fail'] + summary['E5']['gpu_sim_fail']
    stable = gpu_fail / (len(sets['E4']) + len(sets['E5'])) <= .01
    summary.update(passed=ok_e4 and ok_e5 and stable, task_layer_passed=ok_e4 and ok_e5, stability_passed=stable)
    save(out / 'summary.json', summary)
    for name in sets:
        x = summary[name]
        print(name, 'agreement %.4f' % x['agreement'], 'hard', x['hard_force'], 'gpu sim_fail', x['gpu_sim_fail'],
              'cpu sim_fail', x['cpu_sim_fail'], 'buckets', x['buckets'], flush=True)
        for r in x['rows']:
            if not r['agree']:
                print('   disagree', r['label'], r['cpu_outcome'], '->', r['gpu_outcome'], r['gpu_reason'], flush=True)
    print('G3 passed', summary['passed'], flush=True)
    return summary['passed']


def g4(out, workers, count=64):
    """Heterogeneous kappa in one batch; CPU v1.3 reference on the same inputs (28 processes)."""
    from multiprocessing import get_context
    specs = same_bucket_specs(count)
    with get_context('spawn').Pool(min(28, os.cpu_count())) as pool:
        cpu = pool.map(cpu_evaluate, specs)
    for s, c in zip(specs, cpu):
        s['cpu'] = c
    gpu = gpu_run(specs)
    summary = compare_tasks(specs, gpu)
    summary['bucket'] = specs[0]['label']['bucket']
    summary['batched_fields'] = gpu[0]['batched_fields']
    summary['passed'] = summary['gpu_sim_fail'] / len(specs) <= .01
    save(out / 'summary.json', summary)
    print('G4 hetero bucket', summary['bucket'], 'n', len(specs), 'agreement %.4f' % summary['agreement'],
          'gpu sim_fail', summary['gpu_sim_fail'], 'cpu sim_fail', summary['cpu_sim_fail'],
          'batched fields', len(summary['batched_fields']), 'passed', summary['passed'], flush=True)
    return summary['passed']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('g1', 'g2', 'g3', 'g4'))
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--workers', type=int, default=5)
    args = ap.parse_args()
    out = BASE / args.run_id
    out.mkdir(parents=True)
    metadata(out, args)
    os.environ.setdefault('WARP_CACHE_PATH', str(Path.home() / '.cache/warp'))
    ok = {'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4}[args.action](out, args.workers)
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())

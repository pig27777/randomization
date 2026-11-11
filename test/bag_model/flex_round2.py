"""flex 提速第 2 轮（F8–F13）的批量入口：硬盒松装袋、F3 侧夹闭环流程，MJWarp。

与历史入口 flex_batch.py 的流程、阈值相同（控制器与判定逐行照抄），差别只有：
- 步长、求解器选项、确定性开关、环境数由命令行给出；10 ms 检查间隔按步长换算（0.8 ms 时仍为 12 步）；
- 逐环境保存穿透（膜顶点进盒深度、盒角出袋距离分开记）、峰值出现时刻和阶段、结果类别；
- 输出写到 --output 指定的本轮运行目录，不写 bag_sim/outputs/。
正式运行须排队：flock /tmp/claude-gpu.lock <命令>。
"""
import argparse
import json
import math
import socket
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

BAG = Path(__file__).resolve().parents[2] / "bag_sim"
sys.path.insert(0, str(BAG))

from flex_batch import BatchData, MemoryPeak, overflow_count, pkg_com, reactions
from flex_geometry import triangle_distance
from loose_protocols import GAP, _pad_reach, _rot, make, rotate_package
from mj_adapter import PAD_T
from round3_flex import BASE, content_kappa, f3_inputs

NAMES = {0: 'unfinished', 1: 'success', 2: 'grasp_fail', 3: 'drop', 4: 'sim_fail'}
# 第 3 轮 CPU 行为的最大值（693–840 行、231–280 个接触）乘 1.5，与 flex_batch 的取法相同
NJMAX, NCON = 1566, 522


@wp.kernel
def observe(d: BatchData, raw: wp.array2d(dtype=float), state: wp.array2d(dtype=float),
            phase: wp.array(dtype=int), outcome: wp.array(dtype=int), dt: float, check: int):
    w = wp.tid()
    if outcome[w] == 0:
        f = float(20.)
        for p in range(2):
            old = state[w, 7 + p]
            state[w, 7 + p] = old + dt / (.005 + dt) * (wp.min(raw[w, p], wp.max(2. * f, old + 20.)) - old)
        measured = state[w, 9] + dt / (.020 + dt) * (.5 * (raw[w, 0] + raw[w, 1]) - state[w, 9])
        state[w, 9] = measured
        state[w, 19] = wp.max(state[w, 19], measured)
        state[w, 0] = state[w, 0] + dt
        step = int(state[w, 11]) + 1
        state[w, 11] = float(step)
        if step % check == 0 and (phase[w] == 4 or phase[w] == 5):
            lost = float(0.)
            if wp.min(state[w, 7], state[w, 8]) < .05:
                lost = state[w, 10] + .010
            state[w, 10] = lost
            if lost >= .05:
                outcome[w] = 2 if phase[w] == 4 else 3
        if state[w, 0] > 30.:
            outcome[w] = 4


@wp.kernel
def controller(d: BatchData, state: wp.array2d(dtype=float), phase: wp.array(dtype=int),
               outcome: wp.array(dtype=int), previous: wp.array2d(dtype=float),
               start: wp.array(dtype=float), p0: wp.array(dtype=float),
               qa: wp.array(dtype=int), va: wp.array(dtype=int), raw: wp.array2d(dtype=float),
               obj: int, bag: wp.array(dtype=int), masses: wp.array(dtype=float), mass: float,
               travel: float, carry: wp.vec3, carry_t: float, dt: float, check: int):
    w = wp.tid()
    if outcome[w] != 0:
        if state[w, 22] < 0.:
            state[w, 22] = d.xipos[w, obj][2] * 1000.
        for j in range(6):
            d.qvel[w, va[j]] = 0.
        return
    t = state[w, 0]
    tau = t - state[w, 1]
    p = phase[w]
    next_p = p
    com = pkg_com(d, w, obj, bag, masses, mass)
    arm = wp.vec3(state[w, 15], state[w, 16], state[w, 17])
    if p == 0:
        if tau >= .3:
            for j in range(3): state[w, 12 + j] = com[j]
            state[w, 21] = d.xipos[w, obj][2]
            next_p = 1
    elif p == 1:
        for j in range(2):
            if state[w, 4 + j] < 0.:
                if state[w, 7 + j] > .05: state[w, 4 + j] = state[w, 2 + j]
                else: state[w, 2 + j] = state[w, 2 + j] + .025 * dt
        if state[w, 4] >= 0. and state[w, 5] >= 0.: next_p = 2
        elif tau > 5.: outcome[w] = 2
    elif p == 2:
        if .5 * (state[w, 7] + state[w, 8]) >= 19. or state[w, 6] >= travel or tau > 8.: next_p = 3
    elif p == 3:
        if tau >= .2: next_p = 4
    elif p == 4:
        arm = wp.vec3(0., 0., wp.min(.1 * tau, .1))
        if tau >= 1.2:
            state[w, 20] = (d.xipos[w, obj][2] - state[w, 21]) * 1000.
            if com[2] - state[w, 14] < .05: outcome[w] = 2
            else: next_p = 5
    elif p == 5:
        fraction = float(1.)
        if tau < carry_t: fraction = tau / carry_t - wp.sin(2. * wp.pi * tau / carry_t) / (2. * wp.pi)
        arm = wp.vec3(0., 0., .1) + fraction * carry
        if tau >= carry_t + .3:
            if com[2] - state[w, 14] < .03: outcome[w] = 3
            else: next_p = 6
    elif p == 6:
        if int(state[w, 11]) % check == 0 and raw[w, 2] > .3 * mass * 9.81: state[w, 18] = arm[2]
        if state[w, 18] < 0.:
            arm = wp.vec3(0., 0., wp.max(.1 - .2 * tau, .005)) + carry
            if arm[2] <= .005: state[w, 18] = .005
        if state[w, 18] >= 0.: next_p = 7
    elif p == 7:
        arm[2] = state[w, 18]
        opened = wp.min(2. * .025 * tau, .03)
        for j in range(2): state[w, 2 + j] = state[w, 4 + j] + state[w, 6] - opened
        if tau >= 1.1:
            offset = com - wp.vec3(state[w, 12], state[w, 13], state[w, 14]) - carry
            direction = carry / .5
            x = offset[0] * direction[0] + offset[1] * direction[1]
            y = -offset[0] * direction[1] + offset[1] * direction[0]
            outcome[w] = 1 if wp.abs(x) <= .1 and wp.abs(y) <= .1 else 3
    if next_p != p:
        phase[w] = next_p
        state[w, 1] = t
    p = next_p
    if p >= 2 and p <= 6:
        err = 20. - .5 * (state[w, 7] + state[w, 8])
        squeeze = wp.clamp(state[w, 6] + wp.clamp(.0025 * err, -.025, .025) * dt, -.005, travel)
        state[w, 6] = squeeze
        for j in range(2): state[w, 2 + j] = state[w, 4 + j] + squeeze
    for j in range(3): state[w, 15 + j] = arm[j]
    for j in range(6):
        side = j // 3
        axis = j % 3
        target = start[j] + arm[axis]
        if axis == 1: target = target + (1. - 2. * float(side)) * state[w, 2 + side]
        d.qpos[w, qa[j]] = previous[w, j] - p0[j]
        d.qvel[w, va[j]] = (target - previous[w, j]) / dt
        previous[w, j] = target


@wp.kernel
def penetration(vertices: wp.array2d(dtype=wp.vec3), faces: wp.array(dtype=wp.vec3i),
                xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33), obj: int, half: wp.vec3,
                state: wp.array2d(dtype=float), phase: wp.array(dtype=int), outcome: wp.array(dtype=int),
                check: int, nvert: int,
                peak: wp.array2d(dtype=float), when: wp.array2d(dtype=float), escaped: wp.array(dtype=int)):
    """与 flex_geometry 的两个 kernel 同一度量，每个环境一条线程，另记峰值的时刻和阶段。
    peak[:,0] 膜顶点进入盒体的深度；peak[:,1] 盒角在封闭袋面之外时到袋面的距离。"""
    w = wp.tid()
    if int(state[w, 11]) % check != 0 or outcome[w] != 0:
        return
    deepest = float(0.)
    for i in range(nvert):
        p = wp.transpose(xmat[w, obj]) * (vertices[w, i] - xpos[w, obj])
        inside = wp.min(half[0] - wp.abs(p[0]), wp.min(half[1] - wp.abs(p[1]), half[2] - wp.abs(p[2])))
        deepest = wp.max(deepest, inside)
    if deepest > peak[w, 0]:
        peak[w, 0] = deepest
        when[w, 0] = state[w, 0]
        when[w, 1] = float(phase[w])
    for i in range(9):
        local = wp.vec3(0., 0., 0.)
        if i < 8:
            local = wp.vec3((2. * float(i // 4) - 1.) * half[0], (2. * float((i // 2) % 2) - 1.) * half[1],
                            (2. * float(i % 2) - 1.) * half[2])
        p = xpos[w, obj] + xmat[w, obj] * local
        angle = float(0.)
        distance = float(100.)
        for j in range(faces.shape[0]):
            f = faces[j]
            a = vertices[w, f[0]]
            b = vertices[w, f[1]]
            c = vertices[w, f[2]]
            aa = a - p
            bb = b - p
            cc = c - p
            na = wp.length(aa)
            nb = wp.length(bb)
            nc = wp.length(cc)
            numerator = wp.dot(aa, wp.cross(bb, cc))
            denominator = na * nb * nc + wp.dot(aa, bb) * nc + wp.dot(bb, cc) * na + wp.dot(cc, aa) * nb
            angle = angle + 2. * wp.atan2(numerator, denominator)
            if i < 8: distance = wp.min(distance, triangle_distance(p, a, b, c))
        if wp.abs(angle) < 2. * wp.pi:
            if i == 8: escaped[w] = 1
            elif distance > peak[w, 1]:
                peak[w, 1] = distance
                when[w, 2] = state[w, 0]
                when[w, 3] = float(phase[w])
                when[w, 4] = float(i)


def prepare(bag, params, dt, cache):
    """flex_batch.prepare，步长和缓存目录可指定。"""
    sim = make(content_kappa('box_loose'), bag, [('pA', 1, (0, -1, .5)), ('pB', 1, (0, 1, .5))], big_table=True,
               dt=dt, cache_dir=cache)
    cxy, size, h = sim.footprint()
    u, v, psi, F, lv, accel = params
    d_obj = np.array([math.cos(psi), math.sin(psi)])
    e_obj = np.array([d_obj[1], -d_obj[0]])
    g = np.array([u, v]) * size
    ra = _pad_reach(g, -d_obj, e_obj, size / 2)
    rb = _pad_reach(g, d_obj, e_obj, size / 2)
    yaw = math.pi / 2 - math.atan2(d_obj[1], d_obj[0])
    rotate_package(sim, np.r_[cxy, 0.], yaw)
    gw = cxy + _rot(yaw) @ g
    starts = np.array([[gw[0], gw[1] - ra - GAP - PAD_T, h / 2], [gw[0], gw[1] + rb + GAP + PAD_T, h / 2]])
    for p, x in zip(sim.pad_names, starts): sim.set_pad(p, x, np.zeros(3))
    mujoco.mj_forward(sim.m, sim.d)
    return sim, starts, min(ra, rb)


def run(a, case, params, seed, out):
    dt = a.dt
    check = max(1, round(.010 / dt))
    bag = replace(BASE, tolerance=a.tolerance, iterations=a.iterations, ls_iterations=a.ls_iterations)
    sim, starts, travel = prepare(bag, params, dt, out / 'cache')
    nworld = a.nworld
    with MemoryPeak() as mem:
        wm = mjw.put_model(sim.m)
        wm.opt.deterministic = sum(int(getattr(mjw.DeterminismType, x)) for x in a.deterministic.split('+'))
        wd = mjw.put_data(sim.m, sim.d, nworld=nworld, nconmax=NCON, naconmax=NCON * nworld, njmax=NJMAX)
        bd = BatchData()
        bd.qpos, bd.qvel, bd.xipos = wd.qpos, wd.qvel, wd.xipos
        bd.nacon, bd.geom, bd.frame = wd.nacon, wd.contact.geom, wd.contact.frame
        bd.address, bd.worldid, bd.force = wd.contact.efc_address, wd.contact.worldid, wd.efc.force
        bd.njmax = wd.njmax
        state = np.zeros((nworld, 23), dtype=np.float32)
        state[:, 4:6] = -1; state[:, 18] = -1; state[:, 20] = np.nan; state[:, 22] = -1
        s = wp.array(state, dtype=float)
        phase = wp.zeros(nworld, dtype=int)
        outcome = wp.zeros(nworld, dtype=int)
        raw = wp.zeros((nworld, 3), dtype=float)
        prev = wp.array(np.tile(starts.ravel(), (nworld, 1)).astype(np.float32), dtype=float)
        start = wp.array(starts.ravel().astype(np.float32), dtype=float)
        p0 = wp.array(np.concatenate([sim.pad_p0[p] for p in sim.pad_names]).astype(np.float32), dtype=float)
        qa = wp.array(np.concatenate([sim.pad_q[p] for p in sim.pad_names]).astype(np.int32), dtype=int)
        va = wp.array(np.concatenate([sim.pad_v[p] for p in sim.pad_names]).astype(np.int32), dtype=int)
        lookup = np.full(sim.m.ngeom, -1, dtype=np.int32)
        for j, p in enumerate(sim.pad_names): lookup[list(sim.pad_geoms[p])] = j
        lookup = wp.array(lookup, dtype=int)
        bodies = wp.array(sim.bag_bodies.astype(np.int32), dtype=int)
        masses = wp.array(sim.m.body_mass.astype(np.float32), dtype=float)
        angle = float(np.random.default_rng(seed).uniform(0, 2 * math.pi))
        carry = wp.vec3(.5 * math.cos(angle), .5 * math.sin(angle), 0.)
        carry_t = math.sqrt(2 * math.pi * .5 / 2.)
        obj = sim.seg_bodies[0]
        faces = wp.array(sim.meta['faces'].astype(np.int32), dtype=wp.vec3i)
        peak = wp.zeros((nworld, 2), dtype=float)
        when = wp.zeros((nworld, 5), dtype=float)
        escaped = wp.zeros(nworld, dtype=int)
        overflow_bits = wp.zeros(nworld, dtype=int)
        overflow_events = wp.zeros(nworld, dtype=int)
        niter_sum = wp.zeros(nworld, dtype=int)
        half = wp.vec3(*sim.desc.segments[0].half)

        def step():
            wp.launch(controller, nworld, inputs=[bd, s, phase, outcome, prev, start, p0, qa, va, raw, obj, bodies,
                                                  masses, sim.pkg_mass, travel, carry, carry_t, dt, check])
            wd.overflow.zero_()
            mjw.step(wm, wd)
            wp.launch(overflow_count, nworld, inputs=[wd.overflow, overflow_bits, overflow_events])
            raw.zero_()
            wp.launch(reactions, wd.naconmax, inputs=[bd, lookup, sim.table_geom, raw, 1])
            wp.launch(observe, nworld, inputs=[bd, raw, s, phase, outcome, dt, check])
            wp.launch(penetration, nworld, inputs=[wd.flexvert_xpos, faces, wd.xpos, wd.xmat, obj, half, s, phase,
                                                   outcome, check, sim.m.nflexvert, peak, when, escaped])
        step(); wp.synchronize()
        with wp.ScopedCapture() as cap: step()
        wp.synchronize()
        t0 = time.perf_counter()
        executed = 2
        limit = round(30 / dt) + 2 * check
        nefc_max = ncon_max = 0
        iters = []
        while executed < limit:
            for _ in range(2 * check): wp.capture_launch(cap.graph)
            wp.synchronize(); executed += 2 * check
            codes = outcome.numpy()
            if executed % (100 * check) < 2 * check:      # 约每秒仿真时间抽查一次规模和迭代数，不进时间关键路径
                nefc_max = max(nefc_max, int(wd.nefc.numpy().max()))
                ncon_max = max(ncon_max, int(wd.nacon.numpy()[0]))
                iters.append(float(wd.solver_niter.numpy()[codes == 0].mean()) if (codes == 0).any() else 0.)
            if np.all(codes != 0) or np.any(overflow_bits.numpy() != 0): break
            if a.max_steps and executed >= a.max_steps: break
        wall = time.perf_counter() - t0
        codes = outcome.numpy(); states = s.numpy(); overflow = overflow_bits.numpy()
        finite = np.isfinite(wd.qpos.numpy()).all(axis=1)
        codes[~finite] = 4
        codes[escaped.numpy() != 0] = 4
        pk = peak.numpy() * 1e3
        invalid = int(np.count_nonzero(overflow))
        categories = {name: int((codes == code).sum()) for code, name in NAMES.items()}
        valid = invalid == 0 and bool(finite.all()) and bool(np.all(codes != 0))
        tag = f'{case}_s{seed}_n{nworld}_{a.tag}'
        np.savez_compressed(out / f'{tag}.npz', outcome=codes, peak_mm=pk, when=when.numpy(), state=states,
                            escaped=escaped.numpy(), overflow=overflow)
        worst = pk.max(axis=1)
        result = dict(tag=a.tag, case=case, params=params, seed=seed, nworld=nworld, dt=dt, check_steps=check,
                      deterministic=a.deterministic, iterations=a.iterations, ls_iterations=a.ls_iterations,
                      tolerance=a.tolerance, njmax=NJMAX, ncon_per_world=NCON, nv=int(sim.m.nv),
                      wall_s=wall, steps=executed - 2, step_ms=1e3 * wall / (executed - 2),
                      completed=int((codes != 0).sum()), valid=valid,
                      evals_per_s=int((codes != 0).sum()) / wall if valid else None,
                      env_steps_per_s=nworld * (executed - 2) / wall, overflow_worlds=invalid,
                      overflow_world_steps=int(overflow_events.numpy().sum()), finite_worlds=int(finite.sum()),
                      escaped_worlds=int(np.count_nonzero(escaped.numpy())), outcomes=categories,
                      majority=max(categories, key=categories.get),
                      sampled_nefc_max=nefc_max, sampled_nacon_max=ncon_max,
                      sampled_niter_mean=float(np.mean(iters)) if iters else None,
                      max_penetration_mm=float(worst.max()), vertex_in_box_max_mm=float(pk[:, 0].max()),
                      corner_outside_max_mm=float(pk[:, 1].max()),
                      penetration_quantiles_mm=dict(zip(['p50', 'p90', 'p99', 'max'],
                                                        np.percentile(worst, [50, 90, 99, 100]).tolist())),
                      worlds_over_1mm=int((worst > 1.).sum()), distinct_outcome_patterns=len(np.unique(codes)),
                      identical_worlds=bool(np.all(states == states[0])),
                      hostname=socket.gethostname(), finished_at=datetime.now().astimezone().isoformat(),
                      per_world=f'{tag}.npz')
    result['peak_total_gpu_mib'] = mem.peak
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True, help='本轮运行目录 test/runs/YYYY-MM-DD/bag/<run-id>')
    ap.add_argument('--tag', required=True, help='配置名，进文件名')
    ap.add_argument('--nworld', type=int, required=True)
    ap.add_argument('--cases', nargs='*', default=None, help='如 centre:1；缺省为 F3 的 9 个输入')
    ap.add_argument('--dt', type=float, default=.0008)
    ap.add_argument('--deterministic', default='NONE', help="DeterminismType 名，可用 + 组合，如 CONTACTS+CONSTRAINT")
    ap.add_argument('--iterations', type=int, default=BASE.iterations)
    ap.add_argument('--ls-iterations', type=int, default=BASE.ls_iterations)
    ap.add_argument('--tolerance', type=float, default=BASE.tolerance)
    ap.add_argument('--max-steps', type=int, default=0, help='只跑这么多步（入口检查用，不出结论）')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    wanted = None if a.cases is None else {tuple(c.split(':')) for c in a.cases}
    path = a.output / f'{a.tag}_n{a.nworld}.jsonl'
    if path.exists(): raise FileExistsError(f'refusing to overwrite {path}')
    for name, params, seed in f3_inputs():
        if wanted is not None and (name, str(seed)) not in wanted: continue
        result = run(a, name, params, seed, a.output)
        with path.open('a') as f: f.write(json.dumps(result) + '\n')
        print(json.dumps({k: result[k] for k in ('tag', 'case', 'seed', 'nworld', 'wall_s', 'step_ms', 'evals_per_s',
                                                 'outcomes', 'max_penetration_mm', 'worlds_over_1mm',
                                                 'identical_worlds', 'overflow_worlds', 'sampled_niter_mean')}),
              flush=True)


if __name__ == '__main__':
    main()

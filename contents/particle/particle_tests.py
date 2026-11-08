"""Tests for the particle-bond soft model. Usage: particle_tests.py <shape> <test> [--out DIR]

shape: flat | blob
test:  settle | cantilever | squeeze | finger | edge_lift | top_pinch
Calibrated gains are read from <out>/gains_<shape>.json when present.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np

import particle_soft as ps

HERE = Path(__file__).resolve().parent

FLOOR = f'<geom name="floor" type="plane" size="2 2 .01" rgba=".82 .84 .87 1" {ps.TOOL_CONTACT.format(mu=0.6)}/>'
NO_TOOL = ''
BUILDERS = dict(flat=ps.build_flat, blob=ps.build_blob, pillow=ps.build_pillow)
SHAPE_DEFAULTS = dict(pillow=dict(mass=0.8))


def platen_half(shape, x):
    p = BUILDERS[shape](x)[0]
    return (float(np.ptp(p[:, 0])) / 2 + 0.03, float(np.ptp(p[:, 1])) / 2 + 0.03)


def build(shape, x, tools=NO_TOOL, table=FLOOR, offset=(0, 0, 0)):
    return ps.Soft(BUILDERS[shape](x), x, tools, table, offset)


def step(soft, ctrl=None, n=1):
    for _ in range(n):
        soft.apply()
        if ctrl:
            ctrl()
        mujoco.mj_step(soft.model, soft.data)
        soft.advance_history()
    soft.check()


def press_tool(half, xy, z):
    contact = ps.TOOL_CONTACT.format(mu=0.8)
    return (f'<body name="tool" pos="{xy[0]} {xy[1]} {z}" gravcomp="1"><joint name="tz" type="slide" axis="0 0 1" armature="2"/>'
            f'<geom name="tool_g" type="box" size="{half[0]} {half[1]} .01" mass=".1" rgba=".3 .5 .8 .55" {contact}/></body>')


def press(shape, x, half, xy, f_cap, depth_cap, speed=0.01, hold_s=2.0, creep_f=None, out=None, unload=True):
    top = 0.25
    soft = build(shape, x, press_tool(half, xy, top))
    m, d = soft.model, soft.data
    dof, q = m.joint('tz').dofadr[0], m.joint('tz').qposadr[0]
    geom = {m.geom('tool_g').id}
    state = dict(target=0.0, mode='pos', force=0.0)

    def ctrl():
        if state['mode'] == 'pos':
            d.qfrc_applied[dof] = 8e4 * (state['target'] - d.qpos[q]) - 800 * d.qvel[dof]
        else:
            d.qfrc_applied[dof] = -state['force'] - 400 * d.qvel[dof]

    def fz():
        return ps.tool_force(m, d, geom)[2]

    step(soft, ctrl, 2500)
    settled_top = soft.positions()[:, 2].max() + ps.RADIUS
    state['target'] = settled_top + 0.012 - top
    step(soft, ctrl, 1500)
    while True:
        state['target'] -= speed * soft.dt * 10
        step(soft, ctrl, 10)
        if fz() > 0.05 or d.qpos[q] < -0.3:
            break
    z0 = d.qpos[q]
    load = []
    while True:
        state['target'] -= speed * soft.dt * 10
        step(soft, ctrl, 10)
        depth, f = z0 - d.qpos[q], fz()
        load.append((depth, f))
        if f >= f_cap or depth >= depth_cap:
            break
    load = np.array(load)
    res = dict(n=soft.n, thick=soft.meta['thick'], area=soft.meta['area'], settled_top=float(settled_top), load=load,
               peak_force=float(load[-1, 1]), peak_depth=float(load[-1, 0]),
               force_at_depth={f'{mm}mm': float(np.interp(mm * 1e-3, load[:, 0], load[:, 1]))
                               for mm in (2, 5, 8, 12, 16, 20, 30) if load[-1, 0] >= mm * 1e-3},
               monotonic_drop=float(np.max(np.maximum.accumulate(load[:, 1]) - load[:, 1])))
    if not unload:
        return res
    hold = []
    for _ in range(int(hold_s / soft.dt / 25)):
        step(soft, ctrl, 25)
        hold.append(fz())
    if out:
        ps.render(soft, out)
    unload_trace = []
    while state['target'] < z0 + 0.006:
        state['target'] += speed * soft.dt * 10
        step(soft, ctrl, 10)
        unload_trace.append((z0 - d.qpos[q], fz()))
    unload_trace = np.array(unload_trace)
    inside = unload_trace[unload_trace[:, 0] >= 0]
    area_l, area_u = np.trapezoid(load[:, 1], load[:, 0]), -np.trapezoid(inside[:, 1], inside[:, 0])
    touching = np.where(unload_trace[:, 1] > 0.05)[0]
    res.update(hysteresis=float((area_l - area_u) / max(area_l, 1e-9)),
               residual=float(unload_trace[touching[-1], 0]) if len(touching) else 0.0,
               relax=float(np.mean(hold[-10:]) / max(np.mean(hold[:3]), 1e-9)))
    if creep_f:
        step(soft, ctrl, int(1.5 / soft.dt))
        state.update(mode='force', force=creep_f)
        trace = []
        for _ in range(int(8.0 / soft.dt / 25)):
            step(soft, ctrl, 25)
            trace.append(d.qpos[q])
        trace = np.array(trace)
        res.update(creep_depth=float(trace[int(0.5 / (25 * soft.dt))] - trace[-1]),
                   creep_last_2s=float(trace[int(6 / (25 * soft.dt))] - trace[-1]))
    return res


def test_settle(shape, x, out):
    soft = build(shape, x)
    t0 = time.perf_counter()
    top0 = soft.positions()[:, 2].max()
    step(soft, None, 5000)
    wall = time.perf_counter() - t0
    ps.render(soft, out / f'{shape}_settle.png')
    return dict(n=soft.n, nv=int(soft.model.nv), top0=float(top0), top=float(soft.positions()[:, 2].max()),
                frames_per_s=1.0 / wall * 100, ncon=int(soft.data.ncon))


def test_cantilever(shape, x, out, overhang=0.10):
    edge = 0.15 - overhang
    table = (f'<geom name="table" type="box" pos="{edge - 1} 0 -.05" size="1 1 .05" rgba=".82 .84 .87 1" '
             f'{ps.TOOL_CONTACT.format(mu=0.9)}/>')
    soft = build(shape, x, NO_TOOL, table)
    held = np.where(soft.start[:, 0] < edge - 0.03)[0]

    def clamp():
        disp = soft.positions()[held] - soft.start[held]
        force = -300 * disp - 2 * soft.velocities()[held]
        soft.data.qfrc_applied[soft.d0 + 3 * held[:, None] + np.arange(3)] += force

    step(soft, clamp, int(3.0 / soft.dt))
    p = soft.positions()
    nx, ny, nz = soft.meta['idx']
    mid = ny // 2
    line = p[[a * ny * nz + mid * nz + 1 for a in range(nx)]]
    tip = line[-1]
    root = line[int(np.argmin(np.abs(line[:, 0] - edge)))]
    theta = float(np.arctan2(root[2] - tip[2], tip[0] - root[0]))
    # Peirce: bending length c = l * (cos(theta/2) / (8 tan(theta)))^(1/3)
    length = float(np.linalg.norm(np.diff(line[line[:, 0] >= root[0]], axis=0), axis=1).sum())
    c = length * (np.cos(theta / 2) / (8 * np.tan(theta))) ** (1 / 3) if 1e-3 < theta < 1.5 else None
    rho = x.mass / soft.meta['area']
    ps.render(soft, out / f'{shape}_cantilever_kb{x.k_bend:g}.png', azimuth=90, elevation=-8, distance=0.6, lookat=(edge, 0, -0.05))
    return dict(k_bend=x.k_bend, k_shear=x.k_shear, tip_angle_deg=float(np.degrees(theta)), arc_length_mm=length * 1e3,
                bending_length_mm=None if c is None else c * 1e3, D_total_Nm=None if c is None else rho * 9.81 * c ** 3)


def squeeze_targets(x, meta):
    return dict(creep_force=x.p_creep * meta['area'], creep_depth=meta['thick'] * x.creep_eps)


def test_squeeze(shape, x, out):
    meta = BUILDERS[shape](x)[4]
    res = press(shape, x, platen_half(shape, x), (0, 0), 500.0, meta['thick'] * 0.32, creep_f=x.p_creep * meta['area'],
                out=out / f'{shape}_squeeze.png')
    load = res.pop('load')
    area, thick = res['area'], res['thick']
    eps2 = 0.30
    p_at = lambda eps: float(np.interp(eps * thick, load[:, 0], load[:, 1]) / area)
    target2 = x.p_knee + (eps2 - x.eps_knee) / (x.eps_lock - x.eps_knee) * (x.p_lock - x.p_knee)
    res.update(p_knee_measured=p_at(x.eps_knee), p_knee_target=x.p_knee, p_eps30_measured=p_at(eps2), p_eps30_target=target2,
               creep_target=thick * x.creep_eps)
    return res


def test_finger(shape, x, out):
    cases = {'20mm center': ((.01, .01), (0, 0)), '20mm off-node': ((.01, .01), (.0125, .0125)),
             '20mm side': ((.01, .01), (.06, .04)), '60mm': ((.03, .03), (0, 0)), 'platen': (platen_half(shape, x), (0, 0))}
    res = {}
    for name, (half, xy) in cases.items():
        try:
            r = press(shape, x, half, xy, 30.0, 0.02, unload=False)
            res[name] = dict(peak_force=r['peak_force'], peak_depth=r['peak_depth'], force_at_depth=r['force_at_depth'],
                             largest_drop=r['monotonic_drop'])
        except FloatingPointError as err:
            res[name] = dict(FAILED=str(err))
    return res


def test_edge_lift(shape, x, out, overhang=0.07, lift=0.12, speed=0.04):
    """Top and bottom jaws pinch one overhanging corner, then rise."""
    probe = BUILDERS[shape](x)[0]
    x_max = probe[:, 0].max()
    edge = x_max - overhang
    corner_y = probe[:, 1].max() - 0.035 if shape in ('flat', 'pillow') else float(probe[probe[:, 0] > x_max - 0.03, 1].mean())
    contact = ps.TOOL_CONTACT.format(mu=1.0)
    table = (f'<geom name="table" type="box" pos="{edge - 1} 0 -.05" size="1 1 .05" rgba=".82 .84 .87 1" '
             f'{ps.TOOL_CONTACT.format(mu=0.6)}/>')
    gx = x_max - 0.03
    high = probe[:, 2].max() + 0.05
    tools = (f'<body name="car" pos="{gx} {corner_y} 0" gravcomp="1"><joint name="lz" type="slide" axis="0 0 1" armature="2"/>'
             '<inertial pos="0 0 0" mass=".05" diaginertia="1e-4 1e-4 1e-4"/>'
             f'<geom name="low" type="box" pos="0 0 -.005" size=".025 .025 .005" mass=".01" rgba=".3 .5 .8 1" {contact}/>'
             f'<body name="up" pos="0 0 {high}" gravcomp="1"><joint name="cz" type="slide" axis="0 0 1" armature="1"/>'
             f'<geom name="upg" type="box" size=".025 .025 .005" mass=".01" rgba=".3 .5 .8 1" {contact}/></body></body>')
    soft = build(shape, x, tools, table)
    m, d = soft.model, soft.data
    dl, ql, dc = m.joint('lz').dofadr[0], m.joint('lz').qposadr[0], m.joint('cz').dofadr[0]
    state = dict(target=0.0, close=0.0)

    def ctrl():
        d.qfrc_applied[dl] = 3e4 * (state['target'] - d.qpos[ql]) - 500 * d.qvel[dl]
        d.qfrc_applied[dc] = -state['close'] - 150 * d.qvel[dc]

    step(soft, ctrl, 5000)
    for k in range(60):
        state['close'] = 25.0 * (k + 1) / 60
        step(soft, ctrl, 100)
    step(soft, ctrl, 2500)
    before = soft.positions().copy()
    com0 = soft.com()
    for k in range(int(lift / speed / soft.dt / 10)):
        state['target'] = speed * k * soft.dt * 10
        step(soft, ctrl, 10)
    step(soft, ctrl, 7500)
    p = soft.positions()
    ps.render(soft, out / f'{shape}_edge_lift.png', azimuth=115, elevation=-15, distance=0.8, lookat=(gx - 0.1, 0, 0.05))
    near = np.linalg.norm(before[:, :2] - [gx, corner_y], axis=1) < 0.04
    far = before[:, 0] < before[:, 0].min() + 0.05
    return dict(lift_force_N=float(d.qfrc_applied[dl]), weight_N=x.mass * 9.81, lifted_share=float(d.qfrc_applied[dl] / (x.mass * 9.81)),
                near_grip_rise_mm=float((p[near, 2] - before[near, 2]).mean() * 1e3),
                far_end_rise_mm=float((p[far, 2] - before[far, 2]).mean() * 1e3),
                nodes_up_50mm=int(((p[:, 2] - before[:, 2]) > 0.05).sum()), n=soft.n,
                com_rise_mm=float((soft.com() - com0)[2] * 1e3), com_shift_x_mm=float((soft.com() - com0)[0] * 1e3),
                max_speed=float(np.abs(soft.velocities()).max()))


def test_top_pinch(shape, x, out, lift=0.10, speed=0.04, gap=0.03, limit=15.0, depth=0.015):
    """Jaws descend until they touch, press in, close to a commanded gap under a force limit, rise."""
    probe = BUILDERS[shape](x)[0]
    surface = probe[:, 2].max() + ps.RADIUS
    contact = ps.TOOL_CONTACT.format(mu=1.2)
    start = surface + 0.06
    spread = 0.05
    tools = (f'<body name="hand" pos="0 0 {start}" gravcomp="1"><joint name="hz" type="slide" axis="0 0 1" armature="2"/>'
             '<inertial pos="0 0 0" mass=".05" diaginertia="1e-4 1e-4 1e-4"/>'
             f'<body name="ja" pos="{-spread} 0 0" gravcomp="1"><joint name="xa" type="slide" axis="1 0 0" armature="1"/>'
             f'<geom name="ga" type="box" size=".006 .025 .03" mass=".01" rgba=".3 .5 .8 1" {contact}/></body>'
             f'<body name="jb" pos="{spread} 0 0" gravcomp="1"><joint name="xb" type="slide" axis="1 0 0" armature="1"/>'
             f'<geom name="gb" type="box" size=".006 .025 .03" mass=".01" rgba=".3 .5 .8 1" {contact}/></body></body>')
    soft = build(shape, x, tools)
    m, d = soft.model, soft.data
    dh, qh = m.joint('hz').dofadr[0], m.joint('hz').qposadr[0]
    da, db = m.joint('xa').dofadr[0], m.joint('xb').dofadr[0]
    qa, qb = m.joint('xa').qposadr[0], m.joint('xb').qposadr[0]
    geoms = {m.geom('ga').id, m.geom('gb').id}
    state = dict(z=0.0, close=0.0)

    def ctrl():
        d.qfrc_applied[dh] = 3e4 * (state['z'] - d.qpos[qh]) - 500 * d.qvel[dh]
        d.qfrc_applied[da] = np.clip(4e3 * (state['close'] - d.qpos[qa]), -limit, limit) - 100 * d.qvel[da]
        d.qfrc_applied[db] = np.clip(4e3 * (-state['close'] - d.qpos[qb]), -limit, limit) - 100 * d.qvel[db]

    step(soft, ctrl, 2500)
    touched = False
    while d.qpos[qh] > -start:
        state['z'] -= 0.04 * soft.dt * 10
        step(soft, ctrl, 10)
        if abs(ps.tool_force(m, d, geoms)[2]) > 0.1:
            touched = True
            break
    z_touch = float(d.qpos[qh])
    for _ in range(int(depth / 0.04 / soft.dt / 10)):
        state['z'] -= 0.04 * soft.dt * 10
        step(soft, ctrl, 10)
    step(soft, ctrl, 2500)
    travel = spread - 0.006 - gap / 2
    for k in range(100):
        state['close'] = travel * (k + 1) / 100
        step(soft, ctrl, 100)
    step(soft, ctrl, 5000)
    gap_closed = float(d.qpos[qb] - d.qpos[qa] + 2 * spread - 0.012)
    squeeze_force = float(np.abs(d.qfrc_applied[da]))
    before = soft.positions().copy()
    com0 = soft.com()
    z_start = state['z']
    peak = 0.0
    for k in range(int(lift / speed / soft.dt / 10)):
        state['z'] = z_start + speed * k * soft.dt * 10
        step(soft, ctrl, 10)
        peak = max(peak, float(d.qfrc_applied[dh]))
    step(soft, ctrl, 7500)
    p = soft.positions()
    ps.render(soft, out / f'{shape}_top_pinch_gap{gap * 1e3:.0f}.png', azimuth=125, elevation=-15, distance=0.8, lookat=(0, 0, 0.08))
    rise = p[:, 2] - before[:, 2]
    held = abs(ps.tool_force(m, d, geoms)[2])
    return dict(touched=touched, touch_drop_mm=float((start - (start + z_touch)) * -1e3), gap_mm=gap_closed * 1e3,
                jaw_force_N=squeeze_force, lift_force_end_N=float(d.qfrc_applied[dh]), lift_force_peak_N=peak,
                held_vertical_N=float(held), weight_N=x.mass * 9.81, lifted_share=float(d.qfrc_applied[dh] / (x.mass * 9.81)),
                nodes_up_30mm=int((rise > 0.03).sum()), nodes_up_70mm=int((rise > 0.07).sum()), n=soft.n,
                max_rise_mm=float(rise.max() * 1e3), com_rise_mm=float((soft.com() - com0)[2] * 1e3),
                max_speed=float(np.abs(soft.velocities()).max()))


def load_x(shape, out, **overrides):
    gains = {}
    path = out / f'gains_{shape}.json'
    if path.exists():
        gains = json.loads(path.read_text())
    return ps.SoftX(**{**SHAPE_DEFAULTS.get(shape, {}), **gains, **overrides})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('shape', choices=sorted(BUILDERS))
    ap.add_argument('test')
    ap.add_argument('--out', default=str(HERE / 'results_particle'))
    ap.add_argument('--k-bend', type=float, default=None)
    ap.add_argument('--gap', type=float, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    overrides = {} if args.k_bend is None else dict(k_bend=args.k_bend)
    x = load_x(args.shape, out, **overrides)
    t0 = time.perf_counter()
    name = args.test + (f'_kb{args.k_bend:g}' if args.k_bend is not None else '') + (f'_gap{args.gap * 1e3:.0f}' if args.gap else '')
    try:
        extra = dict(gap=args.gap) if args.gap else {}
        res = globals()[f'test_{args.test}'](args.shape, x, out, **extra)
    except FloatingPointError as err:
        res = dict(FAILED=str(err))
    res['wall_total_s'] = round(time.perf_counter() - t0, 1)
    (out / f'{args.shape}_{name}.json').write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(args.shape, name, json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

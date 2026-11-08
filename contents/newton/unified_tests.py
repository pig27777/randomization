"""Tests for the unified layered model and the cloth references, on Newton VBD.

Usage: unified_tests.py <item> <test> [--seed N] [--out DIR]
item: folded_clothing | crumpled_clothing | pillow | cloth_folded | cloth_crumpled
test: speed | squeeze | finger | edge_pinch | top_pinch
Grips close until a force limit, like a force-controlled gripper.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton_soft as ns
import unified_soft as us

HERE = Path(__file__).resolve().parent

GANTRY_KE = 5e4
JAW_KE = 2e4
GRIP_FORCE = 15.0
PRESS_KD = 2e3
JAW_KD = 100.0
GANTRY_KD = 3e3
TABLE_Z = 0.2
REF_OUT = HERE / 'results_newton'


def content(builder, item, base_z):
    if item in us.PRESETS:
        shape = us.PRESETS[item]
        info = us.add_item(builder, shape, base_z)
        info['mass'] = shape.mass
        return info
    info = ns.add_content(builder, item, base_z, REF_OUT)
    info['mass'] = ns.MASS
    return info


def footprint(item):
    """Unloaded positions before any simulation, used to place rigs."""
    if item in us.PRESETS:
        return us.generate(us.PRESETS[item])[0]
    if item == 'cloth_crumpled':
        q = np.load(REF_OUT / 'crumple_state.npy')
        return q - [q[:, 0].mean(), q[:, 1].mean(), q[:, 2].min()]
    verts, _, _ = ns.folded_sheet()
    return verts - [0, 0, verts[:, 2].min()]


def make_sim(builder, info, item, base_z):
    sim = ns.Sim(builder, info)
    if item == 'cloth_crumpled':
        ns.place_content(sim, item, base_z, REF_OUT)
    return sim


def set_target(sim, idx, value):
    sim.targets[idx] = value
    sim.apply_targets()


# ---------- press ----------

def press_rig(builder, half, xy, top):
    link = builder.add_link(xform=wp.transform(wp.vec3(xy[0], xy[1], top), wp.quat_identity()), mass=0.5, label='press')
    builder.add_shape_box(link, hx=half[0], hy=half[1], hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.8))
    joint = builder.add_joint_prismatic(parent=-1, child=link, axis=wp.vec3(0, 0, 1),
                                        parent_xform=wp.transform(wp.vec3(xy[0], xy[1], 0), wp.quat_identity()),
                                        target_ke=GANTRY_KE, target_kd=PRESS_KD, target_pos=top, label='press_z')
    builder.add_articulation([joint], label='press')
    builder.joint_q[-1] = top
    return link


def press(item, half, xy, depth_frac=0.4, f_cap=60.0, speed=0.01, hold_s=2.0, unload=True, view=None):
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.add_ground_plane()
    info = content(builder, item, ns.Z0)
    top = float(footprint(item)[:, 2].max()) + 0.08
    link = press_rig(builder, half, xy, top)
    sim = make_sim(builder, info, item, ns.Z0)
    sim.frame(80)
    surface = float(sim.q()[:, 2].max())
    z = top
    while z > surface + 0.015:
        z -= 0.002
        set_target(sim, 0, z)
        sim.frame(1)
    sim.frame(40)
    # Static offset of the drive under the press's own weight; a pushing load raises the body above its target.
    baseline = GANTRY_KE * (sim.body_z(link) - z)

    last = dict(z=sim.body_z(link))

    def force():
        z_now = sim.body_z(link)
        velocity = (z_now - last['z']) / ns.FRAME_DT
        last['z'] = z_now
        return GANTRY_KE * (z_now - sim.targets[0]) + PRESS_KD * velocity - baseline

    while force() < 0.1:
        z -= speed * ns.FRAME_DT
        set_target(sim, 0, z)
        sim.frame(1)
        if z < surface - 0.05:
            raise RuntimeError('press never touched')
    z0 = sim.body_z(link)
    thickness = surface
    load = []
    while True:
        z -= speed * ns.FRAME_DT
        set_target(sim, 0, z)
        sim.frame(1)
        depth, f = z0 - sim.body_z(link), force()
        load.append((depth, f))
        if f >= f_cap or depth >= depth_frac * thickness:
            break
    load = np.array(load)
    res = dict(contact_height=float(z0), peak_force=float(load[-1, 1]), peak_depth_mm=float(load[-1, 0] * 1e3),
               peak_strain=float(load[-1, 0] / thickness),
               force_at_strain={f'{int(e * 100)}%': float(np.interp(e * thickness, load[:, 0], load[:, 1]))
                                for e in (0.05, 0.1, 0.2, 0.3, 0.4) if load[-1, 0] >= e * thickness},
               largest_drop=float(np.max(np.maximum.accumulate(load[:, 1]) - load[:, 1])))
    if not unload:
        return res
    held = []
    for _ in range(int(hold_s / ns.FRAME_DT / 5)):
        sim.frame(5)
        held.append(force())
    if view:
        ns.save_state(sim, view, [(link, half[0], half[1], 0.01)])
    back = []
    while z < z0 + 0.01:
        z += speed * ns.FRAME_DT
        set_target(sim, 0, z)
        sim.frame(1)
        back.append((z0 - sim.body_z(link), force()))
    back = np.array(back)
    inside = back[back[:, 0] >= 0]
    area_l, area_u = np.trapezoid(load[:, 1], load[:, 0]), -np.trapezoid(inside[:, 1], inside[:, 0])
    touching = np.where(back[:, 1] > 0.1)[0]
    res.update(hysteresis=float((area_l - area_u) / max(area_l, 1e-9)),
               residual_mm=float(back[touching[-1], 0] * 1e3) if len(touching) else 0.0,
               relax=float(np.mean(held[-5:]) / max(np.mean(held[:2]), 1e-9)))
    return res


def test_speed(item, out, seed):
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.add_ground_plane()
    info = content(builder, item, ns.Z0)
    sim = make_sim(builder, info, item, ns.Z0)
    sim.frame(20)
    wp.synchronize()
    t0 = time.perf_counter()
    sim.frame(100)
    wp.synchronize()
    wall = time.perf_counter() - t0
    ns.save_state(sim, out / f'{item}_settled.npz')
    q = sim.q()
    return dict(particles=len(q), tets=info.get('n_tets'), frames_per_s=100 / wall, height=float(q[:, 2].max()),
                footprint=[float(np.ptp(q[:, 0])), float(np.ptp(q[:, 1]))])


def test_squeeze(item, out, seed):
    p = footprint(item)
    half = (float(np.ptp(p[:, 0])) / 2 + 0.03, float(np.ptp(p[:, 1])) / 2 + 0.03)
    return press(item, half, (0, 0), view=out / f'{item}_squeeze.npz')


def test_finger(item, out, seed):
    p = footprint(item)
    platen = (float(np.ptp(p[:, 0])) / 2 + 0.03, float(np.ptp(p[:, 1])) / 2 + 0.03)
    res = {}
    for name, half, xy in (('20mm', (.01, .01), (0, 0)), ('20mm off-center', (.01, .01), (.06, .04)),
                           ('60mm', (.03, .03), (0, 0)), ('platen', platen, (0, 0))):
        try:
            res[name] = press(item, half, xy, depth_frac=0.3, f_cap=40.0, unload=False)
        except (FloatingPointError, RuntimeError) as err:
            res[name] = dict(FAILED=str(err))
    return res


# ---------- grips ----------

def jaw_rig(builder, x, y, z_start, axis, half, offset):
    gantry = builder.add_link(xform=wp.transform(wp.vec3(x, y, z_start), wp.quat_identity()), mass=0.1, label='gantry')
    a = builder.add_link(xform=wp.transform(wp.vec3(x, y, z_start) + wp.vec3(*axis) * offset, wp.quat_identity()), label='a')
    b = builder.add_link(xform=wp.transform(wp.vec3(x, y, z_start) - wp.vec3(*axis) * offset, wp.quat_identity()), label='b')
    cfg = newton.ModelBuilder.ShapeConfig(density=500.0, mu=1.0)
    builder.add_shape_box(a, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
    builder.add_shape_box(b, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
    j0 = builder.add_joint_prismatic(parent=-1, child=gantry, axis=wp.vec3(0, 0, 1),
                                     parent_xform=wp.transform(wp.vec3(x, y, 0), wp.quat_identity()),
                                     target_ke=GANTRY_KE, target_kd=GANTRY_KD, target_pos=z_start, label='gz')
    j1 = builder.add_joint_prismatic(parent=gantry, child=a, axis=wp.vec3(*axis), target_ke=JAW_KE, target_kd=JAW_KD, target_pos=offset)
    j2 = builder.add_joint_prismatic(parent=gantry, child=b, axis=-wp.vec3(*axis), target_ke=JAW_KE, target_kd=JAW_KD, target_pos=offset)
    builder.add_articulation([j0, j1, j2], label='gripper')
    builder.joint_q[-3:] = [z_start, offset, offset]
    return gantry, a, b


class Grip:
    def __init__(self, sim, gantry, a, b, axis):
        self.sim, self.g, self.a, self.b, self.axis = sim, gantry, a, b, np.array(axis, float)

    def opening(self, link):
        bq = self.sim.s0.body_q.numpy()
        return float(np.dot(bq[link][:3] - bq[self.g][:3], self.axis) * (1 if link == self.a else -1))

    def squeeze_force(self):
        t = self.sim.targets
        openings = (self.opening(self.a), self.opening(self.b))
        previous = getattr(self, '_previous', openings)
        self._previous = openings
        rate = [(o - p) / ns.FRAME_DT for o, p in zip(openings, previous)]
        return 0.5 * sum(JAW_KE * (o - target) + JAW_KD * r for o, target, r in zip(openings, (t[1], t[2]), rate))

    def load(self, baseline):
        return GANTRY_KE * (self.sim.targets[0] - self.sim.body_z(self.g)) - baseline

    def close(self, limit=GRIP_FORCE, floor=0.0):
        off = self.sim.targets[1]
        self.squeeze_force()
        while self.squeeze_force() < limit and off > floor:
            off = max(off - 0.0002, floor)
            self.sim.targets[1] = self.sim.targets[2] = off
            self.sim.apply_targets()
            self.sim.frame(1)
        self.sim.frame(40)
        return self.squeeze_force()

    def lift(self, dz, step=0.0002, hold_frames=100):
        z = self.sim.targets[0]
        loads = []
        for k in range(int(dz / step)):
            z += step
            set_target(self.sim, 0, z)
            self.sim.frame(1)
        for _ in range(hold_frames // 10):
            self.sim.frame(10)
            loads.append(self.load(self.baseline))
        return np.array(loads)


def lift_report(grip, item_mass, before, sim, path, links, half, extra=None):
    q = sim.q()
    rise = q[:, 2] - before[:, 2]
    loads = grip.lift_loads
    held = float(np.mean(loads[-3:]))
    res = dict(held_load_N=held, weight_N=item_mass * 9.81, held_share=held / (item_mass * 9.81),
               share_up_30mm=float((rise > 0.03).mean()), share_up_80mm=float((rise > 0.08).mean()),
               max_rise_mm=float(rise.max() * 1e3), squeeze_N=grip.squeeze_force(),
               success=bool(held > 0.05 * item_mass * 9.81 and (rise > 0.08).mean() > 0.02))
    res.update(extra or {})
    ns.save_state(sim, path, [(l, *half) for l in links])
    return res


def test_edge_pinch(item, out, seed, overhang=0.05, lift=0.10):
    p = footprint(item)
    x_max = float(p[:, 0].max())
    grip_x = x_max - 0.025
    grip_y = 0.07 if item in ('folded_clothing', 'cloth_folded') else 0.0
    local = p[(np.abs(p[:, 0] - grip_x) < 0.025) & (np.abs(p[:, 1] - grip_y) < 0.025)]
    thick = float(local[:, 2].max()) if len(local) else float(p[:, 2].max())
    half = (0.02, 0.02, 0.005)
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.add_ground_plane()
    info = content(builder, item, TABLE_Z + ns.Z0)
    edge = x_max - overhang
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(edge - 0.5, 0, TABLE_Z - 0.1), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.1,
                          cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
    offset = thick / 2 + half[2] + 0.03
    gantry, a, b = jaw_rig(builder, grip_x, grip_y, TABLE_Z + thick / 2, (0, 0, 1), half, offset)
    sim = make_sim(builder, info, item, TABLE_Z + ns.Z0)
    grip = Grip(sim, gantry, a, b, (0, 0, 1))
    sim.frame(80)
    grip.baseline = GANTRY_KE * (sim.targets[0] - sim.body_z(gantry))
    squeeze = grip.close(floor=half[2] + 0.001)
    before = sim.q().copy()
    grip.lift_loads = grip.lift(lift)
    return lift_report(grip, info['mass'], before, sim, out / f'{item}_edge_pinch_s{seed}.npz', (a, b), half,
                       dict(thickness_at_grip=thick, squeeze_before_lift_N=squeeze, opening_mm=2 * grip.opening(a) * 1e3 - 2 * half[2] * 1e3))


def test_top_pinch(item, out, seed, lift=0.10, press_in=0.012):
    p = footprint(item)
    half = (0.01, 0.02, 0.03)
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.add_ground_plane()
    info = content(builder, item, ns.Z0)
    top = float(p[:, 2].max()) + 0.08
    gantry, a, b = jaw_rig(builder, 0.0, 0.0, top, (1, 0, 0), half, 0.05)
    sim = make_sim(builder, info, item, ns.Z0)
    grip = Grip(sim, gantry, a, b, (1, 0, 0))
    sim.frame(80)
    q = sim.q()
    surface = float(q[(np.abs(q[:, 0]) < 0.07) & (np.abs(q[:, 1]) < 0.04), 2].max())
    z = top
    while z > surface + half[2] - press_in:
        z -= 0.002
        set_target(sim, 0, z)
        sim.frame(1)
    sim.frame(40)
    grip.baseline = GANTRY_KE * (sim.targets[0] - sim.body_z(gantry))
    squeeze = grip.close(floor=half[0] + 0.001)
    before = sim.q().copy()
    grip.lift_loads = grip.lift(lift)
    return lift_report(grip, info['mass'], before, sim, out / f'{item}_top_pinch_s{seed}.npz', (a, b), half,
                       dict(squeeze_before_lift_N=squeeze, gap_mm=2 * grip.opening(a) * 1e3 - 2 * half[0] * 1e3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('item')
    ap.add_argument('test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(HERE / 'results_unified'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    try:
        res = globals()[f'test_{args.test}'](args.item, out, args.seed)
    except (FloatingPointError, RuntimeError) as err:
        res = dict(FAILED=str(err))
    res['wall_s'] = round(time.perf_counter() - t0, 1)
    suffix = f'_s{args.seed}' if args.test.endswith('pinch') else ''
    (out / f'{args.item}_{args.test}{suffix}.json').write_text(json.dumps(res, indent=2))
    print(args.item, args.test, suffix, json.dumps(res), flush=True)


if __name__ == '__main__':
    main()

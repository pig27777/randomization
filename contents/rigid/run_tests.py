"""Three tests on the one unified model: squeeze, finger width, micro-lift.

Usage: run_tests.py [--dt 0.0005] [--items book clothing bottle] [--out results]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

import unified_content as uc

HERE = Path(__file__).resolve().parent

F_TOUCH = 0.05
SPEED = 0.005


def contact_force_on(model, data, geom_ids):
    total, wrench = 0.0, np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom[0] in geom_ids or c.geom[1] in geom_ids:
            mujoco.mj_contactForce(model, data, i, wrench)
            total += wrench[0]
    return total


class PressRig:
    """Item on the floor, pressed from above by a box tool on a stiff position servo."""

    def __init__(self, x, tool_half, tool_xy=(0.0, 0.0), dt=0.0005):
        body, tendons, self.n_active = uc.item_xml(x, pos=(0, 0, x.half[2] + 0.002))
        top = 2 * x.half[2] + 0.03
        xml = (uc.scene_header(dt) + '<worldbody><light pos="0 0 2"/>'
               f'<geom name="floor" type="plane" size="1 1 .01" rgba=".75 .78 .8 1" {uc.TOOL_CONTACT}/>' + body
               + f'<body name="tool" pos="{tool_xy[0]} {tool_xy[1]} {top}" gravcomp="1">'
                 '<joint name="tool_z" type="slide" axis="0 0 1" armature="5"/>'
                 f'<geom name="tool_g" type="box" size="{tool_half[0]} {tool_half[1]} .01" mass=".1" rgba=".3 .5 .8 .6" {uc.TOOL_CONTACT}/></body>'
               f'</worldbody><tendon>{tendons}</tendon></mujoco>')
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.dof = self.model.joint('tool_z').dofadr[0]
        self.qadr = self.model.joint('tool_z').qposadr[0]
        self.tool = {self.model.geom('tool_g').id}
        self.dt = dt
        self.target = 0.0
        self.mode = 'pos'
        self.hold_force = 0.0
        self.steps = 0

    @property
    def z(self):
        return float(self.data.qpos[self.qadr])

    def step(self):
        d = self.data
        if self.mode == 'pos':
            d.qfrc_applied[self.dof] = 2e5 * (self.target - self.z) - 2000 * d.qvel[self.dof]
        else:
            d.qfrc_applied[self.dof] = -self.hold_force - 300 * d.qvel[self.dof]
        mujoco.mj_step(self.model, d)
        self.steps += 1
        if not np.isfinite(d.qpos).all() or np.abs(d.qvel).max() > 100:
            raise FloatingPointError(f'unstable at t={d.time:.3f}')

    def force(self):
        return contact_force_on(self.model, self.data, self.tool)

    def run(self, seconds):
        for _ in range(int(seconds / self.dt)):
            self.step()


def press_cycle(x, tool_half, tool_xy, f_cap, depth_cap, creep_force=None, dt=0.0005):
    rig = PressRig(x, tool_half, tool_xy, dt)
    rig.run(1.0)
    while rig.force() < F_TOUCH and rig.target > -0.1:
        rig.target -= SPEED * dt
        rig.step()
    z0 = rig.z

    load = []
    while True:
        rig.target -= SPEED * dt
        rig.step()
        f, depth = rig.force(), z0 - rig.z
        load.append((depth, f))
        if f >= f_cap or depth >= depth_cap:
            break
    peak_depth = load[-1][0]

    hold = []
    for _ in range(int(3.0 / dt)):
        rig.step()
        hold.append(rig.force())

    unload = []
    while rig.target < z0 + 0.008:
        rig.target += SPEED * dt
        rig.step()
        unload.append((z0 - rig.z, rig.force()))

    load, unload = np.array(load), np.array(unload)
    touching = np.where(unload[:, 1] > F_TOUCH)[0]
    residual = float(unload[touching[-1], 0]) if len(touching) else 0.0
    unload_in = unload[unload[:, 0] >= 0]
    area_load = np.trapezoid(load[:, 1], load[:, 0])
    area_unload = -np.trapezoid(unload_in[:, 1], unload_in[:, 0])
    out = dict(load=load, unload=unload, peak_depth=peak_depth, peak_force=float(load[-1, 1]),
               hysteresis_ratio=float((area_load - area_unload) / max(area_load, 1e-9)),
               residual=residual,
               relax_ratio=float(np.mean(hold[-100:]) / max(np.mean(hold[:40]), 1e-9)))

    if creep_force is not None:
        rig.run(2.0)
        rig.mode = 'force'
        rig.hold_force = creep_force
        trace = []
        for _ in range(int(10.0 / dt)):
            rig.step()
            trace.append(rig.z)
        trace = np.array(trace)
        out['creep_trace'] = (z0 - trace)[:: int(0.02 / dt)]
        out['creep_depth'] = float(trace[int(0.5 / dt)] - trace[-1])
        out['creep_last_2s'] = float(trace[int(8.0 / dt)] - trace[-1])
    out['steps'] = rig.steps
    out['n_active'] = rig.n_active
    out['nv'] = int(rig.model.nv)
    return out


def squeeze_test(x, dt):
    """Compare the measured pressure-strain curve with the requested X."""
    area = 4 * x.half[0] * x.half[1]
    thick = 2 * x.half[2]
    f_cap = 300.0
    creep_force = x.p_creep * area if x.creep_eps > 0 else 30.0
    res = press_cycle(x, (x.half[0] + 0.03, x.half[1] + 0.03), (0, 0), f_cap,
                      max(thick * x.eps_lock * 1.2, 0.003), creep_force, dt)
    load = res['load']
    knee_depth = thick * x.eps_knee
    p_at_knee = float(np.interp(knee_depth, load[:, 0], load[:, 1]) / area) if load[-1, 0] >= knee_depth else None
    res.update(target_p_knee=x.p_knee, measured_p_knee=p_at_knee,
               target_eps_lock=x.eps_lock, peak_strain=res['peak_depth'] / thick,
               peak_pressure=res['peak_force'] / area,
               target_creep_depth=thick * x.creep_eps, area=area, thick=thick)
    return res


def finger_test(x, dt):
    """Same item, different contact sizes. The fingertip result is a prediction, not a fit."""
    cell_x, cell_y = 2 * x.half[0] / uc.GRID[0], 2 * x.half[1] / uc.GRID[1]
    area = 4 * x.half[0] * x.half[1]
    depth_ref = max(2 * x.half[2] * x.eps_lock * 0.5, 2e-4)
    cases = {'20mm center': ((.01, .01), (0, 0)),
             '20mm patch boundary': ((.01, .01), (cell_x / 2, cell_y / 2)),
             '60mm center': ((.03, .03), (0, 0)),
             'platen': ((x.half[0] + .03, x.half[1] + .03), (0, 0))}
    out = {}
    for name, (half, xy) in cases.items():
        res = press_cycle(x, half, xy, 40.0, max(depth_ref * 1.05, 0.003), None, dt)
        load = res['load']
        reached = load[-1, 0] >= depth_ref
        out[name] = dict(
            force_at_ref=float(np.interp(depth_ref, load[:, 0], load[:, 1])) if reached else None,
            depth_at_cap=float(load[-1, 0]), force_at_cap=float(load[-1, 1]),
            depth_at_5N=float(np.interp(5.0, load[:, 1], load[:, 0])) if load[:, 1].max() >= 5 else None,
            contact_area=float(min(4 * half[0] * half[1], area)), steps=res['steps'])
    out['depth_ref'] = depth_ref
    return out


def lift_test(x, dt, squeeze=20.0, lift=0.06, speed=0.1):
    """Pinch the overhanging end, lift it, hold. The far end stays on the table as a pivot."""
    overhang = 0.05
    table_edge = x.half[0] - overhang
    grip_x = x.half[0] - overhang / 2
    body, tendons, _ = uc.item_xml(x, pos=(0, 0, x.half[2] + 0.002))
    top = 2 * x.half[2] + 0.02
    xml = (uc.scene_header(dt) + '<worldbody><light pos="0 0 2"/>'
           f'<geom name="table" type="box" pos="{table_edge - 1} 0 -.05" size="1 .5 .05" rgba=".75 .78 .8 1" {uc.TOOL_CONTACT}/>' + body
           + f'<body name="carriage" pos="{grip_x} 0 0" gravcomp="1"><joint name="lift" type="slide" axis="0 0 1" armature="2"/>'
             '<inertial pos="0 0 0" mass=".05" diaginertia="1e-4 1e-4 1e-4"/>'
             '<body name="pivot" gravcomp="1"><joint name="wrist" type="hinge" axis="0 1 0" damping="0.05" armature="1e-3"/>'
             f'<geom name="jaw_low" type="box" pos="0 0 -.005" size=".02 .02 .005" mass=".02" rgba=".3 .5 .8 1" {uc.TOOL_CONTACT}/>'
             f'<body name="upper" pos="0 0 {top}" gravcomp="1"><joint name="close" type="slide" axis="0 0 1" armature="1"/>'
             f'<geom name="jaw_up" type="box" size=".02 .02 .005" mass=".02" rgba=".3 .5 .8 1" {uc.TOOL_CONTACT}/></body>'
             '</body></body>'
             f'</worldbody><tendon>{tendons}</tendon></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    dof_lift, dof_close = m.joint('lift').dofadr[0], m.joint('close').dofadr[0]
    q_lift = m.joint('lift').qposadr[0]
    item = m.body('item').id
    target = 0.0
    trace = []
    close_force = 0.0

    def step():
        d.qfrc_applied[dof_lift] = 2e4 * (target - d.qpos[q_lift]) - 400 * d.qvel[dof_lift]
        d.qfrc_applied[dof_close] = -close_force - 50 * d.qvel[dof_close]
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or np.abs(d.qvel).max() > 100:
            raise FloatingPointError(f'unstable at t={d.time:.3f}')

    def record():
        rot = d.xmat[item].reshape(3, 3)
        pitch = float(np.degrees(np.arctan2(-rot[2, 0], rot[0, 0])))
        grip_in_core = rot.T @ (d.xpos[m.body('pivot').id] - d.xpos[item])
        trace.append((d.time, float(d.qfrc_applied[dof_lift]), pitch, *uc.com_in_core(m, d), grip_in_core[0]))

    for k in range(int(2.0 / dt)):
        close_force = 0.0 if k < int(0.5 / dt) else min(squeeze, squeeze * (k * dt - 0.5) / 1.0)
        step()
    for _ in range(int(1.0 / dt)):
        step()
    start = len(trace)
    for k in range(int(lift / speed / dt)):
        target = speed * k * dt
        step()
        if k % 10 == 0:
            record()
    lift_end = len(trace)
    for k in range(int(4.0 / dt)):
        step()
        if k % 10 == 0:
            record()
    t = np.array(trace)
    com0 = t[0, 3]
    com = t[:, 3] - com0
    hold = t[lift_end:]
    hold_force = hold[:, 1]
    late = hold[hold[:, 0] > hold[0, 0] + 0.3]
    end_com = float(com[-1])
    settle = None
    if abs(end_com) > 1e-4:
        reached = np.where(np.abs(com - end_com) > 0.1 * abs(end_com))[0]
        settle = float(t[reached[-1], 0] - t[lift_end, 0])
    return dict(trace=t, force_end=float(np.mean(hold_force[-200:])), force_at_lift_end=float(np.mean(hold_force[:40])),
                force_swing_after_0p3s=float(late[:, 1].max() - late[:, 1].min()),
                com_at_lift_end=float(com[lift_end]),
                pitch_end=float(t[-1, 2]), com_shift_x=end_com, com_shift_max=float(np.abs(com).max()),
                com_settle_s=settle, grip_slip=float(abs(t[-1, 6] - t[0, 6])),
                lifted_share=float(np.mean(hold_force[-200:]) / (x.mass * 9.81)))


def finite(obj):
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dt', type=float, default=0.0005)
    ap.add_argument('--items', nargs='+', default=list(uc.PRESETS))
    ap.add_argument('--tests', nargs='+', default=['squeeze', 'finger', 'lift'])
    ap.add_argument('--out', default=str(HERE / 'results'))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    summary = {}
    for name in args.items:
        x = uc.PRESETS[name]
        summary[name] = {}
        arrays = {}
        for test, fn in (('squeeze', squeeze_test), ('finger', finger_test), ('lift', lift_test)):
            if test not in args.tests:
                continue
            t0 = time.perf_counter()
            try:
                res = fn(x, args.dt)
                for key, value in res.items():
                    if isinstance(value, np.ndarray):
                        arrays[f'{test}_{key}'] = value
                summary[name][test] = finite(res)
            except FloatingPointError as err:
                summary[name][test] = dict(FAILED=str(err))
            summary[name][test]['wall_s'] = round(time.perf_counter() - t0, 1)
            print(name, test, json.dumps(summary[name][test], ensure_ascii=False), flush=True)
        np.savez_compressed(out_dir / f'{name}.npz', **arrays)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

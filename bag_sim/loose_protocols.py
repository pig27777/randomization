"""Grasp tests of a content in a loose bag (mj_loose.py), MuJoCo 3.17 CPU.

  evaluate(object_spec, grasp_params, seed)   the frozen task (test/policy/README.md); the execution flow is a
                                              copy of policy/env.py evaluate() with the package = content + bag
  r2(...), r3(...)                            test/content_model/README.md R2 (edge grip) and R3 (carry)

Differences from policy/env.py, all forced by the bag (the rest is the same script, constants copied):
- grasp frame, pad start and the observed box come from the bag's top-view footprint (the package outline);
- COM = package COM (content + bag); the table load for "lower" is the table force on the package;
- slip_mm is the bag material at the pads relative to the pads; content_rel_mm (new) is the content COM
  relative to the pads from lift on -- the bag/content sliding a loose package shows;
- sim_fail follows the README: non-finite state, content-model limits, or a content segment centre outside the
  bag; a speed above 1 m/s only marks the log (`speed_flag`).
R2 grips the bag's -x short edge (flex: the mouth end) overhanging the table edge by 50 mm; R3 grips the long
sides at the middle. Both close to 20 N, keep 20 N while lifting (contents/segment/protocols.Closer).

CLI: python -B loose_protocols.py r2|r3|eval --bag plates|flex --content box|bendable|compressible ...
"""
from __future__ import annotations

import argparse
import json
import math
import time
import hashlib
import os
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from mj_loose import LooseBag, LooseSim, loose_scene
from model import G, PAD, ANCHORS, Kappa, build, load_table
from mj_adapter import OMEGA_MAX, PAD_T
from protocols import Closer, move_profile
from metrics import period_mean

OUT = Path(__file__).resolve().parent / "outputs" / "round2"     # round 1 results stay in outputs/round1/

# policy/env.py (test/policy/README.md), copied
THRESH = {"hard": math.inf, "bendable": 25.0, "compressible": 15.0}
CARRY, TARGET_HALF, LIFT_H, LOWER_V, PLACE_H = 0.5, 0.1, 0.10, 0.2, 0.005
LOST_T, OPEN, CLOSE_V, F_TOUCH = 0.05, 0.03, 0.025, 0.05
F_TAU, F_TAU_MEAS, SPIKE, GRIP_GAIN, BACKOFF = 0.005, 0.02, 20.0, 2.5e-3, 0.005
GAP, SETTLE, HOLD, CHECK = 0.002, 0.3, 0.2, 0.01
SQUEEZE_T = 8.0        # v1.1 (I-28): squeeze phase up to 8 s, was 4 s in round 1
# top grasp (v2): fingertip 20 x 40 mm, 8 mm thick (mj_loose.fingertip_xml); descent speed is not fixed by the
# protocol, 25 mm/s like the side approach; press at 10 mm/s
FT_H, FT_T, TOP_CLEAR, DESCEND_V, PRESS_V = 0.040, 0.004, 0.005, 0.025, 0.010
F_GRIP = 20.0          # R2 / R3
SPEED_FLAG = 1.0

# C1 contents (test/bag_model/REPORT.md): loose 120 x 80 x 30 mm 0.3 kg, full 200 x 150 x 30 mm 0.5 kg
BOX = dict(loose=Kappa(size=(0.12, 0.08, 0.03), mass=0.3, segments=(1, 1, 1), rigid=True, squeeze_stiffness_y=50.,
                       squeeze_stiffness_z=50., lock_strain_y=0.02, lock_strain_z=0.02, droop_ratio=0.0),
           full=Kappa(size=(0.20, 0.15, 0.03), mass=0.5, segments=(1, 1, 1), rigid=True, squeeze_stiffness_y=50.,
                      squeeze_stiffness_z=50., lock_strain_y=0.02, lock_strain_z=0.02, droop_ratio=0.0),
           # round 2 F4: tight fit, 10 mm to the 350 x 250 mm bag body on every side
           fit=Kappa(size=(0.33, 0.23, 0.03), mass=0.5, segments=(1, 1, 1), rigid=True, squeeze_stiffness_y=50.,
                     squeeze_stiffness_z=50., lock_strain_y=0.02, lock_strain_z=0.02, droop_ratio=0.0))
KRAFT_SIZE = (0.35, 0.25)


def content_kappa(name):
    return BOX[name.split("_")[1]] if name.startswith("box") else ANCHORS[name]


def _rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def rotate_package(sim: LooseSim, center, yaw):
    """Rotate every free body (content segments, plate roots) and every flex vertex about the vertical
    axis through center (policy/env.py _rotate_object, extended to slide-joint flex vertices)."""
    m, d = sim.m, sim.d
    q = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    R = _rot(yaw)
    pad_joints = {m.joint(f"{p}_{a}").id for p in sim.pad_names for a in "xyz"}
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        a = m.jnt_qposadr[j]
        p = d.qpos[a:a + 3].copy()
        p[:2] = center[:2] + R @ (p[:2] - center[:2])
        d.qpos[a:a + 3] = p
        qn = np.zeros(4)
        mujoco.mju_mulQuat(qn, q, d.qpos[a + 3:a + 7].copy())
        d.qpos[a + 3:a + 7] = qn
    if sim.bag.kind == "flex":
        for b in sim.bag_bodies:
            j0 = m.body_jntadr[b]
            if j0 < 0 or j0 in pad_joints:
                continue
            adr = m.jnt_qposadr[j0:j0 + 3]
            x = m.body_pos[b] + d.qpos[adr]
            x[:2] = center[:2] + R @ (x[:2] - center[:2])
            d.qpos[adr] = x - m.body_pos[b]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


def make(k: Kappa, bag: LooseBag, pads, table_xmin=-0.6, big_table=False, dt=None, fingertips=(), cache_dir=None):
    """dt: scene step (None: the content model's 0.4 ms). fingertips: v2 fingertips instead of the 40 mm pads."""
    desc = build(k, omega_max=OMEGA_MAX, table=load_table())
    if bag.kind == 'effect':
        from effect_model import effect_scene
        lb = effect_scene(desc, bag, pads, table_xmin=table_xmin, dt=dt, fingertips=fingertips)
    else:
        lb = loose_scene(desc, bag, pads, table_xmin=table_xmin, dt=dt, fingertips=fingertips)
    if big_table:      # policy/env.py: a 4 x 4 m table for the 0.5 m carry in any direction
        import re
        lb.built.xml = re.sub(r'<geom name="table"[^>]*?size="[^"]*"', lambda mt: re.sub(
            r'pos="[^"]*" size="[^"]*"', 'pos="0 0 -0.025" size="2 2 0.025"', mt.group(0)), lb.built.xml, count=1)
    if bag.kind == 'effect':
        from effect_model import EffectSim
        sim = EffectSim(lb, [p[0] for p in (pads or fingertips)])
    else:
        sim = LooseSim(lb, [p[0] for p in (pads or fingertips)])
    t0 = time.perf_counter()
    sim.cache_hit = False
    cache = None
    if cache_dir is not None:
        if not k.rigid:
            raise ValueError('F6 cache currently supports rigid content only')
        from mj_loose import CONTENT_RELEASE_SHA256
        key = hashlib.sha256((lb.built.xml + CONTENT_RELEASE_SHA256 + str(bag)).encode()).hexdigest()
        cache = Path(cache_dir) / (key + '.npz')
    state_type = mujoco.mjtState.mjSTATE_INTEGRATION
    if cache is not None and cache.exists():
        with np.load(cache) as saved:
            mujoco.mj_setState(sim.m, sim.d, saved['state'], state_type)
            warm = sim.d.qacc_warmstart.copy()
            mujoco.mj_forward(sim.m, sim.d)
            sim.d.qacc_warmstart[:] = warm
        sim.cache_hit = True
    else:
        sim.wrap()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            state = np.empty(mujoco.mj_stateSize(sim.m, state_type))
            mujoco.mj_getState(sim.m, sim.d, state, state_type)
            tmp = cache.with_name(cache.stem + f'.{os.getpid()}.npz')
            np.savez(tmp, state=state)
            tmp.replace(cache)
    sim.wrap_wall_s = time.perf_counter() - t0
    return sim


def nearest_bag_body(sim: LooseSim, p):
    if sim.bag.kind == 'effect':
        return int(sim.bag_bodies[0])
    x = sim.bag_points()
    return int(sim.bag_bodies[np.argmin(np.linalg.norm(x - p, axis=1))])


def package_height(sim: LooseSim):
    return float(max(sim.bag_points()[:, 2].max(), sim.seg_pos()[:, 2].max()) + (sim.bag.thickness / 2 if sim.bag.kind == "plates" else 0.001))


# ------------------------------------------------------------------ evaluate (task protocol)

def _cycloid(t, T, D):
    if t >= T:
        return D
    return D * (t / T - math.sin(2 * math.pi * t / T) / (2 * math.pi))


def _exit(p, d, half):
    t = math.inf
    for i in range(2):
        if abs(d[i]) > 1e-12:
            t = min(t, ((half[i] if d[i] > 0 else -half[i]) - p[i]) / d[i])
    return t


def _pad_reach(g, d, e, half):
    best = 0.0
    for s in np.linspace(-PAD / 2, PAD / 2, 9):
        p = g + s * e
        if np.all(np.abs(p) <= half + 1e-12):
            best = max(best, _exit(p, d, half))
    return best


def evaluate(object_spec: dict, grasp_params, seed: int, trace=False, observer=None) -> dict:
    """object_spec = {"kappa": Kappa dict, "cls": class, "bag": LooseBag dict, "dt": step or None}.
    grasp_params relative to the bag's top-view box:
      side grasp (v1.1, 40 mm pads): (u, v, psi, F, v_lift, a_max); flow policy/env.py evaluate(), 8 s squeeze;
      top grasp (v2, 20 x 40 mm fingertips): (u, v, psi, w, d, F, v_lift, a_max), see test/policy/README.md."""
    t_wall = time.perf_counter()
    top = len(grasp_params) == 8
    if top:
        u, v, psi, w_open, d_press, F, v_lift, a_max = [float(x) for x in grasp_params]
    else:
        u, v, psi, F, v_lift, a_max = [float(x) for x in grasp_params]
    k = Kappa.from_dict(object_spec["kappa"])
    bag = LooseBag(**object_spec["bag"]).resolved(k.size)
    thresh = THRESH[object_spec["cls"]]
    rng = np.random.default_rng(seed)
    carry_dir = float(rng.uniform(0, 2 * math.pi))
    park = [("pA", 1, (0.0, -1.0, 0.5)), ("pB", 1, (0.0, 1.0, 0.5))]
    sim = make(k, bag, [] if top else park, big_table=True, dt=object_spec.get("dt"), fingertips=park if top else (),
               cache_dir=object_spec.get('cache_dir'))
    cxy, (Lb, Wb), h = sim.footprint()          # observed top-view box (the bag is built along x)
    half = np.array([Lb, Wb]) / 2
    long_ax = np.array([1.0, 0.0]) if Lb >= Wb else np.array([0.0, 1.0])
    short_ax = np.array([-long_ax[1], long_ax[0]])
    g = u * 2 * half @ np.abs(long_ax) * long_ax + v * 2 * half @ np.abs(short_ax) * short_ax
    d_obj = math.cos(psi) * long_ax + math.sin(psi) * short_ax
    e_obj = np.array([d_obj[1], -d_obj[0]])
    reach_b = _pad_reach(g, d_obj, e_obj, half)
    reach_a = _pad_reach(g, -d_obj, e_obj, half)
    yaw = math.atan2(1.0, 0.0) - math.atan2(d_obj[1], d_obj[0])
    R = _rot(yaw)
    center = np.r_[cxy, 0.0]
    rotate_package(sim, center, yaw)
    gw = cxy + R @ g
    zc = h / 2
    pa0 = np.array([gw[0], gw[1] - reach_a - GAP - PAD_T, zc])
    pb0 = np.array([gw[0], gw[1] + reach_b + GAP + PAD_T, zc])
    if top:
        # open to w above the grasp point, fingertip bottoms TOP_CLEAR above the package top
        d_press = min(d_press, 0.6 * h)
        z0 = h + TOP_CLEAR + FT_H / 2
        pa0 = np.array([gw[0], gw[1] - w_open / 2 - FT_T, z0])
        pb0 = np.array([gw[0], gw[1] + w_open / 2 + FT_T, z0])
    m, dd = sim.m, sim.d
    dt = m.opt.timestep
    check_every = max(1, int(round(CHECK / dt)))
    carry_vec = np.array([math.cos(carry_dir), math.sin(carry_dir), 0.0]) * CARRY
    T_carry = math.sqrt(2 * math.pi * CARRY / a_max)
    max_travel = w_open / 2 if top else min(reach_a, reach_b)
    W_pkg = sim.pkg_mass * G
    z_grip, z_touch, f_vert = 0.0, None, 0.0    # top grasp: arm height offset, offset at touch, vertical force

    close = {"pA": 0.0, "pB": 0.0}
    sgn = {"pA": 1.0, "pB": -1.0}
    start = {"pA": pa0, "pB": pb0}
    arm = np.zeros(3)
    prev = {p: start[p].copy() for p in start}
    for p in start:
        sim.set_pad(p, start[p], np.zeros(3))
    mujoco.mj_forward(m, dd)
    touched = {}
    s_sq, regulate = 0.0, False
    f_ctrl = {"pA": 0.0, "pB": 0.0}
    f_meas, f_max = 0.0, 0.0
    phase, t_phase, t = "settle", 0.0, 0.0
    com_rest, lift_mm, place_h, content_lift_mm = None, None, None, None
    track, rel0, slip = None, None, 0.0
    ctrack, crel0, crel = None, None, 0.0
    outcome, reason, flag = None, None, None
    lost, n, pen = 0.0, 0, 0.0
    rows = []

    def go(next_phase):
        nonlocal phase, t_phase
        phase, t_phase = next_phase, t

    while outcome is None:
        tau = t - t_phase
        if phase == "settle":
            if tau >= SETTLE:
                com_rest = sim.pkg_com()
                com_c0 = sim.com()
                go("descend" if top else "approach")
        elif phase == "descend":
            # down to touch (> F_TOUCH on either fingertip); the arm stops at the table (bottoms at z = 0)
            if f_vert > F_TOUCH or pa0[2] + z_grip - FT_H / 2 <= 0.0:
                z_touch = z_grip
                go("press")
            else:
                z_grip -= DESCEND_V * dt
            arm = np.array([0.0, 0.0, z_grip])
        elif phase == "press":
            if z_grip <= z_touch - d_press or pa0[2] + z_grip - FT_H / 2 <= 0.0:
                touched = {"pA": 0.0, "pB": 0.0}
                regulate = True
                go("squeeze")
            else:
                z_grip -= PRESS_V * dt
            arm = np.array([0.0, 0.0, z_grip])
        elif phase == "approach":
            for p in close:
                if p not in touched:
                    if f_ctrl[p] > F_TOUCH:
                        touched[p] = close[p]
                    else:
                        close[p] += CLOSE_V * dt
            if len(touched) == 2:
                regulate = True
                go("squeeze")
            elif tau > 5.0:
                outcome, reason = "grasp_fail", "pads never touched"
        elif phase == "squeeze":
            if 0.5 * (f_ctrl["pA"] + f_ctrl["pB"]) >= 0.95 * F or s_sq >= max_travel or tau > SQUEEZE_T:
                regulate = True
                go("hold")
        elif phase == "hold":
            if tau >= HOLD:
                go("lift")
        elif phase == "lift":
            arm = np.array([0.0, 0.0, z_grip + min(v_lift * tau, LIFT_H)])
            if tau >= LIFT_H / v_lift + HOLD:
                lift_mm = float((sim.pkg_com()[2] - com_rest[2]) * 1e3)
                content_lift_mm = float((sim.com()[2] - com_c0[2]) * 1e3)
                if lift_mm < 0.5 * LIFT_H * 1e3:
                    outcome, reason = "grasp_fail", f"COM rose {lift_mm:.0f} mm"
                else:
                    go("carry")
        elif phase == "carry":
            arm = np.array([0.0, 0.0, z_grip + LIFT_H]) + _cycloid(tau, T_carry, 1.0) * carry_vec
            if tau >= T_carry + 0.3:
                if sim.pkg_com()[2] - com_rest[2] < 0.3 * LIFT_H:
                    outcome, reason = "drop", "COM near the table after carry"
                else:
                    go("lower")
        elif phase == "lower":
            if n % check_every == 0 and sim.table_force_pkg() > 0.3 * W_pkg:
                place_h = arm[2]
            if place_h is None:
                arm = np.array([0.0, 0.0, z_grip + max(LIFT_H - LOWER_V * tau, PLACE_H)]) + carry_vec
                if arm[2] <= z_grip + PLACE_H:
                    place_h = z_grip + PLACE_H
            if place_h is not None:
                regulate = False
                go("open")
        elif phase == "open":
            o = min(2 * CLOSE_V * tau, OPEN)
            arm[2] = place_h
            for p in close:
                close[p] = touched[p] + s_sq - o
            if tau >= OPEN / (2 * CLOSE_V) + 0.5:
                off = sim.pkg_com()[:2] - (com_rest[:2] + carry_vec[:2])
                local = _rot(-carry_dir) @ off
                if np.all(np.abs(local) <= TARGET_HALF):
                    outcome = "success"
                else:
                    outcome, reason = "drop", f"outside target ({local[0]*1e3:.0f}, {local[1]*1e3:.0f}) mm"
                break

        if regulate:
            err = F - 0.5 * (f_ctrl["pA"] + f_ctrl["pB"])
            s_sq = float(np.clip(s_sq + np.clip(GRIP_GAIN * err, -CLOSE_V, CLOSE_V) * dt, -BACKOFF, max_travel))
            for p in close:
                close[p] = touched[p] + s_sq

        vel = {}
        for p in close:
            x = start[p] + [0.0, sgn[p] * close[p], 0.0] + arm
            vel[p] = (x - prev[p]) / dt
            sim.set_pad(p, prev[p], vel[p])
            prev[p] = x
        sim.step()
        if observer is not None:
            observer(sim)
        t += dt
        n += 1
        raw = {p: sim.pad_normal_force(p, 1) for p in f_ctrl}
        if top and phase in ("descend", "press"):
            f_vert += dt / (F_TAU + dt) * (max(sim.pad_normal_force(p, 2) for p in f_ctrl) - f_vert)
        a5, a20 = dt / (F_TAU + dt), dt / (F_TAU_MEAS + dt)
        for p in f_ctrl:
            f_ctrl[p] += a5 * (min(raw[p], max(2 * F, f_ctrl[p] + SPIKE)) - f_ctrl[p])
        f_meas += a20 * (0.5 * (raw["pA"] + raw["pB"]) - f_meas)
        f_max = max(f_max, f_meas)
        if f_meas > thresh and phase != "open":
            outcome, reason = "over_squeeze", f"{f_meas:.1f} N in {phase}"
        if n % check_every == 0:
            gripped = phase in ("lift", "carry", "lower")
            if phase in ("lift", "carry"):
                lost = lost + CHECK if min(f_ctrl.values()) < F_TOUCH else 0.0
                if lost >= LOST_T and outcome is None:
                    outcome, reason = ("grasp_fail" if phase == "lift" else "drop"), f"pad contact lost in {phase}"
            pen = max(pen, sim.penetration())
            f, diag = sim.bag_failure(vel["pA"], gripped)
            if f is not None:
                outcome, reason = "sim_fail", f"{f} in {phase}"
            if diag is not None and flag is None:
                flag = f"{diag} in {phase}"
            grip = 0.5 * (sim.pad_pos("pA") + sim.pad_pos("pB"))
            if phase in ("lift", "carry"):
                if track is None:
                    b = nearest_bag_body(sim, grip)
                    track = (b, sim.point_local(b, grip))
                    ctrack = sim.com() - grip
                rel = sim.point_world(track[0], track[1]) - grip
                rel0 = rel if rel0 is None else rel0
                slip = max(slip, float(np.linalg.norm(rel - rel0)))
                crel = max(crel, float(np.linalg.norm(sim.com() - grip - ctrack)))
            if trace and n % (5 * check_every) == 0:
                rows.append(dict(t=round(t, 3), phase=phase, F=round(f_meas, 2), grip=np.round(grip, 4).tolist(),
                                 gap=round(float(np.linalg.norm(sim.pad_pos("pA") - sim.pad_pos("pB"))) * 1e3 - 2 * PAD_T * 1e3, 1),
                                 com=np.round(sim.pkg_com(), 4).tolist(), content=np.round(sim.com(), 4).tolist()))
        if t > 30.0 and outcome is None:
            outcome, reason = "sim_fail", "timeout"

    out = {"outcome": outcome, "max_grip_force_N": f_max, "slip_mm": slip * 1e3, "lift_height_mm": lift_mm,
           "wall_s": time.perf_counter() - t_wall, "sim_s": t, "reason": reason, "cls": object_spec["cls"],
           "content_rel_mm": crel * 1e3, "speed_flag": flag, "wrap_wall_s": sim.wrap_wall_s,
           "outside": sim.outside_fraction(), "penetration_mm": pen * 1e3, "dt": dt, "nv": sim.m.nv,
           "mode": "top" if top else "side", "press_mm": (z_touch - z_grip) * 1e3 if z_touch is not None else None,
           "content_lift_mm": content_lift_mm, "content_in_bag": sim.outside_fraction()[0] == 0}
    out.update(cache_hit=sim.cache_hit, max_ncon=sim.max_ncon, max_nefc=sim.max_nefc,
               step_ms=1e3 * sim.step_wall_s / max(1, sim.measured_steps),
               mean_iterations=sim.solver_iterations / max(1, sim.measured_steps))
    if trace:
        out["trace"] = rows
    return out


# ------------------------------------------------------------------ R2 / R3

def _grip_run(sim: LooseSim, pads, direction, travel, phases, normal, record_every=0.01):
    """Close (protocols.Closer, 20 N), then run the arm phases keeping 20 N. Records package, bag and
    content motion relative to the grip; the bag reference point is the bag body nearest the grip at lift."""
    dt = sim.m.opt.timestep
    start = {p[0]: np.array(p[2], float) for p in pads}
    for p, x in start.items():
        sim.set_pad(p, x, np.zeros(3))
    mujoco.mj_forward(sim.m, sim.d)
    ctrl = Closer(start, direction, 0.025, travel, F_GRIP)
    prev = {p: x.copy() for p, x in start.items()}
    forces = {p: 0.0 for p in start}
    rec_every = max(1, int(round(record_every / dt)))
    rows, ev = [], {}
    t, n, closed, base, t_phase, phase_i = 0.0, 0, None, None, None, 0
    fail, flag, track = None, None, None
    pen = 0.0
    wall = time.perf_counter()
    while fail is None and t < 40:
        gripped = False
        if t < SETTLE:
            tgt = start
        elif closed is None or t < closed + 0.2:
            tgt = ctrl(dt, forces, t)
            if ctrl.frozen and closed is None:
                closed = t
                ev["closed"] = t
                base = {p: tgt[p].copy() for p in tgt}
            if t - SETTLE > 8.0 and closed is None:
                ev["close_timeout"] = t
                closed = t
                base = {p: tgt[p].copy() for p in tgt}
        else:
            if t_phase is None:
                t_phase = t
                ev[phases[0][0]] = t
            while phase_i < len(phases) and t - t_phase >= phases[phase_i][1] - 1e-12:
                t_phase += phases[phase_i][1]
                phase_i += 1
                if phase_i < len(phases):
                    ev[phases[phase_i][0]] = t_phase
            if phase_i >= len(phases):
                break
            o = phases[phase_i][2](t - t_phase)
            held = ctrl.hold(dt, forces, base)
            tgt = {p: held[p] + o for p in base}
            gripped = True
        grip = np.mean([sim.pad_pos(p) for p in sim.pad_names], axis=0)
        if track is None and "lift" in ev:
            b = nearest_bag_body(sim, grip)
            track = (b, sim.point_local(b, grip), sim.com() - grip)
        pad_vel = np.zeros(3)
        for p, x in tgt.items():
            x = np.asarray(x, float)
            pad_vel = (x - prev[p]) / dt
            sim.set_pad(p, prev[p], pad_vel)
            prev[p] = x
        sim.step()
        t += dt
        n += 1
        forces = {p: sim.pad_normal_force(p, normal) for p in sim.pad_names}
        if n % rec_every == 0:
            pen = max(pen, sim.penetration())
            f, diag = sim.bag_failure(pad_vel, gripped)
            if f is not None:
                fail = f"{f} at {t:.2f} s"
            if diag is not None and flag is None:
                flag = f"{diag} at {t:.2f} s"
            grip = np.mean([sim.pad_pos(p) for p in sim.pad_names], axis=0)
            row = dict(t=t, F=float(np.mean(list(forces.values()))), F_min=float(min(forces.values())), grip=grip,
                       com=sim.pkg_com(), content=sim.com(), table=sim.table_force_pkg(),
                       bag=sim.bag_points().copy())
            if track is not None:
                row["bag_slip"] = float(np.linalg.norm(sim.point_world(track[0], track[1]) - grip))
                row["content_rel"] = sim.com() - grip - track[2]
                row["content_dist"] = float(np.linalg.norm(sim.com() - grip))
                row["content_dist0"] = float(np.linalg.norm(track[2]))
            rows.append(row)
    return dict(rows=rows, ev=ev, fail=fail, flag=flag, sim_t=t, wall_s=time.perf_counter() - wall, penetration=pen,
                outside=sim.outside_fraction(), touch=dict(ctrl.touch_s), closed=closed)


def r2(content="box_loose", bag: LooseBag = LooseBag(), dt=None, end=-1):
    """end = -1 grips the -x end (flex: the mouth/flap end of kraft.npz), +1 the +x end (the package is turned
    by pi first, so the gripped end is always at -x next to the table edge)."""
    k = content_kappa(content)
    bag_r = bag.resolved(k.size)

    def built(pads, table_xmin=-0.6):
        s = make(k, bag, pads, table_xmin=table_xmin, dt=dt)
        if end > 0:
            c, _, _ = s.footprint()
            rotate_package(s, np.r_[c, 0.0], math.pi)
        return s

    sim = built([("pA", 2, (0, 0, 0.5)), ("pB", 2, (0, 0, -0.5))])
    (cxy, size, _) = sim.footprint()
    x_edge = cxy[0] - size[0] / 2
    x0 = x_edge + PAD / 2
    # rebuild with the table edge 50 mm inside the gripped end
    sim = built([("pA", 2, (x0, cxy[1], 0.5)), ("pB", 2, (x0, cxy[1], -0.5))], table_xmin=x_edge + 0.05)
    pts = sim.bag_points()
    near = pts[np.abs(pts[:, 0] - x0) < PAD / 2 + 0.01]
    top = float(near[:, 2].max()) + 0.002
    pads = [("pA", 2, (x0, cxy[1], top + GAP + PAD_T)), ("pB", 2, (x0, cxy[1], -GAP - PAD_T))]
    lift_h = max(0.1, size[0] + 0.05)
    sim.length0 = float(size[0])
    direction = {"pA": np.array([0, 0, -1.0]), "pB": np.array([0, 0, 1.0])}
    phases = [("lift", lift_h / 0.05, lambda t: np.array([0, 0, min(0.05 * t, lift_h)])),
              ("hold", 2.0, lambda t: np.array([0, 0, lift_h]))]
    run = _grip_run(sim, pads, direction, 0.01, phases, 2)
    out = _summary("R2", run, sim, k, bag_r, content)
    out.update(end=end, dt=sim.m.opt.timestep)
    return out


def r3(content="box_loose", bag: LooseBag = LooseBag(), dt=None):
    k = content_kappa(content)
    bag_r = bag.resolved(k.size)
    sim = make(k, bag, [("pA", 1, (0, -1, 0.5)), ("pB", 1, (0, 1, 0.5))], dt=dt)
    (cxy, size, h) = sim.footprint()
    W = size[1]
    pads = [("pA", 1, (cxy[0], cxy[1] - (W / 2 + GAP + PAD_T), h / 2)),
            ("pB", 1, (cxy[0], cxy[1] + W / 2 + GAP + PAD_T, h / 2))]
    direction = {"pA": np.array([0, 1.0, 0]), "pB": np.array([0, -1.0, 0])}
    up, T_move = 0.05, 0.8
    phases = [("lift", 1.0, lambda t: np.array([0, 0, min(0.05 * t, up)])),
              ("pre", 0.5, lambda t: np.array([0, 0, up])),
              ("move", T_move, lambda t: np.array([move_profile(t), 0, up])),
              ("hold", 1.0, lambda t: np.array([move_profile(T_move), 0, up]))]
    run = _grip_run(sim, pads, direction, 0.45 * W, phases, 1)
    out = _summary("R3", run, sim, k, bag_r, content)
    out["dt"] = sim.m.opt.timestep
    out['sink_mm'] = 50. - out['content_com_rise_mm'] if out.get('completed') else None
    return out


def _summary(test, run, sim: LooseSim, k, bag, content):
    rows, ev = run["rows"], run["ev"]
    out = {"test": test, "content": content, "bag": asdict(bag), "fail": run["fail"], "speed_flag": run["flag"],
           "sim_t": run["sim_t"], "wall_s": run["wall_s"], "rtf": run["sim_t"] / run["wall_s"],
           "wrap_wall_s": sim.wrap_wall_s, "outside_end": run["outside"], "events": {a: round(b, 3) for a, b in ev.items()},
           "pkg_mass": sim.pkg_mass, "nv": sim.m.nv, "penetration_mm": run["penetration"] * 1e3}
    out.update(max_ncon=sim.max_ncon, max_nefc=sim.max_nefc,
               step_ms=1e3 * sim.step_wall_s / max(1, sim.measured_steps),
               mean_iterations=sim.solver_iterations / max(1, sim.measured_steps))
    if run["fail"] or "hold" not in ev:
        out["completed"] = False
        return out
    W = sim.pkg_mass * G
    hold = [r for r in rows if r["t"] >= ev["hold"]]
    lifted = [r for r in rows if r["t"] >= ev["lift"]]
    last = [r for r in rows if r["t"] >= hold[-1]["t"] - 0.5]
    com_rise = float(np.mean([r["com"][2] for r in last]) - rows[0]["com"][2])
    crel = np.array([r["content_rel"] for r in lifted if "content_rel" in r])
    out.update({
        "completed": True,
        "grip_force_hold": float(np.mean([r["F"] for r in last])),
        "pad_gap_mm": float(np.linalg.norm(sim.pad_pos("pA") - sim.pad_pos("pB")) * 1e3) - 2 * PAD_T * 1e3,
        "dropped": bool(last[-1]["F_min"] < F_TOUCH),
        "table_touch_hold": bool(max(r["table"] for r in hold) > 0.01 * W),
        "pkg_com_rise_mm": com_rise * 1e3,
        "content_com_rise_mm": float(np.mean([r["content"][2] for r in last]) - rows[0]["content"][2]) * 1e3,
        "bag_slip_mm": float(max(r.get("bag_slip", 0) for r in lifted) * 1e3),
        "content_rel_max_mm": float(np.linalg.norm(crel, axis=1).max() * 1e3) if len(crel) else None,
        "content_rel_end_mm": (crel[-1] * 1e3).round(1).tolist() if len(crel) else None,
    })
    if test == "R2":
        # change of the content-to-grip distance: sliding along the bag, without the bag's swing to hanging
        tracked = [r for r in lifted if 'content_dist0' in r]
        out["slide_mm"] = float(np.mean([r["content_dist"] for r in last]) - tracked[0]["content_dist0"]) * 1e3
        tt = [r["t"] for r in rows]
        tip = [r["grip"][2] - r["bag"][:, 2].min() for r in rows]
        out["droop_bag"] = period_mean(tt, tip, ev["hold"], ev["hold"] + 2.0) / sim.length0
    else:
        after = [r for r in rows if r["t"] >= ev["hold"]]
        rel = np.array([r["com"] - r["grip"] for r in after])
        ref = rel[-int(0.2 / 0.01):].mean(0)
        dev = np.linalg.norm(rel - ref, axis=1)
        mv = [r for r in rows if r["t"] >= ev["move"]]
        F = np.array([r["F"] for r in mv])
        out.update(com_swing_mm=float(dev.max() * 1e3), force_p2p=float(F.max() - F.min()),
                   grip_force_pre=float(np.mean([r["F"] for r in rows if ev["pre"] <= r["t"] < ev["move"]])))
    return out


# ------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=["r2", "r3", "eval"])
    ap.add_argument("--bag", default="plates", choices=["plates", "flex"])
    ap.add_argument("--content", default="box_loose")
    ap.add_argument("--fill", type=float, default=None, help="default: kraft size for boxes, 0.75 for anchors")
    ap.add_argument("--set", default="{}", help="JSON overrides of LooseBag fields")
    ap.add_argument("--params", default="[0, 0, 1.5708, 20, 0.1, 2]", help="eval: u v psi F v_lift a_max")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cls", default=None)
    ap.add_argument("--out", default=None, help="append the JSON line to this file (in outputs/round1/)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dt", type=float, default=None, help="scene step (default: the content model's 0.4 ms)")
    ap.add_argument("--end", type=int, default=None, help="r2: -1 / +1 end (default: +1 for flex, -1 for plates)")
    a = ap.parse_args()
    k = content_kappa(a.content)
    if a.fill is None and a.content.startswith("box"):
        bag = LooseBag(kind=a.bag, size=KRAFT_SIZE)
    else:
        bag = LooseBag(kind=a.bag, fill=a.fill or 0.75)
    bag = replace(bag, **json.loads(a.set))
    if a.test == "r2":
        res = r2(a.content, bag, a.dt, a.end if a.end is not None else (1 if a.bag == "flex" else -1))
    elif a.test == "r3":
        res = r3(a.content, bag, a.dt)
    else:
        cls = a.cls or ("hard" if k.rigid else a.content)
        spec = {"kappa": k.to_dict(), "cls": cls, "bag": asdict(bag), "dt": a.dt}
        res = evaluate(spec, json.loads(a.params), a.seed)
        res.update(content=a.content, params=json.loads(a.params), seed=a.seed, bag=asdict(bag.resolved(k.size)))
    res["tag"] = a.tag
    line = json.dumps(res, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    print(line, flush=True)
    if a.out:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / a.out, "a") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()

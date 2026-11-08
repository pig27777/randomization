"""Grasp-carry-place task on the reduced-order content model (test/policy/README.md), CPU MuJoCo.

evaluate(object_spec, grasp_params, seed) -> outcome dict. object_spec is a dict
{"kappa": Kappa dict, "cls": "hard" | "bendable" | "compressible", "friction": pad-package friction}.

Choices not fixed by the README (recorded in test/policy/REPORT.md, round 1):
- the scene is rotated so the gripper always closes along sim y; the object yaw follows from psi, and the seed
  draws the placement yaw, i.e. the carry direction relative to the gripper;
- gripper: each pad approaches at 25 mm/s until it touches, then both pads move together with the force error
  (force-limited gripper, GRIP_MODE "force"): the grip settles at F and is kept at F while lifting and carrying.
  Holding the position where F was first reached ("position") lets the grip of soft objects relax to zero on
  lifting, so no soft anchor can be picked up at any F below its threshold;
- pad-object contact stiffness is scaled by PAD_K_SCALE (rubber fingertip, about 40 N/mm); with the content
  model's 3e5 N/m per contact point a rigid box jumps far past F on contact;
- lift 0.1 m at the lift speed, carry 0.5 m on a cycloidal profile with peak acceleration a_max, lower at
  0.2 m/s until the table carries 30 % of the weight, open 30 mm per pad, wait 0.5 s;
- force reading = mean pad normal force, 20 ms low-pass; over_squeeze as soon as it exceeds the class threshold;
  the controller uses a 5 ms filter with single-step contact spikes clipped;
- grasp_fail: COM rose < 50 % of the lift height, or both-pad contact lost while lifting; drop: contact lost
  while carrying or COM back near the table; success: COM within the 0.2 x 0.2 m target box after release;
- sim_fail: the content model's failure check (test/content_model/README.md), every 10 ms;
- surface friction is not in kappa yet: pad-package friction is drawn log-uniform in FRICTION_RANGE for R1 and R3.
"""
from __future__ import annotations

import copy
import math
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

# frozen copy of contents/segment (the content line is changing it during round 2), see content_snapshot/README.md
sys.path.insert(0, str(Path(__file__).resolve().parent / "content_snapshot"))
from model import G, PAD, Kappa, build, load_table, sample_kappa  # noqa: E402
from mj_adapter import OMEGA_MAX, PAD_T, Sim, scene  # noqa: E402

# task (test/policy/README.md)
THRESH = {"hard": math.inf, "bendable": 25.0, "compressible": 15.0}
F_RANGE = (2.0, 50.0)
VLIFT_RANGE = (0.05, 0.3)
ACC_RANGE = (0.5, 4.0)
CARRY = 0.5
TARGET_HALF = 0.1
# execution script
LIFT_H = 0.10
LOWER_V = 0.2
PLACE_H = 0.005          # release this far above the pick-up height
LOST_T = 0.05            # both-pad contact lost this long while carrying -> drop
OPEN = 0.03
CLOSE_V = 0.025
F_TOUCH = 0.05
F_TAU = 0.005            # gripper controller force filter
F_TAU_MEAS = 0.02        # force reading (max_grip_force_N, over-squeeze)
SPIKE = 20.0             # controller ignores single-step contact spikes above max(2F, filtered + SPIKE)
GRIP_GAIN = 2.5e-3       # m/s per N of force error; stable against ~4e4 N/m pad stiffness with the 5 ms filter
BACKOFF = 0.005          # the gap may open this far past the touch point when the force exceeds F
GAP = 0.002
SETTLE = 0.3
HOLD = 0.2
CHECK = 0.01
# gripper model (see module docstring). "force": after reaching F the gripper keeps closing whenever the grip
# force drops below F (force-limited electric gripper); "position": it holds the position where F was reached.
GRIP_MODE = "force"
# rubber fingertip: pad-object contact stiffness scaled down from the content model's 3e5 N/m per point
PAD_K_SCALE = 0.1
# surface friction (not in kappa yet, package_params.md "等效外皮"): same draw for R1 and R3
FRICTION_RANGE = (0.4, 1.0)

_TABLE = None


def table():
    global _TABLE
    if _TABLE is None:
        _TABLE = load_table()
    return _TABLE


# ------------------------------------------------------------------ object distributions

def sample_object(rng: np.random.Generator, dist: str) -> dict:
    """R3: the content line's kappa sampler as is. R1: same sizes / mass / COM, made a single rigid box."""
    cls = str(copy.deepcopy(rng).choice(["hard", "bendable", "compressible"], p=[0.2, 0.4, 0.4]))  # = sampler's class
    k = sample_kappa(rng)
    mu = float(math.exp(rng.uniform(*np.log(FRICTION_RANGE))))
    if dist == "R1":
        k = replace(k, rigid=True, mobile=False, segments=(1, 1, 1))
        cls = "hard"
    return {"kappa": k.to_dict(), "cls": cls, "friction": mu}


def observe(spec: dict) -> np.ndarray:
    """Top-view bounding box of the package: (long side, short side, height) in m."""
    L = spec["kappa"]["size"]
    return np.array([max(L[0], L[1]), min(L[0], L[1]), L[2]])


# ------------------------------------------------------------------ geometry helpers

def _rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _exit(p, d, half):
    """Distance from p (inside the rectangle |x| <= half) along d to its boundary."""
    t = math.inf
    for i in range(2):
        if abs(d[i]) > 1e-12:
            t = min(t, ((half[i] if d[i] > 0 else -half[i]) - p[i]) / d[i])
    return t


def _pad_reach(g, d, e, half):
    """Largest boundary distance along d over the pad footprint (offsets along e) that lies on the object."""
    best = 0.0
    for s in np.linspace(-PAD / 2, PAD / 2, 9):
        p = g + s * e
        if np.all(np.abs(p) <= half + 1e-12):
            best = max(best, _exit(p, d, half))
    return best


def _rotate_object(sim: Sim, center, yaw):
    """Rotate every free body of the object about the vertical axis through center."""
    m, d = sim.m, sim.d
    q = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    R = _rot(yaw)
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
    mujoco.mj_forward(m, d)


# ------------------------------------------------------------------ evaluate

def _cycloid(t, T, D):
    """Smooth point-to-point: peak acceleration 2*pi*D/T^2."""
    if t >= T:
        return D
    return D * (t / T - math.sin(2 * math.pi * t / T) / (2 * math.pi))


def evaluate(object_spec: dict, grasp_params, seed: int) -> dict:
    """grasp_params = (u, v, psi, F, v_lift, a_max), ranges as in test/policy/README.md."""
    t_wall = time.perf_counter()
    u, v, psi, F, v_lift, a_max = [float(x) for x in grasp_params]
    k = Kappa.from_dict(object_spec["kappa"])
    thresh = THRESH[object_spec["cls"]]
    rng = np.random.default_rng(seed)
    carry_dir = float(rng.uniform(0, 2 * math.pi))      # placement yaw relative to the gripper
    Lx, Ly, Lz = k.size
    half = np.array([Lx, Ly]) / 2

    # grasp frame in object coordinates: u along the long side, v along the short side, psi = angle between the
    # closing direction and the long side
    long_ax = np.array([1.0, 0.0]) if Lx >= Ly else np.array([0.0, 1.0])
    short_ax = np.array([-long_ax[1], long_ax[0]])
    g = u * 2 * half @ np.abs(long_ax) * long_ax + v * 2 * half @ np.abs(short_ax) * short_ax
    d_obj = math.cos(psi) * long_ax + math.sin(psi) * short_ax          # closing direction (pad A -> B)
    e_obj = np.array([d_obj[1], -d_obj[0]])                            # pad face direction
    reach_b = _pad_reach(g, d_obj, e_obj, half)
    reach_a = _pad_reach(g, -d_obj, e_obj, half)
    # object yaw that maps the closing direction onto sim +y
    yaw = math.atan2(1.0, 0.0) - math.atan2(d_obj[1], d_obj[0])
    R = _rot(yaw)
    center = np.array([0.0, 0.0, Lz / 2 + 1e-4])
    gw = R @ g
    zc = Lz / 2
    pa0 = np.array([gw[0], gw[1] - reach_a - GAP - PAD_T, zc])
    pb0 = np.array([gw[0], gw[1] + reach_b + GAP + PAD_T, zc])
    pads = [("pA", 1, tuple(pa0)), ("pB", 1, tuple(pb0))]

    desc = build(k, omega_max=OMEGA_MAX, table=table())
    built = scene(desc, center, pads)
    built.xml = re.sub(r'<geom name="table"[^>]*?size="[^"]*"', lambda mt: re.sub(
        r'pos="[^"]*" size="[^"]*"', 'pos="0 0 -0.025" size="2 2 0.025"', mt.group(0)), built.xml, count=1)
    sim = Sim(built, ["pA", "pB"])
    sim.m.pair_friction[:, 0:2] = object_spec["friction"]
    sim.m.pair_solref[:, 0] *= PAD_K_SCALE
    sim.m.pair_solref[:, 1] *= math.sqrt(PAD_K_SCALE)
    _rotate_object(sim, center, yaw)
    m, dd = sim.m, sim.d
    dt = m.opt.timestep
    check_every = max(1, int(round(CHECK / dt)))
    carry_vec = np.array([math.cos(carry_dir), math.sin(carry_dir), 0.0]) * CARRY
    T_carry = math.sqrt(2 * math.pi * CARRY / a_max)
    max_travel = min(reach_a, reach_b)                  # pads may close to the grasp point at most

    # pad position = start + closing along y (approach, then common squeeze s_sq) + arm offset (lift, carry, ...)
    close = {"pA": 0.0, "pB": 0.0}
    sgn = {"pA": 1.0, "pB": -1.0}
    start = {"pA": pa0, "pB": pb0}
    arm = np.zeros(3)
    prev = {p: start[p].copy() for p in start}
    touched = {}
    s_sq = 0.0
    regulate = False
    f_ctrl = {"pA": 0.0, "pB": 0.0}                     # 5 ms filter, spikes clipped: gripper controller
    f_meas = 0.0                                        # 20 ms filter of the mean: force reading, over-squeeze
    f_max = 0.0
    phase, t_phase, t = "settle", 0.0, 0.0
    com_rest, lift_mm, place_h = None, None, None
    track, rel0, slip = None, None, 0.0
    outcome, reason = None, None
    lost, n = 0.0, 0

    def go(next_phase):
        nonlocal phase, t_phase
        phase, t_phase = next_phase, t

    while outcome is None:
        tau = t - t_phase
        if phase == "settle":
            if tau >= SETTLE:
                com_rest = sim.com()
                go("approach")
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
            if 0.5 * (f_ctrl["pA"] + f_ctrl["pB"]) >= 0.95 * F or s_sq >= max_travel or tau > 4.0:
                regulate = GRIP_MODE == "force"
                go("hold")
        elif phase == "hold":
            if tau >= HOLD:
                go("lift")
        elif phase == "lift":
            arm = np.array([0.0, 0.0, min(v_lift * tau, LIFT_H)])
            if tau >= LIFT_H / v_lift + HOLD:
                lift_mm = float((sim.com()[2] - com_rest[2]) * 1e3)
                if lift_mm < 0.5 * LIFT_H * 1e3:
                    outcome, reason = "grasp_fail", f"COM rose {lift_mm:.0f} mm"
                else:
                    go("carry")
        elif phase == "carry":
            arm = np.array([0.0, 0.0, LIFT_H]) + _cycloid(tau, T_carry, 1.0) * carry_vec
            if tau >= T_carry + 0.3:
                if sim.com()[2] - com_rest[2] < 0.3 * LIFT_H:
                    outcome, reason = "drop", "COM near the table after carry"
                else:
                    go("lower")
        elif phase == "lower":
            # lower until the table carries part of the weight (or PLACE_H above the pick-up height), then open
            if n % check_every == 0 and sim.contact_forces()[1][2] > 0.3 * sim.mass * G:
                place_h = arm[2]
            if place_h is None:
                arm = np.array([0.0, 0.0, max(LIFT_H - LOWER_V * tau, PLACE_H)]) + carry_vec
                if arm[2] <= PLACE_H:
                    place_h = PLACE_H
            if place_h is not None:
                regulate = False
                go("open")
        elif phase == "open":
            o = min(2 * CLOSE_V * tau, OPEN)
            arm[2] = place_h
            for p in close:
                close[p] = touched[p] + s_sq - o
            if tau >= OPEN / (2 * CLOSE_V) + 0.5:
                off = sim.com()[:2] - (com_rest[:2] + carry_vec[:2])
                local = _rot(-carry_dir) @ off               # target box is aligned with the world axes
                if np.all(np.abs(local) <= TARGET_HALF):
                    outcome = "success"
                else:
                    outcome, reason = "drop", f"outside target ({local[0]*1e3:.0f}, {local[1]*1e3:.0f}) mm"
                break

        if regulate:
            # force-limited gripper: the gap moves with the force error, both pads together
            err = F - 0.5 * (f_ctrl["pA"] + f_ctrl["pB"])
            s_sq = float(np.clip(s_sq + np.clip(GRIP_GAIN * err, -CLOSE_V, CLOSE_V) * dt, -BACKOFF, max_travel))
            for p in close:
                close[p] = touched[p] + s_sq

        # step with prescribed pad motion
        vel = {}
        for p in close:
            x = start[p] + [0.0, sgn[p] * close[p], 0.0] + arm
            vel[p] = (x - prev[p]) / dt
            sim.set_pad(p, prev[p], vel[p])
            prev[p] = x
        mujoco.mj_step(m, dd)
        t += dt
        n += 1
        raw = {p: sim.pad_normal_force(p, 1) for p in f_ctrl}
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
            f = sim.failure(vel["pA"], gripped)
            if f is not None:
                outcome, reason = "sim_fail", f"{f} in {phase}"
            if phase in ("lift", "carry"):
                grip = 0.5 * (sim.pad_pos("pA") + sim.pad_pos("pB"))
                if track is None:
                    xb = sim.d.xpos[sim.seg_bodies]
                    b = sim.seg_bodies[int(np.argmin(np.linalg.norm(xb - grip, axis=1)))]
                    track = (b, sim.point_local(b, grip))
                rel = sim.point_world(track[0], track[1]) - grip
                if rel0 is None:
                    rel0 = rel
                slip = max(slip, float(np.linalg.norm(rel - rel0)))
        if t > 30.0 and outcome is None:
            outcome, reason = "sim_fail", "timeout"

    return {"outcome": outcome, "max_grip_force_N": f_max, "slip_mm": slip * 1e3, "lift_height_mm": lift_mm,
            "wall_s": time.perf_counter() - t_wall, "sim_s": t, "reason": reason, "cls": object_spec["cls"]}

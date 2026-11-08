"""R1 / R2 / R3, rest and energy tests on the MuJoCo adapter (test/content_model/README.md).

README 2026-11-04: pads approach until each touches, then squeeze together; R1 stops at 50 % or the 50 N
limit, R2/R3 close at 50 mm/s to a 20 N grip force. R2 lifts max(0.1, Lx + 0.05) m with the gripped end
overhanging the table edge by 50 mm. Implementation details: the force is low-pass filtered (F_TAU) and the
squeeze slows down near the stop force; R1 closes at 5 mm/s per pad for y (10 mm/s relative).
"""
from __future__ import annotations

import math
import time

import mujoco
import numpy as np

from metrics import period_mean, r1_metrics
from model import G, PAD, Kappa, build
from mj_adapter import DT, OMEGA_MAX, PAD_T, Sim, scene

F_CAP = 50.0
F_GRIP = 20.0      # R2 / R3 grip force
F_TOUCH = 0.05
F_TAU = 0.005
GAP = 0.002
SETTLE = 0.3
REC = 0.01          # record interval (s)


def make(k: Kappa, pads, table=True, table_xmin=-0.6, gravity=True, internal=None, dt=DT, desc=None):
    if desc is None:
        desc = build(k, internal=internal, omega_max=OMEGA_MAX)
    origin = (0.0, 0.0, k.size[2] / 2 + 1e-4) if table else (0.0, 0.0, 0.5)
    built = scene(desc, origin, pads, table=table, table_xmin=table_xmin, gravity=gravity, dt=dt)
    return Sim(built, [p[0] for p in pads])


def end_point(sim: Sim, sign):
    """Center of the +x (sign=1) or -x end face, averaged over the end segments."""
    k = sim.desc.kappa
    if sim.desc.rigid:
        return sim.point_world(sim.seg_bodies[0], np.array([sign * k.size[0] / 2, 0, 0]))
    nx = k.segments[0]
    i_end = nx - 1 if sign > 0 else 0
    pts = [sim.point_world(sim.seg_bodies[n], np.array([sign * s.half[0], 0, 0]))
           for n, s in enumerate(sim.desc.segments) if s.idx[0] == i_end]
    return np.mean(pts, axis=0)


def nearest_body(sim: Sim, p):
    pos = sim.seg_pos()
    return sim.seg_bodies[int(np.argmin(np.linalg.norm(pos - p, axis=1)))]


class Runner:
    """Steps a Sim with a pad controller, records every REC seconds and checks failure."""

    def __init__(self, sim: Sim):
        self.sim = sim
        self.t = 0.0
        self.fail = None
        self.fail_t = None
        self.rec_every = max(1, int(round(REC / sim.m.opt.timestep)))
        self.n = 0
        self.prev_pad = {p: sim.pad_pos(p) for p in sim.pad_names}
        self.wall = 0.0

    def step(self, pad_targets: dict, gripped=False):
        sim = self.sim
        dt = sim.m.opt.timestep
        pad_vel = np.zeros(3)
        for p, x in pad_targets.items():
            x = np.asarray(x, float)
            pad_vel = (x - self.prev_pad[p]) / dt
            # prescribe the end-of-step pose: start this step at the previous target moving with pad_vel
            sim.set_pad(p, self.prev_pad[p], pad_vel)
            self.prev_pad[p] = x
        t0 = time.perf_counter()
        mujoco.mj_step(sim.m, sim.d)
        self.wall += time.perf_counter() - t0
        self.t += dt
        self.n += 1
        if self.n % self.rec_every == 0 and self.fail is None:
            f = sim.failure(pad_vel, gripped)
            if f is not None:
                self.fail, self.fail_t = f, self.t
        return self.n % self.rec_every == 0

    @property
    def rtf(self):
        return self.t / max(self.wall, 1e-9)


class Closer:
    """Gripper closing: each pad approaches until it touches (F > F_TOUCH) and waits; once all pads touch,
    all squeeze together by squeeze_per_pad, stopping early when the weaker pad reaches f_stop. Like a
    force-limited gripper, the force is low-pass filtered (F_TAU) and the squeeze slows down near f_stop."""

    def __init__(self, start, direction, speed, squeeze_per_pad, f_stop=F_CAP):
        self.start = {p: np.asarray(v, float) for p, v in start.items()}
        self.dir = direction
        self.active = [p for p in start if np.any(direction[p])]
        self.speed = speed
        self.squeeze = squeeze_per_pad
        self.f_stop = f_stop
        self.s = {p: 0.0 for p in start}
        self.touch_s = {}
        self.s_sq = 0.0
        self.frozen = False
        self.t_touch = None
        self.f = {p: 0.0 for p in start}

    def __call__(self, dt, forces, t):
        a = dt / (F_TAU + dt)
        self.f = {p: self.f[p] + a * (forces[p] - self.f[p]) for p in self.f}
        forces = self.f
        if not self.frozen:
            if len(self.touch_s) < len(self.active):
                for p in self.active:
                    if p not in self.touch_s:
                        if forces[p] > F_TOUCH:
                            self.touch_s[p] = self.s[p]
                        else:
                            self.s[p] += self.speed * dt
                if len(self.touch_s) == len(self.active):
                    self.t_touch = t
            elif min(forces[p] for p in self.active) >= self.f_stop or self.s_sq >= self.squeeze:
                self.frozen = True
            else:
                slow = float(np.clip(1 - min(forces[p] for p in self.active) / self.f_stop, 0.02, 1.0))
                self.s_sq = min(self.s_sq + self.speed * slow * dt, self.squeeze)
                for p in self.active:
                    self.s[p] = self.touch_s[p] + self.s_sq
        return {p: self.start[p] + self.dir[p] * self.s[p] for p in self.start}


# ------------------------------------------------------------------ R1

def r1(k: Kappa, axis="y", strain_stop=0.5, internal=None, desc=None):
    Lx, Ly, Lz = k.size
    if axis == "y":
        W = Ly
        pads = [("pA", 1, (0, -(Ly / 2 + GAP + PAD_T), Lz / 2)), ("pB", 1, (0, Ly / 2 + GAP + PAD_T, Lz / 2))]
        direction = {"pA": np.array([0, 1.0, 0]), "pB": np.array([0, -1.0, 0])}
        speed = 0.005
    else:
        W = Lz
        pads = [("pA", 2, (0, 0, Lz + GAP + PAD_T)), ("pB", 2, (0, 0, 5.0))]
        direction = {"pA": np.array([0, 0, -1.0]), "pB": np.zeros(3)}
        speed = 0.010
    sim = make(k, pads, internal=internal, desc=desc)
    run = Runner(sim)
    start = {p[0]: np.array(p[2], float) for p in pads}
    ctrl = Closer(start, direction, speed, strain_stop * W / (2 if axis == "y" else 1))
    forces = {"pA": 0.0, "pB": 0.0}
    rows = []
    hold_after = None
    sep0 = None
    while True:
        tgt = start if run.t < SETTLE else ctrl(sim.m.opt.timestep, forces, run.t)
        if ctrl.frozen and hold_after is None:
            hold_after = run.t
        run.step(tgt)
        a_ = 1 if axis == "y" else 2
        forces = {p: sim.pad_normal_force(p, a_) for p in forces}
        f_mean = (forces["pA"] + forces["pB"]) / 2 if axis == "y" else forces["pA"]
        if axis == "y":
            sep = (sim.pad_pos("pB")[1] - sim.pad_pos("pA")[1]) - 2 * PAD_T
        else:
            sep = sim.pad_pos("pA")[2] - PAD_T
        if ctrl.t_touch is not None:
            if sep0 is None:
                sep0 = sep
            rows.append((run.t, (sep0 - sep) / W, f_mean))
        if run.fail or (hold_after is not None and run.t > hold_after + 0.2) or run.t > 40:
            break
    arr = np.array(rows)
    out = {"test": f"R1{axis}", "fail": run.fail, "fail_t": run.fail_t, "rtf": run.rtf, "sim_t": run.t,
           "capped": bool(arr[:, 2].max() >= F_CAP), "strain_max": float(arr[:, 1].max()),
           "max_joint_strain": sim.max_strain()}
    out["settled_size_mm"] = float(sep0 * 1e3) if sep0 is not None else None
    out.update(r1_metrics(arr, W))
    out["curve"] = arr[::200].round(5).tolist()
    return out


# ------------------------------------------------------------------ R2 / R3 helpers

# ------------------------------------------------------------------ R2

def r2(k: Kappa, grip="edge", internal=None, desc=None, lift_h=None):
    Lx, Ly, Lz = k.size
    if lift_h is None:
        lift_h = max(0.1, Lx + 0.05)
    if grip == "edge":
        x0 = -Lx / 2 + PAD / 2
        pads = [("pA", 2, (x0, 0, Lz + GAP + PAD_T)), ("pB", 2, (x0, 0, -GAP - PAD_T))]
        direction = {"pA": np.array([0, 0, -1.0]), "pB": np.array([0, 0, 1.0])}
        travel = 0.45 * Lz
        table_xmin, normal = -Lx / 2 + 0.05, 2
    else:
        pads = [("pA", 1, (0, -(Ly / 2 + GAP + PAD_T), Lz / 2)), ("pB", 1, (0, Ly / 2 + GAP + PAD_T, Lz / 2))]
        direction = {"pA": np.array([0, 1.0, 0]), "pB": np.array([0, -1.0, 0])}
        travel = 0.45 * Ly
        table_xmin, normal = -0.6, 1
    lift = lambda t: np.array([0, 0, min(0.05 * t, lift_h)])
    phases = [("lift", lift_h / 0.05, lift), ("hold", 2.0, lambda t: np.array([0, 0, lift_h]))]
    sim, run, rows, ev, closed = _grip_run_tracked(k, pads, direction, travel, 0.025, phases, table_xmin,
                                                   internal, desc, normal, track_from="hold", f_stop=F_GRIP)
    out = {"test": f"R2{grip}", "fail": run.fail, "fail_t": run.fail_t, "rtf": run.rtf, "sim_t": run.t}
    if run.fail or "hold" not in ev:
        return out
    t_end = ev["hold"] + 2.0
    last = [r for r in rows if r["t"] >= t_end - 0.5]
    hold = [r for r in rows if r["t"] >= ev["hold"]]
    W = sim.mass * G
    tt = [r["t"] for r in rows]
    drop_p = period_mean(tt, [r["grip"][2] - r["tip_p"][2] for r in rows], ev["hold"], t_end)
    drop_n = period_mean(tt, [r["grip"][2] - r["tip_n"][2] for r in rows], ev["hold"], t_end)
    out.update({
        "grip_force": float(np.mean([r["F"] for r in last])),
        "capped": bool(max(r["F"] for r in rows) >= F_CAP),
        "droop": float(drop_p / Lx) if grip == "edge" else None,
        "droop_ends": [float(drop_n / Lx), float(drop_p / Lx)] if grip == "center" else None,
        "pad_fz_ratio": period_mean(tt, [r["pad_fz"] for r in rows], ev["hold"], t_end) / W,
        "table_touch": bool(max(r["table"] for r in hold) > 0.01 * W),
        "slip_mm": float(max(r["slip"] for r in hold) * 1e3),
        "slipped": bool(max(r["slip"] for r in hold) > 0.005),
        "max_relative_angle_deg": float(max(r["relang"] for r in hold)),
        "dropped": bool(last[-1]["npc"] == 0),
    })
    return out


def _grip_run_tracked(k, pads, direction, travel, speed, phases, table_xmin, internal, desc, normal,
                      track_from, f_stop=F_GRIP):
    """_grip_run plus material-point tracking of the gripped point relative to the pads."""
    sim = make(k, pads, table_xmin=table_xmin, internal=internal, desc=desc)
    run = Runner(sim)
    start = {p[0]: np.array(p[2], float) for p in pads}
    ctrl = Closer(start, direction, speed, travel, f_stop)
    forces = {p[0]: 0.0 for p in pads}
    rows, closed = [], None
    t_phase, phase_i, base = None, 0, None
    ev = {}
    track = None
    while run.fail is None and run.t < 40:
        gripped = False
        if run.t < SETTLE:
            tgt = start
        elif closed is None or run.t < closed + 0.2:
            tgt = ctrl(sim.m.opt.timestep, forces, run.t)
            if ctrl.frozen and closed is None:
                closed = run.t
                base = {p: tgt[p].copy() for p in tgt}
        else:
            if t_phase is None:
                t_phase = run.t
                ev[phases[0][0]] = run.t
            while phase_i < len(phases) and run.t - t_phase >= phases[phase_i][1] - 1e-12:
                t_phase += phases[phase_i][1]
                phase_i += 1
                if phase_i < len(phases):
                    ev[phases[phase_i][0]] = t_phase
            if phase_i >= len(phases):
                break
            o = phases[phase_i][2](run.t - t_phase)
            tgt = {p: base[p] + o for p in base}
            gripped = True
        rec = run.step(tgt, gripped)
        forces = {p: sim.pad_normal_force(p, normal) for p in sim.pad_names}
        f_pads = list(forces.values())
        if rec:
            pf, tf, npc = sim.contact_forces()
            grip = np.mean([sim.pad_pos(p) for p in pf], axis=0)
            if track is None and track_from in ev:
                b = nearest_body(sim, grip)
                track = (b, sim.point_local(b, grip))
            slip = 0.0
            if track is not None:
                b, loc = track
                rel = sim.point_world(b, loc) - grip
                if "rel0" not in ev:
                    ev["rel0"] = rel
                slip = float(np.linalg.norm(rel - ev["rel0"]))
            on_obj = -sum(pf.values())
            rows.append(dict(t=run.t, F=float(np.mean(f_pads)), grip=grip, com=sim.com(), pad_fz=float(on_obj[2]),
                             table=float(np.linalg.norm(tf)), npc=npc, slip=slip,
                             tip_p=end_point(sim, 1), tip_n=end_point(sim, -1),
                             relang=sim.max_relative_angle_deg() if gripped else 0.0))
    ev.pop("rel0", None)
    return sim, run, rows, ev, closed


# ------------------------------------------------------------------ R3

def move_profile(t, a=2.0, ta=0.3, tc=0.2):
    """x offset: accelerate a for ta, cruise, decelerate a for ta (300 mm in 0.8 s)."""
    v = a * ta
    if t < ta:
        return 0.5 * a * t * t
    if t < ta + tc:
        return 0.5 * a * ta * ta + v * (t - ta)
    if t < 2 * ta + tc:
        s = t - ta - tc
        return 0.5 * a * ta * ta + v * tc + v * s - 0.5 * a * s * s
    return a * ta * ta + v * tc


def r3(k: Kappa, internal=None, desc=None):
    Lx, Ly, Lz = k.size
    pads = [("pA", 1, (0, -(Ly / 2 + GAP + PAD_T), Lz / 2)), ("pB", 1, (0, Ly / 2 + GAP + PAD_T, Lz / 2))]
    direction = {"pA": np.array([0, 1.0, 0]), "pB": np.array([0, -1.0, 0])}
    travel = 0.45 * Ly
    up = 0.05
    T_move = 0.8
    phases = [("lift", 1.0, lambda t: np.array([0, 0, min(0.05 * t, up)])),
              ("pre", 0.5, lambda t: np.array([0, 0, up])),
              ("move", T_move, lambda t: np.array([move_profile(t), 0, up])),
              ("hold", 1.0, lambda t: np.array([move_profile(T_move), 0, up]))]
    sim, run, rows, ev, closed = _grip_run_tracked(k, pads, direction, travel, 0.025, phases, -0.6,
                                                   internal, desc, 1, track_from="move")
    out = {"test": "R3", "fail": run.fail, "fail_t": run.fail_t, "rtf": run.rtf, "sim_t": run.t}
    if run.fail or "hold" not in ev:
        out["dropped"] = None
        return out
    t_stop = ev["hold"]
    t_end = t_stop + 1.0
    mv = [r for r in rows if r["t"] >= ev["move"]]
    after = [r for r in rows if r["t"] >= t_stop]
    rel = np.array([r["com"] - r["grip"] for r in after])
    ref = np.mean([r["com"] - r["grip"] for r in rows if r["t"] >= t_end - 0.2], axis=0)
    dev = np.linalg.norm(rel - ref, axis=1)
    amp = float(dev.max())
    ts = np.array([r["t"] for r in after]) - t_stop
    above = np.where(dev > 0.1 * amp)[0]
    decay = float(ts[above[-1]]) if len(above) else 0.0
    W = sim.mass * G
    F = np.array([r["F"] for r in mv])
    out.update({
        "grip_force": float(np.mean([r["F"] for r in rows if ev["pre"] <= r["t"] < ev["move"]])),
        "capped": bool(max(r["F"] for r in rows) >= F_CAP),
        "slip_mm": float(max(r["slip"] for r in mv) * 1e3),
        "com_swing_mm": amp * 1e3,
        "com_decay_s": decay,
        "decay_unfinished": bool(decay >= 0.95),
        "force_p2p": float(F.max() - F.min()),
        "dropped": bool(after[-1]["npc"] == 0),
        "pad_fz_ratio": period_mean([r["t"] for r in rows], [r["pad_fz"] for r in rows], t_stop, t_end) / W,
        "table_touch": bool(max(r["table"] for r in mv) > 0.01 * W),
    })
    return out


# ------------------------------------------------------------------ T1 rest, energy

def rest(k: Kappa, T=2.0, internal=None, desc=None):
    sim = make(k, [], internal=internal, desc=desc)
    run = Runner(sim)
    rows = []
    while run.fail is None and run.t < T - 1e-9:
        if run.step({}):
            _, tf, _ = sim.contact_forces()
            rows.append((run.t, tf[2], *sim.com()))
    out = {"test": "T1", "fail": run.fail, "fail_t": run.fail_t, "rtf": run.rtf, "sim_t": run.t}
    if run.fail:
        return out
    arr = np.array(rows)
    W = sim.mass * G
    v = sim.seg_speed()
    c1 = arr[arr[:, 0] >= T - 1.0][0, 2:]
    out.update({"support_ratio": float(arr[arr[:, 0] >= T - 0.5, 1].mean() / W),
                "speed_median_mm_s": float(np.median(v) * 1e3), "speed_max_mm_s": float(v.max() * 1e3),
                "drift_mm": float(np.linalg.norm(arr[-1, 2:] - c1) * 1e3),
                "max_joint_strain": sim.max_strain()})
    return out


def energy(k: Kappa, T=1.0, seed=0, internal=None, desc=None):
    """Zero gravity, no table: random segment velocities, total KE + elastic must not increase."""
    sim = make(k, [], table=False, gravity=False, internal=internal, desc=desc)
    rng = np.random.default_rng(seed)
    m = sim.m
    for b in sim.seg_bodies:
        j = m.body_jntadr[b]
        da = m.jnt_dofadr[j]
        sim.d.qvel[da:da + 3] = rng.normal(0, 0.05, 3)
        sim.d.qvel[da + 3:da + 6] = rng.normal(0, 0.5, 3)
    mujoco.mj_forward(m, sim.d)
    run = Runner(sim)
    E = []
    every = max(1, int(round(0.001 / m.opt.timestep)))
    E.append(sim.kinetic_energy() + sim.elastic_energy())
    while run.fail is None and run.t < T - 1e-9:
        run.step({})
        if run.n % every == 0:
            E.append(sim.kinetic_energy() + sim.elastic_energy())
    E = np.array(E)
    rise = float(np.max(E - np.minimum.accumulate(E)) / E[0])
    return {"test": "energy", "fail": run.fail, "E0": float(E[0]), "E_end_ratio": float(E[-1] / E[0]),
            "max_rise_ratio": rise, "pass": rise <= 0.01 and run.fail is None, "rtf": run.rtf}

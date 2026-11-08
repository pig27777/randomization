"""T0 Newton VBD check (newton_latest env): can the closed-loop joint grid be built and solved, and is R1
stable on the hard and compressible anchors?

Same ContentDesc as MuJoCo. Each segment is a free body; each adjacent pair gets a D6 joint (3 linear + 3
angular axes, drive stiffness/damping per axis, compression and bending limits) or a FIXED joint (hard
anchor) outside any articulation, so the grid closes loops. Pads are kinematic boxes (no edge rounding).

    python newton_check.py [--dt 0.001] [--iterations 20]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from metrics import r1_metrics
from model import ANCHORS, load_table, build

PAD, PAD_T, GAP = 0.040, 0.004, 0.002
F_CAP, F_TOUCH, F_TAU = 50.0, 0.05, 0.005
KE, KD = 3e5, 0.0              # contact stiffness per shape pair, same order as the MuJoCo CONTACT_K
RES = Path(__file__).resolve().parent / "results"


def _e(a, s=1.0):
    v = [0.0, 0.0, 0.0]
    v[a] = s
    return v


def build_scene(desc, pads, limit_ke=1e6):
    k = desc.kappa
    origin = np.array([0.0, 0.0, k.size[2] / 2 + 1e-4])
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    seg_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=KE, kd=KD, mu=0.5)
    bodies, shapes = [], []
    for s in desc.segments:
        bid = b.add_body(xform=wp.transform(wp.vec3(*(origin + s.pos)), wp.quat_identity()),
                         com=wp.vec3(*s.com), inertia=wp.mat33(*np.diag(s.inertia).flatten()), mass=s.mass)
        shapes.append(b.add_shape_box(bid, hx=s.half[0], hy=s.half[1], hz=s.half[2], cfg=seg_cfg))
        bodies.append(bid)
    Dof = newton.ModelBuilder.JointDofConfig
    for j in desc.joints:
        d = j.rest
        px = wp.transform(wp.vec3(*_e(j.axis, d / 2)), wp.quat_identity())
        cx = wp.transform(wp.vec3(*_e(j.axis, -d / 2)), wp.quat_identity())
        if j.fixed:
            b.add_joint_fixed(bodies[j.a], bodies[j.b], parent_xform=px, child_xform=cx, collision_filter_parent=True)
            continue
        o = [a for a in range(3) if a != j.axis]
        zeta, m = j.damping_ratio, j.m_eff
        cdamp = lambda kk, mm: 2 * zeta * math.sqrt(kk * mm)
        lin = [Dof(axis=_e(j.axis), target_ke=j.k_comp, target_kd=cdamp(j.k_comp, m),
                   limit_lower=-j.lock_comp * d, limit_upper=d, limit_ke=limit_ke, limit_kd=0.0)]
        lin += [Dof(axis=_e(a), target_ke=j.k_shear[i], target_kd=cdamp(j.k_shear[i], m)) for i, a in enumerate(o)]
        ang = [Dof(axis=_e(a), target_ke=j.k_bend[i], target_kd=cdamp(j.k_bend[i], j.i_eff[i]),
                   limit_lower=-j.lock_bend, limit_upper=j.lock_bend, limit_ke=1e4, limit_kd=0.0)
               for i, a in enumerate(o)]
        ang += [Dof(axis=_e(j.axis), target_ke=j.k_twist, target_kd=cdamp(j.k_twist, j.i_eff[2]))]
        b.add_joint_d6(bodies[j.a], bodies[j.b], linear_axes=lin, angular_axes=ang, parent_xform=px,
                       child_xform=cx, collision_filter_parent=True)
    joined = {(min(j.a, j.b), max(j.a, j.b)) for j in desc.joints}
    for a, c in desc.exclude:
        if (a, c) not in joined:
            b.add_shape_collision_filter_pair(shapes[a], shapes[c])
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.025), wp.quat_identity()), hx=0.6, hy=0.5, hz=0.025,
                    cfg=newton.ModelBuilder.ShapeConfig(density=0.0, ke=KE, kd=KD, mu=0.5))
    pad_ids = {}
    for name, normal, pos in pads:
        pid = b.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=1.0,
                         inertia=wp.mat33(*np.eye(3).flatten() * 1e-3), is_kinematic=True)
        h = [PAD / 2] * 3
        h[normal] = PAD_T
        b.add_shape_box(pid, hx=h[0], hy=h[1], hz=h[2],
                        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, ke=KE, kd=KD, mu=0.8))
        pad_ids[name] = pid
    b.color()
    return b.finalize(), bodies, pad_ids


def r1y(name, dt, iterations, strain_stop=0.5, t_max=30.0, limit_ke=1e6):
    k = ANCHORS[name]
    desc = build(k, table=load_table(), geometry=name)
    Lx, Ly, Lz = k.size
    pads = [("pA", 1, (0.0, -(Ly / 2 + GAP + PAD_T), Lz / 2)), ("pB", 1, (0.0, Ly / 2 + GAP + PAD_T, Lz / 2))]
    t_build = time.time()
    model, bodies, pad_ids = build_scene(desc, pads, limit_ke)
    solver = newton.solvers.SolverVBD(model, iterations=iterations, rigid_compliant_alm=True,
                                      rigid_body_contact_buffer_size=256)
    pipe = newton.CollisionPipeline(model, rigid_contact_max=20000)
    contacts = pipe.contacts()
    s0, s1, ctrl = model.state(), model.state(), model.control()
    t_build = time.time() - t_build
    start = {p: np.array(pos) for p, _, pos in pads}
    sign = {"pA": 1.0, "pB": -1.0}
    s_pad = {"pA": 0.0, "pB": 0.0}
    touch, s_sq, frozen, f_filt = {}, 0.0, False, {"pA": 0.0, "pB": 0.0}
    rows, fail, t, n, sep0, t_freeze = [], None, 0.0, 0, None, None
    pairs = np.array([(j.a, j.b, j.rest, j.lock_comp) for j in desc.joints], float) if desc.joints else None
    seg_idx = np.array(bodies)
    t_wall = time.time()
    while t < t_max:
        # pad controller (same logic as protocols.Closer)
        if t >= 0.3 and not frozen:
            if len(touch) < 2:
                for p in s_pad:
                    if p not in touch:
                        if f_filt[p] > F_TOUCH:
                            touch[p] = s_pad[p]
                        else:
                            s_pad[p] += 0.005 * dt
            elif min(f_filt.values()) >= F_CAP or s_sq >= strain_stop * Ly / 2:
                frozen, t_freeze = True, t
            else:
                s_sq += 0.005 * float(np.clip(1 - min(f_filt.values()) / F_CAP, 0.02, 1)) * dt
                for p in s_pad:
                    s_pad[p] = touch[p] + s_sq
        bq = s0.body_q.numpy()
        bqd = s0.body_qd.numpy()
        for p, pid in pad_ids.items():
            pos = start[p] + np.array([0, sign[p] * s_pad[p], 0])
            vel = (pos - bq[pid, :3]) / dt
            bq[pid, :3] = pos
            bqd[pid] = [*vel, 0, 0, 0]
        s0.body_q.assign(bq)
        s0.body_qd.assign(bqd)
        q_prev = wp.clone(solver.body_q_prev) if solver.body_q_prev is not None else wp.clone(s0.body_q)
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, dt)
        b0, b1, _, _, f1, cnt = solver.collect_rigid_contact_forces(s1.body_q, q_prev, contacts, dt)
        nc = int(cnt.numpy()[0])
        F = {"pA": 0.0, "pB": 0.0}
        if nc:
            b0n, b1n, f1n = b0.numpy()[:nc], b1.numpy()[:nc], f1.numpy()[:nc]
            for p, pid in pad_ids.items():
                F[p] = abs(float(f1n[b1n == pid, 1].sum() - f1n[b0n == pid, 1].sum()))
        a = dt / (F_TAU + dt)
        f_filt = {p: f_filt[p] + a * (F[p] - f_filt[p]) for p in F}
        s0, s1 = s1, s0
        t += dt
        n += 1
        q = s0.body_q.numpy()
        if not np.all(np.isfinite(q)):
            fail = "non-finite state"
            break
        if pairs is not None and n % 10 == 0:
            pa, pb = q[seg_idx[pairs[:, 0].astype(int)], :3], q[seg_idx[pairs[:, 1].astype(int)], :3]
            strain = (pairs[:, 2] - np.linalg.norm(pb - pa, axis=1)) / pairs[:, 2]
            if strain.max() > 0.5:
                fail = f"adjacent overlap {strain.max():.2f}"
                break
            if np.max(strain / np.maximum(pairs[:, 3], 1e-9)) > 1.2:
                fail = f"compression {np.max(strain / pairs[:, 3]):.2f} x limit"
                break
        if n % 10 == 0:
            v = np.linalg.norm(s0.body_qd.numpy()[seg_idx, :3], axis=1)
            if v.max() > 1.0:
                fail = f"flung: {v.max():.2f} m/s"
                break
        if len(touch) == 2:
            sep = (start["pB"][1] - s_pad["pB"]) - (start["pA"][1] + s_pad["pA"]) - 2 * PAD_T
            sep0 = sep if sep0 is None else sep0
            rows.append((t, (sep0 - sep) / Ly, (F["pA"] + F["pB"]) / 2))
        if t_freeze is not None and t > t_freeze + 0.2:
            break
    wall = time.time() - t_wall
    arr = np.array(rows) if rows else np.zeros((0, 3))
    out = {"anchor": name, "segments": k.segments, "rigid": k.rigid, "dt": dt, "iterations": iterations,
           "limit_ke": limit_ke,
           "n_bodies": len(bodies), "n_joints": len(desc.joints),
           "joint_types": "FIXED" if k.rigid else "D6", "build_s": t_build, "fail": fail, "sim_t": t, "rtf": t / wall,
           "touched": len(touch) == 2, "frozen_by": None if t_freeze is None else
           ("force cap" if min(f_filt.values()) >= F_CAP * 0.98 else "strain")}
    if len(arr) > 10:
        out.update(r1_metrics(arr, Ly))
        out["strain_max"] = float(arr[:, 1].max())
        out["f_max"] = float(arr[:, 2].max())
        out["curve"] = arr[:: max(1, len(arr) // 200)].round(5).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--anchors", nargs="+", default=["hard", "compressible"])
    ap.add_argument("--t-max", type=float, default=30.0)
    ap.add_argument("--limit-ke", type=float, default=1e6, help="compression limit stiffness of the D6 joints")
    ap.add_argument("--out", default="t0_newton")
    args = ap.parse_args()
    wp.init()
    rows = []
    for name in args.anchors:
        t0 = time.time()
        try:
            r = r1y(name, args.dt, args.iterations, t_max=args.t_max, limit_ke=args.limit_ke)
        except Exception as e:      # a build/solve error is itself the T0 answer; keep it with context
            import traceback
            r = {"anchor": name, "fail": f"exception: {type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        r["wall_s"] = time.time() - t0
        print({kk: v for kk, v in r.items() if kk not in ("curve", "traceback")}, flush=True)
        rows.append(r)
    RES.mkdir(exist_ok=True)
    (RES / f"{args.out}.json").write_text(json.dumps({"rows": rows, "newton": newton.__version__}, indent=1))
    print(f"-> results/{args.out}.json")


if __name__ == "__main__":
    main()

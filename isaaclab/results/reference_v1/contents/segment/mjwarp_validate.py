"""S3: MJWarp vs CPU MuJoCo on the same scenes, plus batch stability and throughput.

Open-loop replay: the R1-y / R2 protocol runs on CPU and logs the prescribed pad state of every step; the
same pad trajectory is then replayed on CPU and on MJWarp (a Warp kernel writes the pad qpos/qvel each step,
so the whole step is graph-captured). Both replays measure the pad force as the constraint force on the pad
slide dofs and the droop from segment poses, so they compare like with like.

    python mjwarp_validate.py [--nworld 1024] [--dt 0.0005] [--mode truss] [--groups center,bend,shear]
"""
from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

import mj_adapter as A
import protocols as P
from metrics import r1_metrics
from model import ANCHORS, G, build, load_table

RES = Path(__file__).resolve().parent / "results" / "round2"


@wp.kernel
def _set_pads(traj_q: wp.array2d(dtype=float), traj_v: wp.array2d(dtype=float), step: wp.array(dtype=int),
              qadr: wp.array(dtype=int), vadr: wp.array(dtype=int),
              qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float)):
    w, i = wp.tid()
    s = step[0]
    qpos[w, qadr[i]] = traj_q[s, i]
    qvel[w, vadr[i]] = traj_v[s, i]


@wp.kernel
def _inc(step: wp.array(dtype=int), n: int):
    step[0] = wp.min(step[0] + 1, n - 1)


def record_cpu(k, desc, test):
    """Run the protocol on CPU and log pad (qpos, qvel) per step; return log, the Sim, protocol output."""
    log = []
    orig = A.Sim.set_pad

    def set_pad(self, p, pos, vel):
        orig(self, p, pos, vel)
        self._log.setdefault(p, []).append((np.asarray(pos) - self.pad_p0[p], np.asarray(vel)))

    holder = {}
    orig_init = A.Sim.__init__

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        self._log = {}
        holder["sim"] = self
    A.Sim.set_pad, A.Sim.__init__ = set_pad, init
    try:
        out = P.r1(k, "y", desc=desc) if test == "R1y" else P.r2(k, desc=desc)
    finally:
        A.Sim.set_pad, A.Sim.__init__ = orig, orig_init
    sim = holder["sim"]
    pads = sorted(sim._log)
    T = len(sim._log[pads[0]])
    q = np.concatenate([np.array([sim._log[p][t][0] for t in range(T)]) for p in pads], axis=1)
    v = np.concatenate([np.array([sim._log[p][t][1] for t in range(T)]) for p in pads], axis=1)
    p0 = np.concatenate([sim.pad_p0[p] for p in pads])
    return sim.built.xml, pads, q, v, out, p0


def _measure(m, xpos, xmat, qfrc, pads, seg_bodies, desc, axis):
    """Mean |pad force along axis| and droop of the +x end (edge grip)."""
    f = []
    for p in pads:
        dof = m.jnt_dofadr[m.joint(f"{p}_{'xyz'[axis]}").id]
        f.append(abs(qfrc[dof]))
    k = desc.kappa
    if desc.rigid:
        b = seg_bodies[0]
        tip = xpos[b] + xmat[b].reshape(3, 3) @ np.array([k.size[0] / 2, 0, 0])
    else:
        nx = k.segments[0]
        pts = [xpos[seg_bodies[n]] + xmat[seg_bodies[n]].reshape(3, 3) @ np.array([s.half[0], 0, 0])
               for n, s in enumerate(desc.segments) if s.idx[0] == nx - 1]
        tip = np.mean(pts, axis=0)
    grip = np.mean([xpos[m.body(p).id] for p in pads], axis=0)
    return float(np.mean(f)), float((grip[2] - tip[2]) / k.size[0])


def replay(xml, pads, q, v, desc, axis, gpu, every=20):
    m = mujoco.MjModel.from_xml_string(xml)
    seg_bodies = [m.body("obj").id] if desc.rigid else [m.body(f"s{i}").id for i in range(len(desc.segments))]
    qadr = np.array([m.jnt_qposadr[m.joint(f"{p}_{a}").id] for p in pads for a in "xyz"], np.int32)
    vadr = np.array([m.jnt_dofadr[m.joint(f"{p}_{a}").id] for p in pads for a in "xyz"], np.int32)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    T = len(q)
    rows = []
    finite = True
    if not gpu:
        for t in range(T):
            d.qpos[qadr], d.qvel[vadr] = q[t], v[t]
            mujoco.mj_step(m, d)
            if t % every == 0:
                rows.append((t,) + _measure(m, d.xpos, d.xmat, d.qfrc_constraint, pads, seg_bodies, desc, axis))
        finite = bool(np.all(np.isfinite(d.qpos)))
        return rows, finite
    md = mjw.put_model(m)
    dd = mjw.put_data(m, d, nworld=1, nconmax=4000, njmax=12000)
    tq, tv = wp.array(q.astype(np.float32)), wp.array(v.astype(np.float32))
    step = wp.zeros(1, dtype=int)
    wq, wv = wp.array(qadr), wp.array(vadr)

    def one():
        wp.launch(_set_pads, dim=(1, len(qadr)), inputs=[tq, tv, step, wq, wv, dd.qpos, dd.qvel])
        mjw.step(md, dd)
        wp.launch(_inc, dim=1, inputs=[step, T])
    one()
    with wp.ScopedCapture() as cap:
        for _ in range(every):
            one()
    for t0 in range(every, T, every):
        wp.capture_launch(cap.graph)
        xp, xm = dd.xpos.numpy()[0], dd.xmat.numpy()[0].reshape(-1, 9)
        qf = dd.qfrc_constraint.numpy()[0]
        if not np.all(np.isfinite(xp)):
            finite = False
            break
        rows.append((t0,) + _measure(m, xp, xm, qf, pads, seg_bodies, desc, axis))
    return rows, finite


def compare(name, test):
    k = ANCHORS[name]
    desc = build(k, omega_max=A.OMEGA_MAX, table=load_table(), geometry=name)
    xml, pads, q, v, out, p0 = record_cpu(k, desc, test)
    axis = 1 if test == "R1y" else 2
    res = {"anchor": name, "test": test, "cpu_protocol": {kk: out.get(kk) for kk in ("stiffness", "lock_strain", "droop", "fail")}}
    for label, gpu in (("cpu_replay", False), ("mjwarp_replay", True)):
        rows, finite = replay(xml, pads, q, v, desc, axis, gpu)
        arr = np.array(rows)
        r = {"finite": finite}
        if test == "R1y":
            # strain from both-pad contact, as in R1: absolute pad gap vs the gap at contact (CPU protocol)
            gap = (p0[4] + q[:, 4]) - (p0[1] + q[:, 1]) - 2 * A.PAD_T
            sep0 = out["settled_size_mm"] / 1e3
            t_idx = arr[:, 0].astype(int)
            e = (sep0 - gap[t_idx]) / k.size[1]
            sel = e >= 0
            if sel.sum() > 5:
                met = r1_metrics(np.column_stack([arr[sel, 0], e[sel], arr[sel, 1]]), k.size[1])
                r.update({"stiffness": met["stiffness"], "lock_strain": met["lock_strain"]})
        else:
            hold = arr[arr[:, 0] >= arr[-1, 0] - int(0.5 / A.DT)]
            r.update({"droop": float(np.mean(hold[:, 2])), "pad_force": float(np.mean(hold[:, 1]))})
        res[label] = r
    for q_ in ("stiffness", "droop"):
        a, b = res["cpu_replay"].get(q_), res["mjwarp_replay"].get(q_)
        if a is not None and b is not None and a:
            res[f"{q_}_rel_diff"] = b / a - 1
    return res


def batch(name, segs, nworld, seconds=1.0):
    """Stability and throughput: object gripped by static pads, nworld copies, `seconds` of sim time."""
    from dataclasses import replace
    k = replace(ANCHORS[name], segments=segs)
    desc = build(k, omega_max=A.OMEGA_MAX, table=load_table(), geometry=name)
    Ly, Lz = k.size[1], k.size[2]
    pads = [("pA", 1, (0, -(Ly / 2 + A.PAD_T - 0.002), Lz / 2)), ("pB", 1, (0, Ly / 2 + A.PAD_T - 0.002, Lz / 2))]
    m = mujoco.MjModel.from_xml_string(A.scene(desc, (0, 0, Lz / 2 + 1e-4), pads).xml)
    d = mujoco.MjData(m)
    for _ in range(int(0.1 / A.DT)):
        mujoco.mj_step(m, d)
    md = mjw.put_model(m)
    dd = mjw.put_data(m, d, nworld=nworld, nconmax=6 * max(d.ncon, 1), njmax=6 * max(d.nefc, 1) + 64)
    mjw.step(md, dd)
    with wp.ScopedCapture() as cap:
        mjw.step(md, dd)
    n = int(seconds / A.DT)
    wp.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    el = time.perf_counter() - t
    qp = dd.qpos.numpy()
    bad = int(np.sum(~np.all(np.isfinite(qp), axis=1)))
    return {"anchor": name, "segments": segs, "nworld": nworld, "dt": A.DT, "ntendon": m.ntendon, "neq": m.neq,
            "bad_worlds": bad, "env_steps_per_s": nworld * n / el, "sim_s_per_s": nworld * n * A.DT / el}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=1024)
    ap.add_argument("--dt", type=float, default=A.DT)
    ap.add_argument("--mode", default=A.JOINT_MODE)
    ap.add_argument("--groups", default=",".join(A.DAMPED_GROUPS))
    ap.add_argument("--solver", default=A.SOLVER)
    ap.add_argument("--out", default="s3_mjwarp")
    ap.add_argument("--skip-compare", action="store_true")
    args = ap.parse_args()
    if args.mode == "bushing":
        ap.error("CPU bushings require Sim.step(); MJWarp has no implementation of these forces. "
                 "Use --mode truss only for explicitly labelled legacy comparisons.")
    A.DT, A.OMEGA_MAX, A.JOINT_MODE = args.dt, 0.8 / args.dt, args.mode
    A.DAMPED_GROUPS, A.SOLVER = tuple(args.groups.split(",")), args.solver
    P.make = functools.partial(P.make, dt=args.dt)
    out = {"config": vars(args), "compare": [], "batch": []}
    if not args.skip_compare:
        for name in ("hard", "bendable", "compressible"):
            for test in ("R1y", "R2"):
                r = compare(name, test)
                print(r, flush=True)
                out["compare"].append(r)
    for name, segs in (("hard", (4, 3, 2)), ("bendable", (6, 3, 2)), ("compressible", (4, 4, 3)),
                       ("bendable", (6, 4, 3))):
        r = batch(name, segs, args.nworld)
        print(r, flush=True)
        out["batch"].append(r)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{args.out}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

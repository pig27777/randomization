"""F2 (test/bag_model/REPORT.md round 2): the flex loose bag in MJWarp 3.17, batched, against CPU MuJoCo.

The force-controlled evaluate() state machine is not ported. Instead every world runs the same open-loop
script on CPU and GPU: settle, both 40 mm pads close across the bag at 25 mm/s to a fixed gap, lift 50 mm at
50 mm/s, hold. Compared: content COM, bag COM and content-in-bag at the end; measured: environment steps/s.
Only rigid contents are supported (the segment bushings are Python forces on CPU; MJWarp has no
implementation, contents/segment/model.py notes).

Usage: python -B loose_mjwarp.py --set JSON [--dt 0.0004] [--nworld 64 256] [--gap 0.088] [--out f2_mjwarp.jsonl]
"""


import argparse
import json
import time
from dataclasses import asdict

import mujoco
import numpy as np
import warp as wp

from loose_protocols import GAP, KRAFT_SIZE, OUT, content_kappa, make
from mj_adapter import PAD_T
from mj_loose import LooseBag

SETTLE, CLOSE_V, LIFT, LIFT_V, HOLD = 0.3, 0.025, 0.05, 0.05, 0.5
NCON, NJMAX = 400, 1200


@wp.kernel
def _set_pads(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), tq: wp.array2d(dtype=float),
              tv: wp.array2d(dtype=float), iq: wp.array(dtype=int), iv: wp.array(dtype=int), k: wp.array(dtype=int)):
    w, j = wp.tid()
    qpos[w, iq[j]] = tq[k[0], j]
    qvel[w, iv[j]] = tv[k[0], j]


@wp.kernel
def _advance(k: wp.array(dtype=int)):
    k[0] = k[0] + 1


def script(sim, gap):
    """Pad positions (2, 3) as a function of time, from the wrapped package's footprint."""
    cxy, size, h = sim.footprint()
    y0 = size[1] / 2 + GAP + PAD_T
    y1 = gap / 2 + PAD_T
    t_close = (y0 - y1) / CLOSE_V
    T = SETTLE + t_close + LIFT / LIFT_V + HOLD

    def at(t):
        s = min(max(t - SETTLE, 0.0) * CLOSE_V, y0 - y1)
        z = h / 2 + min(max(t - SETTLE - t_close, 0.0) * LIFT_V, LIFT)
        return np.array([[cxy[0], cxy[1] - y0 + s, z], [cxy[0], cxy[1] + y0 - s, z]])

    return at, T


def measures(sim, com, bagpts, content_pts=None):
    return dict(content_z_mm=float(com[2] * 1e3), bag_z_mean_mm=float(bagpts[:, 2].mean() * 1e3))


def run_cpu(sim, at, T):
    dt = sim.m.opt.timestep
    n = int(round(T / dt))
    prev = at(0.0)
    for p, x in zip(("pA", "pB"), prev):
        sim.set_pad(p, x, np.zeros(3))
    t0 = time.perf_counter()
    for i in range(n):
        x = at((i + 1) * dt)
        for j, p in enumerate(("pA", "pB")):
            sim.set_pad(p, prev[j], (x[j] - prev[j]) / dt)
        prev = x
        sim.step()
    wall = time.perf_counter() - t0
    out = measures(sim, sim.com(), sim.bag_points())
    out.update(penetration_mm=sim.penetration() * 1e3, finite=bool(np.isfinite(sim.d.qpos).all()),
               steps=n, wall_s=wall, steps_per_s=n / wall)
    return out


def run_gpu(sim, at, T, nworld):
    import mujoco_warp as mjw
    m, d = sim.m, sim.d
    dt = m.opt.timestep
    n = int(round(T / dt))
    qa = np.r_[sim.pad_q["pA"], sim.pad_q["pB"]]
    va = np.r_[sim.pad_v["pA"], sim.pad_v["pB"]]
    p0 = np.r_[sim.pad_p0["pA"], sim.pad_p0["pB"]]
    prev = at(0.0)
    d.qpos[qa] = prev.ravel() - p0
    d.qvel[va] = 0
    mujoco.mj_forward(m, d)
    wm = mjw.put_model(m)
    # buffers from the CPU run: <= 179 contacts and ~400-500 constraint rows per world (4-world smoke run)
    wd = mjw.put_data(m, d, nworld=nworld, nconmax=NCON * nworld, njmax=NJMAX)
    qpos, qvel = wd.qpos.numpy(), wd.qvel.numpy()
    # pad states for the whole script, uploaded once per step from a host table would dominate the run;
    # the pads move identically in every world, so the kernel writes the same row everywhere
    traj = np.array([at((i + 1) * dt).ravel() for i in range(n)])
    vel = np.diff(np.vstack([prev.ravel(), traj]), axis=0) / dt
    start_q = np.vstack([prev.ravel(), traj[:-1]]) - p0
    tq = wp.array(start_q.astype(np.float32), dtype=float)
    tv = wp.array(vel.astype(np.float32), dtype=float)
    iq = wp.array(qa.astype(np.int32), dtype=int)
    iv = wp.array(va.astype(np.int32), dtype=int)
    step = wp.zeros(1, dtype=int)

    def one():
        wp.launch(_set_pads, dim=(nworld, 6), inputs=[wd.qpos, wd.qvel, tq, tv, iq, iv, step])
        mjw.step(wm, wd)
        wp.launch(_advance, dim=1, inputs=[step])

    one()                                               # compile
    with wp.ScopedCapture() as cap:
        one()
    graph = cap.graph
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(n - 2):
        wp.capture_launch(graph)
    wp.synchronize()
    wall = time.perf_counter() - t0
    q = wd.qpos.numpy()
    xipos = wd.xipos.numpy()
    flexv = wd.flexvert_xpos.numpy() if hasattr(wd, "flexvert_xpos") else None
    b = sim.seg_bodies[0]
    out = dict(nworld=nworld, steps=n, wall_s=wall, env_steps_per_s=nworld * (n - 2) / wall,
               finite_worlds=int(np.isfinite(q).all(axis=1).sum()),
               content_z_mm=[float(x) for x in np.percentile(xipos[:, b, 2] * 1e3, [0, 50, 100])])
    if flexv is not None:
        out["bag_z_mean_mm"] = float(flexv[0, :, 2].mean() * 1e3)
    out["nacon_max"] = int(np.max(wd.nacon.numpy())) if hasattr(wd, "nacon") else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="{}")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--gap", type=float, default=0.088, help="final pad gap (m); box 80 mm + film")
    ap.add_argument("--nworld", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--content", default="box_loose")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    bag = LooseBag(kind="flex", size=KRAFT_SIZE, **json.loads(a.set))
    k = content_kappa(a.content)
    pads = [("pA", 1, (0, -1, 0.5)), ("pB", 1, (0, 1, 0.5))]
    sim = make(k, bag, pads, dt=a.dt)
    at, T = script(sim, a.gap)
    res = dict(set=json.loads(a.set), dt=sim.m.opt.timestep, nv=sim.m.nv, sim_s=T, bag=asdict(bag))
    res["cpu"] = run_cpu(sim, at, T)
    for nw in a.nworld:                    # one line per world count, so a later failure keeps earlier results
        sim_g = make(k, bag, pads, dt=a.dt)
        line = json.dumps(dict(res, gpu=run_gpu(sim_g, at, T, nw)), default=str)
        print(line, flush=True)
        if a.out:
            with open(OUT / a.out, "a") as fh:
                fh.write(line + "\n")


if __name__ == "__main__":
    main()

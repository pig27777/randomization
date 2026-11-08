"""MJWarp throughput of the MuJoCo adapter scene (round 1, recorded only).

Scene: object resting on the table with two side pads closed onto it (R3 grip pose, pads static), so the
step includes table, pad and self contacts plus the tendon truss. Reports env-steps/s for nworld = 64, 256.

    python mjwarp_speed.py [--steps 500] [--no-tendon-damping]

MJWarp 3.17 diverges within ~20-220 steps on the tendon-truss objects unless tendon damping is removed
(CPU MuJoCo is stable with it); --no-tendon-damping measures cost on that stable variant.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from model import ANCHORS, build, load_table
from mj_adapter import OMEGA_MAX, PAD_T, scene

RES = Path(__file__).resolve().parent / "results"


def make_model(name, segs, no_tendon_damping=False):
    k = replace(ANCHORS[name], segments=segs)
    desc = build(k, omega_max=OMEGA_MAX, table=load_table(), geometry=name)
    Ly, Lz = k.size[1], k.size[2]
    pads = [("pA", 1, (0, -(Ly / 2 + PAD_T - 0.002), Lz / 2)), ("pB", 1, (0, Ly / 2 + PAD_T - 0.002, Lz / 2))]
    # This historical benchmark measures native tendon trusses, not CPU bushings.
    built = scene(desc, (0.0, 0.0, Lz / 2 + 1e-4), pads, mode="truss")
    xml = built.xml
    if no_tendon_damping:
        xml = re.sub(r'damping="[0-9.e-]+"><site', 'damping="0"><site', xml)
    return mujoco.MjModel.from_xml_string(xml)


def bench(mjm, nworld, steps):
    mjd = mujoco.MjData(mjm)
    for _ in range(200):                         # settle on CPU so contacts are populated
        mujoco.mj_step(mjm, mjd)
    ncon = max(mjd.ncon, 1)
    nefc = max(mjd.nefc, 1)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=4 * ncon, njmax=4 * nefc)
    mjw.step(m, d)
    with wp.ScopedCapture() as cap:
        mjw.step(m, d)
    wp.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    dt = time.perf_counter() - t
    q = d.qpos.numpy()
    return {"nworld": nworld, "steps": steps, "wall_s": dt, "env_steps_per_s": nworld * steps / dt,
            "sim_s_per_wall_s_per_env": steps * mjm.opt.timestep / dt, "finite": bool(np.all(np.isfinite(q))),
            "cpu_ncon": int(mjd.ncon), "cpu_nefc": int(mjd.nefc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--no-tendon-damping", action="store_true")
    args = ap.parse_args()
    rows = []
    for name, segs in (("compressible", (4, 4, 3)), ("bendable", (6, 4, 3)), ("hard", (4, 3, 2))):
        mjm = make_model(name, segs, args.no_tendon_damping)
        # single-env CPU reference
        mjd = mujoco.MjData(mjm)
        t = time.perf_counter()
        for _ in range(2000):
            mujoco.mj_step(mjm, mjd)
        cpu = 2000 / (time.perf_counter() - t)
        for nw in (64, 256):
            r = bench(mjm, nw, args.steps)
            r.update({"object": name, "segments": segs, "tendon_damping": not args.no_tendon_damping, "nbody": mjm.nbody, "ntendon": mjm.ntendon,
                      "cpu_steps_per_s": cpu, "timestep": mjm.opt.timestep})
            print(r, flush=True)
            rows.append(r)
    RES.mkdir(exist_ok=True)
    out = "mjwarp_speed_nodamp.json" if args.no_tendon_damping else "mjwarp_speed.json"
    (RES / out).write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()

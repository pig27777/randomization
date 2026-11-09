"""Watch an M2 grip-and-lift run of the current CPU model in the MuJoCo viewer (writes no results).

Usage: DISPLAY=:1 view_m2.py [bendable|compressible] [center|edge] [--dt 0.0004] [--rate 1.0]
Segments are an orange/grey checkerboard; those held by an active compaction weld turn green. Pads are
translucent blue. Tendons are hidden unless --tendons (Rendering > Tendon toggles them). The overlay shows time, the active weld count and the largest relative angle between
connected segments; pass/slip/support print at the end. Space pauses; close the window to stop.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from pathlib import Path
import sys

CONTENT = Path(__file__).resolve().parents[2] / "contents/segment"
sys.path.insert(0, str(CONTENT))

import mj_adapter as A
import validate_model as V

ap = argparse.ArgumentParser()
ap.add_argument("anchor", nargs="?", default="compressible", choices=("bendable", "compressible"))
ap.add_argument("grip", nargs="?", default="edge", choices=("center", "edge"))
ap.add_argument("--dt", type=float, default=A.BUSHING_DT)
ap.add_argument("--rate", type=float, default=1.0, help="playback speed relative to real time")
ap.add_argument("--tendons", action="store_true", help="draw the native stop / damper / shear-lock tendons")
args = ap.parse_args()
A.BUSHING_DT = args.dt
V.save = lambda name, result: print(f"{name}: pass={result['pass']}, slip_mm={result.get('slip_mm')}, "
                                    f"support={result.get('pad_support_ratio')}, fail={result['fail']}", flush=True)

state = {"viewer": None, "wall0": None, "sim_t0": None}
record = V.layer_record


def show(sim):
    row = record(sim)
    if state["viewer"] is None:
        state["viewer"] = viewer = mujoco.viewer.launch_passive(sim.m, sim.d)
        viewer.cam.lookat[:] = sim.com()
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = .9, 120, -20
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = args.tendons
        state["geoms"] = [np.flatnonzero(sim.m.geom_bodyid == b) for b in sim.seg_bodies]
        state["rgba"] = sim.m.geom_rgba.copy()
        state["wall0"], state["sim_t0"] = time.perf_counter(), sim.d.time
    viewer = state["viewer"]
    if not viewer.is_running():
        raise SystemExit
    welded = set()
    for i in np.flatnonzero(sim.d.eq_active[sim.welds]):
        welded |= {sim.bushings.node_a[i], sim.bushings.node_b[i]}
    with viewer.lock():
        for s, g in enumerate(state["geoms"]):
            sim.m.geom_rgba[g] = (.1, .85, .2, 1) if s in welded else state["rgba"][g]
        viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "t\nwelds\nmax relative angle",
                          f"{sim.d.time:.2f} s\n{int(sim.d.eq_active[sim.welds].sum())}\n"
                          f"{sim.max_relative_angle_deg():.1f} deg"))
    viewer.sync()
    lag = (sim.d.time - state["sim_t0"]) / args.rate - (time.perf_counter() - state["wall0"])
    if lag > 0:
        time.sleep(lag)
    return row


V.layer_record = show
V.integrity(args.anchor, "bushing", "original", args.grip)
viewer = state["viewer"]
while viewer is not None and viewer.is_running():
    viewer.sync()
    time.sleep(.05)

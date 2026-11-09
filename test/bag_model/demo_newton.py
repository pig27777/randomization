"""Drop a profile onto a table step in Newton and record its surface for rendering.

Usage: <newton python> -B test/bag_model/demo_newton.py PROFILE [--seconds 2] [--fps 30] [--out outputs]
The bag starts 10 cm up, tilted 12 degrees, half over a 6 cm step, and falls under gravity.
Writes outputs/<profile>_drop.npz with frames (T,N,3), faces, the step box and timing.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parents[2] / "bag_sim"
sys.path.insert(0, str(HERE))
import newton  # noqa: E402
from bag_model import parameters  # noqa: E402
from newton_bag import content_shape_config, load_profile_newton  # noqa: E402

STEP = dict(center=(-0.2, 0.0, 0.03), half=(0.2, 0.3, 0.03))


def initial_pose(profile, tilt_deg=12.0, height=0.10, over=0.04):
    spec = json.loads((HERE / "profiles" / f"{profile}.json").read_text())
    with np.load(HERE / spec["geometry"]) as g:
        q = np.array(g["initial_vertices"] if "initial_vertices" in g else g["vertices"], float)
    q -= q.mean(axis=0)
    # Long side along y; tilt about y so the free half droops off the step edge at x = 0.
    a = np.radians(tilt_deg)
    rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    q = q @ rotation.T
    q[:, 0] += over
    q[:, 2] += 2 * STEP["half"][2] + height - q[:, 2].min()
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default=str(HERE / "outputs"))
    args = parser.parse_args()
    wp.config.quiet = True

    spec_params = json.loads((HERE / "profiles" / f"{args.profile}.json").read_text())["params"]

    def add_step(builder, contact):
        cfg = content_shape_config(contact, parameters(spec_params), mu=0.6)
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(*STEP["center"]), wp.quat_identity()),
                              hx=STEP["half"][0], hy=STEP["half"][1], hz=STEP["half"][2], cfg=cfg)

    bag = load_profile_newton(args.profile, initial_vertices=initial_pose(args.profile), extra=add_step)
    frames_per_record = max(1, int(round(1 / (args.fps * bag.params["frame_dt_s"]))))
    frames = [bag.positions.astype(np.float32)]
    start = time.perf_counter()
    for frame in range(int(round(args.seconds / bag.params["frame_dt_s"]))):
        bag.step()
        if (frame + 1) % frames_per_record == 0:
            frames.append(bag.positions.astype(np.float32))
    wall = time.perf_counter() - start
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = bag.state_report()
    np.savez_compressed(out / f"{args.profile}_drop.npz", frames=np.array(frames), faces=bag.faces,
                        step_center=np.array(STEP["center"]), step_half=np.array(STEP["half"]),
                        fps=args.fps, seconds=args.seconds, wall_s=wall, report=json.dumps(report))
    print(json.dumps(dict(profile=args.profile, frames=len(frames), wall_s=round(wall, 1),
                          realtime_factor=round(args.seconds / wall, 4), max_speed_m_s=report["max_speed_m_s"],
                          max_edge_stretch=report["max_edge_stretch"], finite=report["finite"])), flush=True)


if __name__ == "__main__":
    main()

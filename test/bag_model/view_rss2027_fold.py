"""Replay recorded bag or coupon deformation in the native MuJoCo viewer."""
import argparse
import csv
import json
from pathlib import Path
from queue import SimpleQueue
import time

import mujoco
import mujoco.viewer
import numpy as np


def load_recording(run):
    model = mujoco.MjModel.from_binary_path(str(run / "model.mjb"))
    data = mujoco.MjData(model)
    with (run / "curve.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with np.load(run / "coupon_nodes.npz") as source:
        reference = source["reference_m"]
        points = source["current_m"]

    # The original mocap trials have different pose storage. This viewer
    # replays the slide-driven fixture, with one XYZ joint set per pad.
    pad_ids = [model.body(name).id for name in ("left", "right")]
    film_joints = model.body_jntadr[model.flex_vertbodyid, None] + np.arange(3)
    pad_joints = model.body_jntadr[pad_ids, None] + np.arange(3)
    if len(rows) != len(points) or not len(rows) or np.any(pad_joints[:, 0] < 0):
        raise ValueError("Expected one recorded mesh per row and a slide-driven pad fixture")
    frame_times = np.array([float(row["t_s"]) for row in rows])
    if np.any(np.diff(frame_times) <= 0):
        raise ValueError("Recorded time must be strictly increasing")
    film_qpos = model.jnt_qposadr[film_joints]
    pad_qpos = model.jnt_qposadr[pad_joints]
    pad_origins = model.body_pos[pad_ids]
    half = pad_origins[1, 0] - np.array([float(row["gather_per_pad_m"]) for row in rows])
    pad_z = np.array([float(row["pad_z_m"]) for row in rows])
    pad_positions = np.stack((-half, np.zeros_like(half), pad_z, half, np.zeros_like(half), pad_z), axis=1)
    frames = np.repeat(data.qpos[None, :], len(rows), axis=0)
    frames[:, film_qpos] = points - reference
    frames[:, pad_qpos] = pad_positions.reshape(-1, 2, 3) - pad_origins
    frame_info = [(row["phase"], f"{float(row['signed_pull_N']):.3f} N pull") for row in rows]
    return model, data, frames, frame_times, frame_info, points


def load_yam_recording(run):
    model = mujoco.MjModel.from_binary_path(str(run / "model.mjb"))
    data = mujoco.MjData(model)
    rows = json.loads((run / "trace.json").read_text())
    with np.load(run / "trajectory.npz") as source:
        frames, frame_times = source["qpos"], source["time"]
    if frames.shape != (len(rows), model.nq) or not np.allclose(frame_times, [row["t"] for row in rows]):
        raise ValueError("The archived YAM model, poses and trace do not match")
    frame_info = [(row["phase"], f"{row['grip_N']:.3f} N grip") for row in rows]
    return model, data, frames, frame_times, frame_info, rows


def show_pose(model, data, pose):
    data.qpos[:] = pose
    mujoco.mj_fwdPosition(model, data)


def replay(model, data, frames, frame_times, frame_info, index, title, close_view, overview, fold_edge=None):
    keys = SimpleQueue()
    playback_start = None
    show_pose(model, data, frames[index])
    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        def camera(view):
            center, distance, azimuth, elevation = view
            viewer.cam.lookat[:] = center
            viewer.cam.distance = distance
            viewer.cam.azimuth = azimuth
            viewer.cam.elevation = elevation
            if fold_edge is not None:
                viewer.opt.geomgroup[:3] = view is overview

        active_view = close_view
        with viewer.lock():
            camera(active_view)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = True

        print(f"MuJoCo viewer ready. {title}. {len(frames)} samples, showing {frame_times[index]:.3f} s.",
              flush=True)
        displayed = -1
        while viewer.is_running():
            while not keys.empty():
                key = keys.get()
                if key == ord(" "):
                    if playback_start is None:
                        if index == len(frames) - 1:
                            index = 0
                        playback_start = time.monotonic() - 5 * frame_times[index]
                    else:
                        playback_start = None
                elif key in (ord("R"), ord("B"), ord("N"), 263, 262):
                    playback_start = None
                    delta = 1 if key in (ord("N"), 262) else -1
                    index = 0 if key == ord("R") else max(0, min(len(frames) - 1, index + delta))
                elif key in (ord("1"), ord("2")):
                    active_view = close_view if key == ord("1") else overview
                    with viewer.lock():
                        camera(active_view)
                    displayed = -1
            if playback_start is not None:
                elapsed = (time.monotonic() - playback_start) / 5
                index = max(0, int(np.searchsorted(frame_times, elapsed, side="right")) - 1)
                if index == len(frames) - 1:
                    playback_start = None
            if index != displayed:
                with viewer.lock():
                    show_pose(model, data, frames[index])
                    data.time = frame_times[index]
                    if fold_edge is not None and active_view is close_view:
                        viewer.cam.lookat[:] = data.flexvert_xpos[fold_edge].mean(axis=0)
                phase, force = frame_info[index]
                viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    f"{title}\nTime / Phase\nForce / Frame\nPlayback\nView",
                    f"Recorded MuJoCo trajectory\n{frame_times[index]:.3f} s / {phase}\n"
                    f"{force} / {index + 1} of {len(frames)}\nSpace, 0.2x | R: restart | Left/Right\n"
                    "1: close-up | 2: overview"))
                displayed = index
                viewer.sync()
            time.sleep(.01)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--start-time", type=float, default=0., help="initial recorded time in seconds")
    parser.add_argument("--fold-edge", type=int, nargs=2, help="follow a recorded YAM membrane edge in the close-up")
    parser.add_argument("--check", action="store_true", help="verify replay poses without opening a window")
    args = parser.parse_args()
    if not np.isfinite(args.start_time) or args.start_time < 0:
        parser.error("start time must be finite and nonnegative")
    if (args.run / "trajectory.npz").is_file():
        model, data, frames, frame_times, info, rows = load_yam_recording(args.run)
        index = int(np.argmin(np.abs(frame_times - args.start_time)))
        center = np.asarray(rows[index]["tip_mid"]) - [0., 0., .014]
        overview_center = np.asarray(rows[index]["com"]) + [0., 0., .055]
        fold_edge = None
        if args.fold_edge is not None:
            fold_edge = np.asarray(args.fold_edge)
            faces = model.flex_elem.reshape(-1, 3)
            if np.count_nonzero(np.isin(faces, fold_edge).sum(axis=1) == 2) != 2:
                parser.error("fold edge must be shared by two recorded membrane triangles")
            show_pose(model, data, frames[index])
            center = data.flexvert_xpos[fold_edge].mean(axis=0)
        if args.check:
            for frame in (0, index, len(frames) - 1):
                show_pose(model, data, frames[frame])
                assert np.isfinite(data.flexvert_xpos).all()
            print(f"Verified native mesh poses and recorded times across {len(frames)} archived YAM samples.")
        else:
            close = (center, .20, -55., -24.) if fold_edge is not None else (center, .21, 102., -32.)
            replay(model, data, frames, frame_times, info, index, "Repository PE bag | YAM",
                   close, (overview_center, .75, -61., -29.), fold_edge)
        return
    if args.fold_edge is not None:
        parser.error("fold edge follow mode requires a recorded YAM trial")
    model, data, frames, frame_times, info, points = load_recording(args.run)
    if args.check:
        for index in (0, len(frames) // 2, len(frames) - 1):
            show_pose(model, data, frames[index])
            np.testing.assert_allclose(data.flexvert_xpos, points[index], atol=1e-12, rtol=0)
        print(f"Verified first, middle and last mesh poses across {len(frames)} recorded samples.")
    else:
        index = int(np.argmin(np.abs(frame_times - args.start_time)))
        model.flex_rgba[0] = [.15, .78, .38, 1.]
        replay(model, data, frames, frame_times, info, index, "PE fold | YAM pads",
               ((0., 0., .009), .095, 90., -10.), ((0., 0., .009), .125, 125., -30.))


if __name__ == "__main__":
    main()

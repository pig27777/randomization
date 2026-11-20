"""Show a saved Stage 2 YAM trial in the native MuJoCo viewer."""

import argparse
import gzip
import json
from pathlib import Path
from queue import SimpleQueue
import time

import mujoco
import mujoco.viewer
import numpy as np

COLORS = np.array([[.13, .78, 1., 1.], [1., .49, .17, 1.]], dtype=np.float32)


def replay(run_dir):
    model = mujoco.MjModel.from_xml_path(str(run_dir / "model.xml"))
    data = mujoco.MjData(model)
    trace_file = run_dir / "trace.json"
    trace = json.loads(trace_file.read_bytes() if trace_file.is_file()
                       else gzip.decompress((run_dir / "trace.json.gz").read_bytes()))
    with np.load(run_dir / "trajectory.npz") as trajectory:
        qpos, sample_time = trajectory["qpos"], trajectory["time"]
    if not np.array_equal(sample_time, [row["t"] for row in trace]):
        raise ValueError("The platform trace and saved poses are out of sync")
    keys = SimpleQueue()
    hover = next((i for i, row in enumerate(trace) if row["phase"] == "lift_hold"
                  and row["proxy_longest_hover_s"] >= .20), len(trace) - 1)
    index = hover
    playback_start = None
    closeup = False
    size, origin, rotation = np.zeros(3), np.zeros(3), np.eye(3).ravel()

    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        def camera():
            with viewer.lock():
                viewer.cam.lookat[:] = [-.13, -.07, .24]
                viewer.cam.distance = 1.20
                if closeup:
                    viewer.cam.lookat[:] = data.xpos[model.body("parcel").id] + [0., 0., .065]
                    viewer.cam.distance = .53
                viewer.cam.azimuth = 40.
                viewer.cam.elevation = -22.
                viewer.opt.geomgroup[3] = False
                viewer.opt.sitegroup[4] = False

        def draw(i):
            row = trace[i]
            with viewer.lock():
                data.qpos[:] = qpos[i]
                data.time = float(sample_time[i])
                mujoco.mj_fwdPosition(model, data)
                scene = viewer.user_scn
                scene.ngeom = 0
                for side, label in enumerate(("left", "right")):
                    if row["proxy_capture"][side] > .5:
                        geom = scene.geoms[scene.ngeom]
                        scene.ngeom += 1
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, size, origin, rotation, COLORS[side])
                        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .0013,
                                             data.site(f"proxy_{label}").xpos, np.asarray(row["proxy_pad_m"][side]))
            fields = [f"{force:.2f} N tension  |  {normal:.2f} N grip"
                      for force, normal in zip(row["proxy_tension_N"], row["proxy_normal_N"], strict=True)]
            viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              "YAM linear_4310\nTime / phase\nLeft / blue\nRight / orange\nTable clearance\nPlayback",
                              f"Stage 2 PE proxy\n{sample_time[i]:.2f} s  |  {row['phase']}\n{fields[0]}\n{fields[1]}\n"
                              f"{row['proxy_clearance_m'] * 1000:.1f} mm\n"
                              "Space: play/pause  |  R: start  |  H: hover  |  C: camera"))
            viewer.sync()

        camera()
        draw(index)
        print(f"YAM proxy viewer ready at {sample_time[index]:.3f} s", flush=True)
        while viewer.is_running():
            before = index
            while not keys.empty():
                key = keys.get()
                if key == 32:
                    if playback_start is None and index == len(trace) - 1:
                        index = 0
                    playback_start = time.monotonic() - sample_time[index] if playback_start is None else None
                elif key in (ord("R"), ord("H")):
                    playback_start = None
                    index = 0 if key == ord("R") else hover
                elif key == ord("C"):
                    closeup = not closeup
                    camera()
                    viewer.sync()
            if playback_start is not None:
                index = int(np.clip(np.searchsorted(sample_time, time.monotonic() - playback_start) - 1,
                                    0, len(trace) - 1))
                if index == len(trace) - 1:
                    playback_start = None
            if before != index:
                draw(index)
            time.sleep(.012)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    replay(parser.parse_args().run_dir)

"""Replay the Stage 2 one-sided release in the native MuJoCo viewer."""

import argparse
import json
from pathlib import Path
from queue import SimpleQueue
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.low_order_parcel.lumped_contact_proxy import ContactState as C  # noqa: E402
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim  # noqa: E402


SCENE = """
<mujoco model="Stage 2 two-finger proxy replay">
  <visual>
    <global azimuth="105" elevation="-24"/>
    <headlight ambient=".42 .42 .42" diffuse=".60 .60 .60" specular=".15 .15 .15"/>
  </visual>
  <worldbody>
    <light directional="true" pos="-.1 -.3 .55" dir="0 .3 -1" diffuse=".85 .85 .85"/>
    <geom name="table" type="plane" size=".45 .45 .01" rgba=".24 .28 .32 1"/>
    <body name="parcel" mocap="true" pos="0 0 .016">
      <geom name="box" type="box" size=".075 .10 .016" pos="-.003 0 0" rgba=".61 .44 .27 1"/>
      <geom type="box" size=".022 .10 .00035" pos="-.003 0 .016" rgba=".79 .65 .45 1"/>
      <site name="left_mount" pos="-.027 0 .016" size=".003" rgba=".13 .78 1 1"/>
      <site name="right_mount" pos=".021 0 .016" size=".003" rgba="1 .49 .17 1"/>
    </body>
    <body name="left_pad" mocap="true" pos="-.02 0 .10">
      <geom type="box" pos="-.009 0 .012" size=".008 .016 .014" rgba=".13 .67 .93 1"/>
      <site size=".003" rgba=".13 .78 1 1"/>
    </body>
    <body name="right_pad" mocap="true" pos=".02 0 .10">
      <geom type="box" pos=".009 0 .012" size=".008 .016 .014" rgba=".98 .45 .14 1"/>
      <site size=".003" rgba="1 .49 .17 1"/>
    </body>
  </worldbody>
</mujoco>
"""
COLORS = ((.13, .78, 1., 1.), (1., .49, .17, 1.))


def prepare_recording(output):
    config = json.loads((ROOT / "configs/proxy_material_envelopes.yaml").read_bytes())
    nominal = config["materials"]["pe"]["nominal"]
    sim = MacroParcelSim.from_nominal(nominal, .002, com_offset=(.003, 0., 0.))
    trace = sim.run_suspended_load_ramp(grip_N=nominal["probe_grip_N"]).channels
    time_s = trace["time_s"]
    captured = trace["contact_state"][:, :, C.CAPTURE] > .5
    first_lost = np.argmax(captured[:-1] & ~captured[1:], axis=0) + 1
    if not all(captured[first_lost[i] - 1, i] and not captured[first_lost[i], i] for i in range(2)):
        raise ValueError("The recorded action must include both releases")
    metadata = dict(first_release_s=time_s[first_lost].tolist(), initial_frame=int(first_lost.mean()),
                    case="stage2 asymmetric PE suspended load ramp", dt_s=sim.dt)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "proxy-replay.npz", **trace)
    (output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run(output, check):
    metadata = json.loads((output / "summary.json").read_bytes())
    with np.load(output / "proxy-replay.npz") as saved:
        trace = {key: saved[key] for key in saved.files}
    time_s, contacts = trace["time_s"], trace["contact_state"]
    captured = contacts[:, :, C.CAPTURE] > .5
    index = metadata["initial_frame"]
    event = index
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)

    def pose(i):
        data.mocap_pos[:] = np.vstack((trace["com_m"][i], trace["pad_position_m"][i]))
        data.mocap_quat[0] = trace["orientation"][i]
        data.time = float(time_s[i])
        mujoco.mj_fwdPosition(model, data)

    if check:
        assert np.isfinite(trace["com_m"]).all() and np.isfinite(trace["orientation"]).all()
        np.testing.assert_allclose(np.linalg.norm(trace["orientation"], axis=1), 1., atol=1e-12)
        for i in (0, index, len(time_s) - 1):
            pose(i)
            np.testing.assert_allclose(data.xpos[model.body("parcel").id], trace["com_m"][i], atol=1e-12)
        if not captured[index, 0] or captured[index, 1]:
            raise ValueError("The UI start frame must show left hold and right release")
        print(f"Verified {len(time_s)} replay poses and an asymmetric loaded frame at {time_s[index]:.3f} s.")
        return

    keys = SimpleQueue()
    playing = None
    first_frame = int(np.searchsorted(time_s, time_s[event] - .70))
    pose(index)
    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        def show_frame(i):
            with viewer.lock():
                pose(i)
                viewer.cam.lookat[:] = trace["com_m"][i] + [0., 0., .057]
                scene = viewer.user_scn
                scene.ngeom = 0
                for side, name in enumerate(("left", "right")):
                    if captured[i, side]:
                        geom = scene.geoms[scene.ngeom]
                        scene.ngeom += 1
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                            np.eye(3).ravel(), np.asarray(COLORS[side], dtype=np.float32))
                        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .0012,
                                             data.site(f"{name}_mount").xpos, trace["pad_position_m"][i, side])
            fields = []
            for side in range(2):
                status = ("released" if not captured[i, side] else
                          "slipping" if trace["slip_rate_mps"][i, side] > 1e-7 else "holding")
                mm = contacts[i, side, C.SLIP_CUM] * 1000.
                force = trace["tension_N"][i, side]
                fields.append(f"{status}  |  {force:.2f} N tension  |  {mm:.2f} mm slip")
            viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              "Stage 2 contact proxy\nTime\nLeft / blue\nRight / orange\nPlayback",
                              f"PE load ramp replay\n{time_s[i]:.3f} s\n{fields[0]}\n{fields[1]}\n"
                              "Space: play/pause (0.25x)  |  R: before slip  |  E: one-sided hold"))
            viewer.sync()

        with viewer.lock():
            viewer.cam.distance = .49
            viewer.cam.azimuth = 103.
            viewer.cam.elevation = -23.
        show_frame(index)
        print(f"MuJoCo proxy viewer ready at {time_s[index]:.3f} s", flush=True)
        old_index = index
        while viewer.is_running():
            while not keys.empty():
                key = keys.get()
                if key == 32:
                    if playing is None and index == len(time_s) - 1:
                        index = first_frame
                    playing = time.monotonic() - 4 * time_s[index] if playing is None else None
                elif key in (ord("R"), ord("E"), 263, 262):
                    playing = None
                    if key in (ord("R"), ord("E")):
                        index = first_frame if key == ord("R") else event
                    else:
                        index = int(np.clip(index + (1 if key == 262 else -1), 0, len(time_s) - 1))
            if playing is not None:
                index = min(len(time_s) - 1, max(0, int(np.searchsorted(time_s, (time.monotonic() - playing) / 4)) - 1))
                if index == len(time_s) - 1:
                    playing = None
            if index != old_index:
                show_frame(index)
                old_index = index
            time.sleep(.012)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare_recording(args.output)
    run(args.output, args.check)


if __name__ == "__main__":
    main()

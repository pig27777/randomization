"""Check whether saved contact depths agree with their reported directions."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def measure(report, element):
    info = json.loads(report.read_text())
    gpu = Path(info["warp_recording"]) / "trial"
    cpu = Path(info["cpu_trial"])
    model = mujoco.MjModel.from_binary_path(str(gpu / "model.mjb"))
    data = mujoco.MjData(model)
    geom = model.geom("payload_geom_0").id
    face = model.flex_elem.reshape(-1, 3)[element]
    radius = float(model.flex_radius[0])
    result = dict(source=str(report), ccd_tolerance_m=info.get("ccd_tolerance_m", float(model.opt.ccd_tolerance)),
                  element=element, contact_radius_m=radius, frames={})
    with np.load(gpu / "trajectory.npz") as warp, np.load(cpu / "trajectory.npz") as host:
        for name, frame in info["frames"].items():
            source = host if name.startswith("cpu_") else warp
            time = float(frame["t_s"])
            index = int(np.argmin(np.abs(source["time"] - time)))
            if abs(source["time"][index] - time) > 1e-9:
                raise ValueError(f"Cannot align {name} with the archived configuration")
            data.qpos[:] = warp["final_qpos"] if name == "warp_failure" else source["qpos"][index]
            mujoco.mj_forward(model, data)
            surface = data.flexvert_xpos[face] - data.geom_xpos[geom]
            rotation = data.geom_xmat[geom].reshape(3, 3)
            contacts = {}
            for engine, key in (("cpu", "cpu_contacts"), ("warp", "device_contacts")):
                rows = []
                for contact in frame[key]:
                    if contact["element"] != element:
                        continue
                    normal = np.asarray(contact["normal_on_film"])
                    normal /= np.linalg.norm(normal)
                    payload_bound = np.abs(rotation.T @ normal) @ model.geom_size[geom]
                    projected = float(payload_bound - (surface @ normal).min() + radius)
                    native = -float(contact["distance_m"])
                    rows.append(dict(native_depth_m=native, projected_separation_m=projected,
                                     difference_m=projected - native, direction=normal.tolist()))
                contacts[engine] = rows
            result["frames"][name] = dict(t_s=time, **contacts)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--element", type=int, default=191)
    args = parser.parse_args()
    measurements = [measure(path, args.element) for path in args.reports]
    with args.output.open("x") as stream:
        stream.write(json.dumps(measurements, indent=2) + "\n")


if __name__ == "__main__":
    main()

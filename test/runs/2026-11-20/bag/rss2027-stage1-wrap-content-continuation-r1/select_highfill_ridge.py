"""Find a reachable natural ridge on the 90% whole-bag pressure reference."""

import argparse
import itertools
import json
from unittest.mock import patch

import mujoco
import numpy as np

from record_local_top import RUN, at_top_center, backend, protocol, surface_below_pad
from refine_parcel import physical_parcel, prepared_wrap  # noqa: E402
from select_pose import CENTER, position  # noqa: E402

CASE = RUN / "fold-aligned-90-energy/sample-076"
ROOT = RUN.parents[4]


def check_path(config, center, film_height):
    cfg = backend.Config(**config)
    with at_top_center(center), patch.object(physical_parcel, "wrap_generic", prepared_wrap(cfg.geometry)):
        robot = backend.scene(cfg)
    start = robot.initial_target
    tip_offset = start[2] - protocol.tips(robot).mean(axis=0)[2]
    seating = np.r_[center, film_height + tip_offset - .003]
    path = np.vstack((np.linspace(start, seating, 20),
                      np.linspace(seating, seating + [0., 0., cfg.lift_height], 71)[1:]))
    peak_translation = peak_rotation = 0.
    for target in path:
        residual = robot.solve(target)
        peak_translation = max(peak_translation, residual["position_mm"])
        peak_rotation = max(peak_rotation, residual["rotation_deg"])
        if peak_translation > 1. or peak_rotation > 1.:
            return dict(status="unreachable", peak_translation_mm=peak_translation, peak_rotation_deg=peak_rotation)
        robot.kin.qpos[robot.qadr] = robot.qtarget
        robot.kin.qpos[robot.fingers] = cfg.grasp_opening_m / 2.
        mujoco.mj_fwdPosition(robot.sim.m, robot.kin)
        for contact in robot.kin.contact:
            a, b = contact.geom
            if a >= 0 and b >= 0 and robot.robot_mask[a] and robot.robot_mask[b] and contact.dist < -1e-6:
                return dict(status="arm_self_collision", colliders=[robot.sim.m.geom(g).name for g in (a, b)])
    return dict(status="reachable", peak_translation_mm=peak_translation, peak_rotation_deg=peak_rotation)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--rotate", action="store_true", help="turn the same package 180 degrees on its support")
    args = parser.parse_args()
    trial = CASE / "settle/trial"
    result = json.loads((trial / "result.json").read_bytes())
    cfg = json.loads((CASE / "config.json").read_bytes())
    body_size = np.asarray(cfg["payload_size"][:2])
    offset = np.asarray(cfg["payload_offset_xy"])
    if result["outcome"] != "smoke_only" or result["sim_time_s"] < .3 or result["peaks"]["grip_N"]:
        raise ValueError("The ridge search requires a clean settled parcel")
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    data = mujoco.MjData(model)
    with np.load(trial / "trajectory.npz") as archive:
        data.qpos[:] = archive["qpos"][0]
        mujoco.mj_fwdPosition(model, data)
        global_height = float(data.flexvert_xpos[:, 2].max() + model.flex_radius[0])
        data.qpos[:] = archive["final_qpos"]
        mujoco.mj_fwdPosition(model, data)
    triangles = data.flexvert_xpos[model.flex_elem.reshape(-1, 3)]
    target = CASE / ("ridge-rotated" if args.rotate else "ridge-height")
    target.mkdir(exist_ok=False)
    if args.rotate:
        triangles[:, :, :2] = 2. * CENTER - triangles[:, :, :2]
        offset = -offset
        with np.load(physical_parcel.HERE / cfg["geometry"]) as archive:
            asset = dict(archive.items())
        for key in ("vertices", "reference_m", "current_m", "payload_origin_m", "payload_offset_xy"):
            asset[key][..., :2] *= -1.
        rotated = target / "prepared_mesh.npz"
        np.savez_compressed(rotated, **asset)
        cfg.update(geometry=f"../{rotated.relative_to(ROOT)}", payload_offset_xy=offset.tolist())
    axes = [np.linspace(-size / 2 - .01, size / 2 + .01, 18) for size in body_size]
    candidates = [row for x, y in itertools.product(*axes)
                  if (row := position(CENTER + offset + [x, y], cfg.get("yaw", 0.), triangles)) is not None
                  and row["pad_height_difference_m"] <= .002 and row["ridge_over_pads_m"] >= .002]
    candidates.sort(key=lambda row: -row["ridge_over_pads_m"])
    selected = None
    for candidate in candidates:
        shift = np.asarray(candidate["center_xy_m"]) - CENTER
        heights = [surface_below_pad(model, data, name, triangles, shift) for name in ("left_pad", "right_pad")]
        if abs(heights[0] - heights[1]) > .002:
            candidate["path"] = {"status": "uneven_open_pad_surfaces", "pad_surfaces_m": heights}
            continue
        try:
            check = check_path(cfg, candidate["center_xy_m"], max(heights))
        except RuntimeError as error:
            if not str(error).startswith(("YAM initial pose unreachable", "initial film contact penetration")):
                raise
            check = dict(status="initial_pose_unavailable", reason=str(error))
        candidate["path"] = check
        if check["status"] == "reachable":
            selected = candidate | dict(settled_pad_surfaces_m=heights)
            break
    report = dict(package_rotation_deg=180. if args.rotate else 0., candidates=candidates, selected=selected)
    (target / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if selected is None:
        return
    height = dict(source_time_s=result["sim_time_s"], global_braking_surface_m=global_height,
                  settled_pad_surfaces_m=selected["settled_pad_surfaces_m"],
                  local_braking_surface_m=max(selected["settled_pad_surfaces_m"]),
                  target_center_xy_m=selected["center_xy_m"])
    (target / "height.json").write_text(json.dumps(height, indent=2, allow_nan=False) + "\n")
    for name, seconds in (("config", 0.), ("smoke", .02), ("settle", .30)):
        inputs = cfg | {"smoke_seconds": seconds, "wall_limit": 1800.}
        (target / f"{name}.json").write_text(json.dumps(inputs, indent=2) + "\n")


if __name__ == "__main__":
    main()

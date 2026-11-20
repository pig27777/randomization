"""Find a reachable natural ridge on a settled whole-bag pressure reference."""

import argparse
from contextlib import nullcontext
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from record_local_top import RUN, at_top_center, backend, protocol, surface_below_pad
from refine_parcel import physical_parcel, prepared_wrap  # noqa: E402
from select_pose import CENTER, position  # noqa: E402

CASE = RUN / "fold-aligned-90-energy/sample-076"
ROOT = RUN.parents[4]


def top_robot(cfg, center):
    wrap = (patch.object(physical_parcel, "wrap_generic", prepared_wrap(cfg.geometry))
            if cfg.geometry.endswith("/prepared_mesh.npz") else nullcontext())
    with at_top_center(center), wrap:
        return backend.scene(cfg)


def check_path(cfg, robot, center, film_height):
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
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--output", type=Path, help="save a new screen without changing the source run")
    parser.add_argument("--observed-contents", action="store_true", help="center the search on the settled rigid body")
    parser.add_argument("--approach-time-s", type=float, help="use an archived approach pose before any sampled finger load")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rotate", action="store_true", help="turn the same package 180 degrees on its support")
    mode.add_argument("--yaw-scan", action="store_true", help="compare 12 top-gripper headings on the same parcel")
    args = parser.parse_args()
    case = args.case.resolve()
    trial = case / ("settle/trial" if args.approach_time_s is None else "recording/trial")
    result = json.loads((trial / "result.json").read_bytes())
    config = case / "config.json"
    cfg = json.loads(config.read_bytes()) if config.is_file() else result["config"]
    body_size = np.asarray(cfg["payload_size"][:2])
    offset = np.asarray(cfg["payload_offset_xy"])
    source_time = result["sim_time_s"]
    if args.approach_time_s is None:
        if (result["outcome"] != "smoke_only" or source_time < .3 or result["warnings"]
                or result["peaks"]["grip_N"] or result["peaks"]["down_N"] or result["peaks"]["penetration_mm"] > 1.):
            raise ValueError("The ridge search requires a clean settled parcel")
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    data = mujoco.MjData(model)
    with np.load(trial / "trajectory.npz") as archive:
        if args.approach_time_s is None:
            data.qpos[:] = archive["final_qpos"]
        else:
            index = int(np.argmin(abs(archive["time"] - args.approach_time_s)))
            trace = json.loads((trial / "trace.json").read_bytes())
            source_time = float(archive["time"][index])
            prefix = trace[:index + 1]
            if (abs(source_time - args.approach_time_s) > cfg["dt"] or prefix[-1]["phase"] != "approach"
                    or any(not row["finite"] or row["warnings"] or row["penetration_mm"] > 1.
                           or any(row["contact_magnitude_each_N"])
                           or row["payload_geometry"]["outside_corners"] or row["payload_geometry"]["intersecting_triangles"]
                           for row in prefix)):
                raise ValueError("The requested approach frame has no verified contact-free geometry prefix")
            data.qpos[:] = archive["qpos"][index]
        mujoco.mj_fwdPosition(model, data)
    if args.observed_contents:
        if cfg["mode"] != "rigid":
            parser.error("The observed position search requires one rigid payload")
        body = int(model.geom("payload_geom_0").bodyid[0])
        offset = data.xpos[body, :2] - CENTER
        rotation = data.xmat[body].reshape(3, 3)
        body_size = (abs(rotation) @ np.asarray(cfg["payload_size"]))[:2]
    triangles = data.flexvert_xpos[model.flex_elem.reshape(-1, 3)]
    name = "ridge-yaw-scan" if args.yaw_scan else "ridge-rotated" if args.rotate else "ridge-height"
    target = args.output.resolve() if args.output is not None else case / name
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
        cfg.update(geometry=f"../{rotated.relative_to(ROOT)}",
                   payload_offset_xy=(-np.asarray(cfg["payload_offset_xy"])).tolist())
    axes = [np.linspace(-size / 2 - .01, size / 2 + .01, 18) for size in body_size]
    yaws = list(range(0, 180, 15)) if args.yaw_scan else [cfg.get("yaw", 0.)]
    footprints = [row for yaw, x, y in itertools.product(yaws, *axes)
                  if (row := position(CENTER + offset + [x, y], yaw, triangles)) is not None]
    balanced = [row for row in footprints if row["pad_height_difference_m"] <= .002]
    candidates = [row for row in balanced if row["ridge_over_pads_m"] >= .002]
    candidates.sort(key=lambda row: -row["ridge_over_pads_m"])
    selected = None
    for candidate in candidates:
        center = np.asarray(candidate["center_xy_m"])
        config = backend.Config(**(cfg | {"yaw": candidate["yaw_deg"]}))
        try:
            if args.yaw_scan:
                robot = top_robot(config, center)
                robot.sim.d.qpos[robot.fingers] = config.grasp_opening_m / 2.
                mujoco.mj_fwdPosition(robot.sim.m, robot.sim.d)
                footprint = robot.sim.m, robot.sim.d, np.zeros(2)
            else:
                footprint = model, data, center - CENTER
            native_model, native_data, shift = footprint
            heights = [surface_below_pad(native_model, native_data, name, triangles, shift)
                       for name in ("left_pad", "right_pad")]
            if abs(heights[0] - heights[1]) > .002:
                candidate["path"] = {"status": "uneven_open_pad_surfaces", "pad_surfaces_m": heights}
                continue
            if not args.yaw_scan:
                robot = top_robot(config, center)
            check = check_path(config, robot, center, max(heights))
        except RuntimeError as error:
            if not str(error).startswith(("YAM initial pose unreachable", "initial film contact penetration")):
                raise
            check = dict(status="initial_pose_unavailable", reason=str(error))
        except ValueError as error:
            if not str(error).endswith("contains no membrane"):
                raise
            check = dict(status="open_pad_footprint_has_no_membrane", reason=str(error))
        candidate["path"] = check
        if check["status"] == "reachable":
            selected = candidate | dict(settled_pad_surfaces_m=heights)
            cfg["yaw"] = candidate["yaw_deg"]
            break
    report = dict(source_case=case.relative_to(ROOT).as_posix(), source_time_s=source_time,
                  package_rotation_deg=180. if args.rotate else 0., yaws_deg=yaws,
                  search_center_xy_m=(CENTER + offset).tolist(), search_body_xy_size_m=body_size.tolist(),
                  footprint_count=len(footprints), balanced_count=len(balanced),
                  best_balanced=max(balanced, key=lambda row: row["ridge_over_pads_m"], default=None),
                  candidates=candidates, selected=selected)
    (target / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if selected is None:
        return
    height = dict(source_time_s=source_time,
                  global_braking_surface_m=float(robot.initial_target[2] - config.press_start_offset_m),
                  settled_pad_surfaces_m=selected["settled_pad_surfaces_m"],
                  local_braking_surface_m=max(selected["settled_pad_surfaces_m"]),
                  target_center_xy_m=selected["center_xy_m"])
    if args.yaw_scan:
        height["record_integration_state"] = True
    (target / "height.json").write_text(json.dumps(height, indent=2, allow_nan=False) + "\n")
    for name, seconds in (("config", 0.), ("smoke", .02), ("settle", .30)):
        inputs = cfg | {"smoke_seconds": seconds, "wall_limit": 1800.}
        (target / f"{name}.json").write_text(json.dumps(inputs, indent=2) + "\n")


if __name__ == "__main__":
    main()

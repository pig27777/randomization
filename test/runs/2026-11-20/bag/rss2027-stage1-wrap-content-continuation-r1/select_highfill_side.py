"""Find an open-finger YAM approach to settled parcels from either fill group."""

import argparse
import itertools
import json
from unittest.mock import patch

import mujoco
import numpy as np

from select_side_flap import (ADVANCE, backend, choose_open_route, local_pad_contacts, open_route_candidates,
                              physical_parcel, prepared_wrap, screen, shifted_target)

ROOT = ADVANCE.parents[2]
RUN = ROOT / "fold-aligned-90-energy"
GROUPS = {"0.90": (RUN, (66, 76, 82), .020),
          "0.60": (ROOT / "fold-aligned-60-side-energy", (34, 54, 58), .012)}
ANGLES = tuple(itertools.product((70., 75., 85., 95., 105., 115.), (15., 25., 40.)))


def sample_routes(directory, sample, half_depth, include_pad_edges):
    case = directory / f"sample-{sample:03}"
    settle = case / "settle"
    summary = settle / "summary.json"
    if not summary.is_file():
        return case, None, dict(sample=sample, selected=None, rejected="No successful settling output")
    original = json.loads(summary.read_bytes())
    if (original["outcome"] != "smoke_only" or original["sim_time_s"] < .3 or original["warnings"]
            or original["peaks"]["grip_N"] or original["peaks"]["penetration_mm"] > 1.):
        return case, None, dict(sample=sample, selected=None, rejected="Original settling criteria not met")
    with np.load(settle / "trial/trajectory.npz") as archive:
        qpos = archive["final_qpos"]
    config = original["config"] | dict(grasp_face="side", press_target_N=None, smoke_seconds=0.,
                                        wall_limit=1400., simulation_seconds=9., side_start_distance=.08,
                                        side_height_offset=.014, side_insert_speed_m_s=.03)
    candidates, paths = [], []
    for yaw, tilt in ANGLES:
        try:
            options, _ = open_route_candidates(config | {"yaw": yaw, "tilt": tilt}, qpos,
                                               capture_half_depth_m=half_depth, include_pad_edges=include_pad_edges)
        except RuntimeError as error:
            if not str(error).startswith(("YAM initial pose unreachable", "initial film contact penetration")):
                raise
            paths.append(dict(yaw_deg=yaw, tilt_deg=tilt, status=str(error)))
            continue
        candidates.extend(options)
        paths.append(dict(yaw_deg=yaw, tilt_deg=tilt, native_initial_pose="available", candidates=len(options)))
    chosen, selected = choose_open_route(config, candidates)
    proof = dict(sample=sample, source_time_s=original["sim_time_s"], bag_size_m=config["bag_size"],
                 payload_size_m=config["payload_size"], payload_offset_xy_m=config["payload_offset_xy"],
                 support_height_m=config["support_height"], predicted_gap_m=.008, capture_half_depth_m=half_depth,
                 included_supplementary_pad_geoms=include_pad_edges, selected=selected,
                 candidate_count=len(candidates), orientations=paths)
    return case, chosen, proof


def restore_settled_payload(case, config, output, selected):
    cfg = backend.Config(**config)
    with patch.object(physical_parcel, "wrap_generic", prepared_wrap(cfg.geometry)):
        parcel, xml = backend.parcel_world(cfg)
    parcel.m = parcel.compile_model(xml)
    data = mujoco.MjData(parcel.m)
    with np.load(case / "settle/trial/trajectory.npz") as archive:
        data.qpos[:] = archive["final_qpos"][:parcel.m.nq]
        data.qvel[:] = archive["final_qvel"][:parcel.m.nv]
    checkpoint = output / "initial_state.npz"
    backend.save_initial_state(parcel, data, cfg, checkpoint)
    cfg_path = str(checkpoint.relative_to(ROOT.parents[4]))
    new_config = config | {"initial_state": cfg_path}
    with shifted_target(np.asarray(selected["shift_m"]), selected["close_advance_m"]), \
            patch.object(physical_parcel, "wrap_generic", prepared_wrap(cfg.geometry)):
        robot = backend.scene(backend.Config(**new_config))
    np.testing.assert_array_equal(robot.sim.d.qpos[:parcel.m.nq], data.qpos)
    np.testing.assert_array_equal(robot.sim.d.qvel[:parcel.m.nv], data.qvel)
    measure = backend.measure(robot)
    if any(measure["contact_magnitude_each_N"]):
        raise ValueError(f"The candidate already has finger reactions in the restored parcel state: {measure}")
    return new_config


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--group", choices=tuple(GROUPS), default="0.90")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--deeper-seat", action="store_true")
    mode.add_argument("--settled-entry", action="store_true")
    mode.add_argument("--backoff-entry", action="store_true")
    mode.add_argument("--rounded-pad", action="store_true")
    mode.add_argument("--horizontal-entry", action="store_true")
    args = parser.parse_args()
    if (args.settled_entry or args.backoff_entry or args.rounded_pad or args.horizontal_entry) and args.group != "0.60":
        parser.error("The restored-entry control uses the settled 60% sample")
    if args.deeper_seat:
        if args.group != "0.90":
            parser.error("The deeper seating control uses the original 90% sample")
        prepare_deeper_seat()
        return
    if args.backoff_entry:
        prepare_backoff_entry()
        return
    if args.rounded_pad:
        prepare_rounded_pad()
        return
    if args.horizontal_entry:
        prepare_horizontal_entry()
        return
    directory, samples, half_depth = GROUPS[args.group]
    candidates = [sample_routes(directory, sample, half_depth, args.settled_entry) for sample in samples]
    best = max((item for item in candidates if item[2]["selected"]),
               key=lambda item: item[2]["selected"]["paired_depth_score_m"], default=None)
    report = dict(samples=[row[2] for row in candidates], best_sample=best[2]["sample"] if best else None)
    prefix = "side-open-settled" if args.settled_entry else "side-open"
    with (directory / f"{prefix}-screen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if best:
        case, config, proof = best
        output = case / f"{prefix}-route"
        output.mkdir(exist_ok=False)
        if args.settled_entry:
            config = restore_settled_payload(case, config, output, proof["selected"])
        (output / "selection.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
        for name, seconds in (("config", 0.), ("smoke", .020), ("settle", .30)):
            (output / f"{name}.json").write_text(json.dumps(config | {"smoke_seconds": seconds}, indent=2) + "\n")


def prepare_deeper_seat():
    case = RUN / "sample-066"
    prior = case / "side-open-route"
    source = json.loads((prior / "selection.json").read_bytes())
    config = json.loads((prior / "config.json").read_bytes())
    original = source["selected"]
    raw = json.loads((prior / "force-summary.json").read_bytes())
    if raw["bilateral_samples"] > 0 or raw["first_lift"] is None:
        raise ValueError("The seating comparison requires the archived unilateral lift")
    with np.load(case / "settle/trial/trajectory.npz") as archive:
        qpos = archive["final_qpos"]
    options, _ = open_route_candidates(config, qpos)
    deeper = [row for row in options if abs(row["close_advance_m"] - .075) < 1e-9
              and np.linalg.norm(np.asarray(row["shift_m"]) - original["shift_m"]) < 1e-9]
    _, selected = choose_open_route(config, deeper)
    output = case / "side-open-route-deeper"
    output.mkdir(exist_ok=False)
    proof = dict(source, selected=selected, original_close_advance_m=original["close_advance_m"])
    (output / "selection.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    if selected:
        for name in ("config.json", "smoke.json", "settle.json"):
            (output / name).write_bytes((prior / name).read_bytes())


def prepare_backoff_entry():
    case = GROUPS["0.60"][0] / "sample-058"
    prior = case / "side-open-settled-route"
    source = json.loads((prior / "selection.json").read_bytes())
    config = json.loads((prior / "config.json").read_bytes())
    earlier = json.loads((prior / "settle/summary.json").read_bytes())
    if earlier["peaks"]["grip_N"] <= 0.:
        raise ValueError("The initial-position control requires the prior early contact")
    shift_m = .025
    config["side_start_distance"] += shift_m
    selected = source["selected"].copy()
    selected["close_advance_m"] += shift_m
    path = screen(config, np.asarray(selected["shift_m"]), selected["close_advance_m"])
    output = case / "side-open-backoff-route"
    output.mkdir(exist_ok=False)
    selected["path_check"] = {key: value for key, value in path.items() if key != "trace"}
    (output / "selection.json").write_text(json.dumps(dict(source, selected=selected, initial_backoff_m=shift_m),
                                                      indent=2, allow_nan=False) + "\n")
    if path["status"] == "reachable":
        for name, seconds in (("config", 0.), ("smoke", .020), ("settle", .30)):
            (output / f"{name}.json").write_text(json.dumps(config | {"smoke_seconds": seconds}, indent=2) + "\n")


def prepare_rounded_pad():
    case = GROUPS["0.60"][0] / "sample-058"
    prior = case / "side-open-backoff-route"
    source = json.loads((prior / "selection.json").read_bytes())
    force = json.loads((prior / "force-summary.json").read_bytes())
    first_lift = force["first_lift"]
    if first_lift is None or "right_pad_edge_b" not in first_lift["colliders"] or force["bilateral_samples"]:
        raise ValueError("The rounded-pad comparison requires the archived unilateral legacy-edge contact")
    source["disabled_pad_geoms"] = [f"{side}_pad_edge_{edge}" for side in ("left", "right") for edge in ("a", "b")]
    output = case / "side-open-rounded-pad"
    output.mkdir(exist_ok=False)
    (output / "selection.json").write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
    for name in ("config.json", "smoke.json", "settle.json"):
        (output / name).write_bytes((prior / name).read_bytes())


def prepare_horizontal_entry():
    case = GROUPS["0.60"][0] / "sample-058"
    prior = case / "side-open-rounded-pad"
    source = json.loads((prior / "selection.json").read_bytes())
    config = json.loads((prior / "config.json").read_bytes())
    first_lift = json.loads((prior / "force-summary.json").read_bytes())["first_lift"]
    contacts = json.loads((prior / "recording/capture-contacts.json").read_bytes())
    sample = next(row for row in contacts if row["t_s"] == first_lift["t_s"])
    normals = []
    for pad in (0, 1):
        components = [np.asarray(contact["normal_on_pad"]) * contact["force_contact_N"][0]
                      for contact in sample["contacts"] if contact["pad"] == pad and contact["force_contact_N"][0] > .05]
        normal = (2 * pad - 1) * np.sum(components, axis=0)
        normals.append(normal / np.linalg.norm(normal))
    yaw = np.deg2rad(97.5)
    inward = np.array([-np.sin(yaw), np.cos(yaw), 0.])
    up = np.array([0., 0., 1.])
    source_shift = np.asarray(source["selected"]["shift_m"])
    advance = source["selected"]["close_advance_m"]
    evaluations = []
    with local_pad_contacts({}, source["disabled_pad_geoms"]):
        for tilt in (0., 5., 10., 15., 20., 25.):
            angle = np.deg2rad(tilt)
            closing = -np.sin(angle) * inward - np.cos(angle) * up
            projections = np.asarray(normals) @ closing
            height = config["side_start_distance"] * (np.sin(np.deg2rad(config["tilt"])) - np.sin(angle))
            paths = []
            for offset in (0., height):
                try:
                    result = screen(config | {"yaw": 97.5, "tilt": tilt}, source_shift + offset * up, advance)
                except RuntimeError as error:
                    if not str(error).startswith("YAM initial pose unreachable"):
                        raise
                    result = dict(status="initial_pose_unreachable", reason=str(error))
                paths.append(dict(result, shift_vertical_m=float(offset)))
                paths[-1].pop("trace", None)
            evaluations.append(dict(tilt_deg=tilt, left_right_projection=projections.tolist(), paths=paths))
    out = case / "side-open-horizontal"
    out.mkdir(exist_ok=False)
    viable = [row["tilt_deg"] for row in evaluations if min(row["left_right_projection"]) > .70
              and any(path["status"] == "reachable" for path in row["paths"])]
    proof = dict(source_time_s=first_lift["t_s"], yaw_deg=97.5, aligned_reachable_tilts_deg=viable,
                 evaluations=evaluations)
    (out / "selection.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

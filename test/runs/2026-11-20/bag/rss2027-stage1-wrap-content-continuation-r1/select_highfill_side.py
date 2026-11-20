"""Find an open-finger YAM approach to one of three settled 90% parcels."""

import argparse
import itertools
import json

import numpy as np

from select_side_flap import ADVANCE, choose_open_route, open_route_candidates

RUN = ADVANCE.parents[2] / "fold-aligned-90-energy"
SAMPLES = (66, 76, 82)
ANGLES = tuple(itertools.product((70., 75., 85., 95., 105., 115.), (15., 25., 40.)))


def sample_routes(sample):
    case = RUN / f"sample-{sample:03}"
    settle = case / "settle"
    original = json.loads((settle / "summary.json").read_bytes())
    if (original["outcome"] != "smoke_only" or original["sim_time_s"] < .3 or original["warnings"]
            or original["peaks"]["grip_N"] or original["peaks"]["penetration_mm"] > 1.):
        raise ValueError(f"Sample {sample} requires the original successful settling run")
    with np.load(settle / "trial/trajectory.npz") as archive:
        qpos = archive["final_qpos"]
    config = original["config"] | dict(grasp_face="side", press_target_N=None, smoke_seconds=0.,
                                        wall_limit=1400., simulation_seconds=9., side_start_distance=.08,
                                        side_height_offset=.014, side_insert_speed_m_s=.03)
    candidates, paths = [], []
    for yaw, tilt in ANGLES:
        try:
            options, _ = open_route_candidates(config | {"yaw": yaw, "tilt": tilt}, qpos)
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
                 support_height_m=config["support_height"], predicted_gap_m=.008, selected=selected,
                 candidate_count=len(candidates), orientations=paths)
    return case, chosen, proof


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--deeper-seat", action="store_true")
    args = parser.parse_args()
    if args.deeper_seat:
        prepare_deeper_seat()
        return
    candidates = [sample_routes(sample) for sample in SAMPLES]
    best = max((item for item in candidates if item[2]["selected"]),
               key=lambda item: item[2]["selected"]["paired_depth_score_m"], default=None)
    report = dict(samples=[row[2] for row in candidates], best_sample=best[2]["sample"] if best else None)
    with (RUN / "side-open-screen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if best:
        case, config, proof = best
        output = case / "side-open-route"
        output.mkdir(exist_ok=False)
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


if __name__ == "__main__":
    main()

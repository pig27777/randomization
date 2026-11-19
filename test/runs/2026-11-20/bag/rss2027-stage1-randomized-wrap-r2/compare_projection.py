"""Replay the archived randomized inner contours with exact box footprint projection."""

import argparse
import ast
from collections import Counter
import csv
from functools import partial
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

RUN = Path(__file__).resolve().parent
BASE = RUN.with_name("rss2027-stage1-randomized-wrap-r1")
sys.path.insert(0, str(BASE))
from check_wrapping import DESIGN, MESH  # noqa: E402
from bag_sim import physical_parcel as parcel  # noqa: E402


def selected_cases(baseline):
    for group in dict.fromkeys(row["group"] for row in baseline):
        rows = [row for row in baseline if row["group"] == group]
        passed = sorted((row for row in rows if row["status"] == "enclosed"),
                        key=lambda row: float(row["maximum_edge_strain"]))
        if passed:
            yield passed[0]
        failures = [row for row in rows if row["status"] == "initial_strain"
                    and "'box_seam_clearance': False" in row["error"]]
        failures.sort(key=lambda row: preparation(row["error"])["maximum_edge_strain"])
        if failures:
            yield failures[0]
            yield failures[len(failures) // 2]


def preparation(error):
    return ast.literal_eval(error.split(": ", 1)[1])


def evaluate(base, sample, shell):
    width = tuple(float(sample[f"inside_{axis}_m"]) for axis in "xy")
    shift = [(float(sample[f"seam_{axis}_minus_m"]) - float(sample[f"seam_{axis}_plus_m"])) / 2
             for axis in "xy"]
    cfg = parcel.Config(geometry=MESH, bag_size=width, center=tuple(shift),
                       payload_size=tuple(float(sample[f"payload_{ax}_m"]) for ax in "xyz"),
                       payload_offset_xy=tuple(float(sample[f"offset_{ax}_m"]) for ax in "xy"), **shell)
    before = base["maximum_edge_strain"] or preparation(base["error"])["maximum_edge_strain"]
    result = dict(sample=int(base["sample"]), group=base["group"], fill_fraction=float(base["fill_fraction"]),
                  original_status=base["status"], original_maximum_edge_strain=float(before),
                  exact_status="enclosed", exact_error="")
    try:
        env = parcel.load_envelope(cfg)
    except ValueError as exc:
        reason = str(exc)
        if not reason.startswith(("initial film strain exceeds", "initial film does not enclose")):
            raise
        prep = preparation(reason)
        result.update(exact_status="initial_strain" if reason.startswith("initial film strain") else "initial_enclosure",
                      exact_error=reason)
    else:
        prep = env["metadata"]["preparation"]
        np.testing.assert_allclose(env["metadata"]["reference_area_m2"], 2 * np.prod(width), rtol=0, atol=1e-12)
        if (prep["maximum_edge_strain"] > .05 or prep["payload_intersecting_triangles"] or
                np.any(np.abs(prep["payload_corner_winding"]) < .5)):
            raise RuntimeError(f"An invalid envelope was accepted: {result}, {prep}")
    result.update(exact_maximum_edge_strain=prep["maximum_edge_strain"],
                  exact_minimum_wall_gap_m=prep["minimum_wall_gap_m"],
                  payload_intersecting_triangles=prep.get("payload_intersecting_triangles"),
                  outside_corners=(np.count_nonzero(np.abs(prep["payload_corner_winding"]) < .5)
                                   if "payload_corner_winding" in prep else None))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    with (BASE / "wrapping.csv").open(newline="") as stream:
        baseline = [row for row in csv.DictReader(stream) if row["contour"] == "inside_seams"]
    with DESIGN.open(newline="") as stream:
        design = list(csv.DictReader(stream))
    shell = parcel.parcel_profile("pe")["shell"]
    records = []
    module_spec = importlib.util.spec_from_file_location("bag_sim._rss2027_wrap_clip", RUN / "wrap_source.py")
    variant = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(variant)
    with patch.object(parcel, "wrap_generic", partial(variant.wrap_generic, clip_box_footprint=True)):
        for base in selected_cases(baseline):
            sample_id = int(base["sample"])
            result = evaluate(base, design[sample_id - 1], shell)
            records.append(result)
            print(f"{sample_id}: {result['original_status']} -> {result['exact_status']}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "pilot.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    groups = {group: {method: Counter(r[f"{method}_status"] for r in records if r["group"] == group)
                      for method in ("original", "exact")}
              for group in dict.fromkeys(row["group"] for row in records)}
    summary = dict(samples=len(records), groups=groups)
    with (args.output / "pilot.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

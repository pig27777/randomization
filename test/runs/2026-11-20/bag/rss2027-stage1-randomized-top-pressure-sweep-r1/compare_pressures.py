"""Compare actual pressure, film entry and loss in one randomized whole bag."""

import argparse
import csv
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[4]
BASE = RUN.with_name("rss2027-stage1-randomized-top-local-mesh-r1")
sys.path[:0] = [str(BASE), str(RUN.with_name("rss2027-stage1-reference-curves-r1")),
               str(RUN.with_name("rss2027-stage1-top-aperture-height-r1"))]
from export_curves import COLUMNS, export_case, load_json  # noqa: E402
from measure_height import replay  # noqa: E402
from measure_surface_excess import read_case  # noqa: E402

CASES = {"low": RUN / "low/refined", "original": BASE / "refined", "high": RUN / "high/refined"}
TARGETS = {"low": .3, "original": .5, "high": .8}
COLUMN = {name: i for i, name in enumerate(COLUMNS)}


def summarize(name, directory):
    trial = directory / "recording/trial"
    result = load_json(trial / "result.json")
    base = load_json(BASE / "refined/config.json")
    config = load_json(directory / "config.json")
    changed = set(key for key in config if config[key] != base[key])
    if set(config) != set(base) or changed - {"press_target_N", "press_timeout_s", "simulation_seconds"}:
        raise ValueError(f"Unexpected comparison differences for {name}: {changed}")
    if result["normal_press"]["target_N"] != TARGETS[name] or result["dimensions"]["film_vertices"] != 318:
        raise ValueError(f"Wrong pressure or mesh for {name}")
    data = dict(target_N=TARGETS[name], outcome=result["outcome"], reason=result["reason"], warnings=result["warnings"],
                pressure=result["normal_press"], peak_penetration_mm=result["peaks"]["penetration_mm"])
    if not (directory / "capture/capture-audit.csv").exists():
        trace = load_json(trial / "trace.json")
        if any(row["phase"] in ("close", "lift", "lift_hold") for row in trace):
            raise ValueError(f"Missing closure replay for {name}")
        return data | dict(closure_frames=0), []
    case = export_case(name, "pe", trial.relative_to(ROOT / "test/runs"), {})
    surface = {row["t_s"]: row["extra_length_m"] for row in read_case(directory)}
    samples = {row[0]: row for row in case["samples"]}
    positions = replay(case, COLUMN)
    for row in positions:
        trace = samples[row["t_s"]]
        row["actual_down_N"] = trace[COLUMN["downforce_N"]]
        row["left_inner_candidates"] = trace[COLUMN["left_inner_candidates"]]
        row["right_inner_candidates"] = trace[COLUMN["right_inner_candidates"]]
        row["bag_support_N"] = trace[COLUMN["support_N"]]
        row["rim_height_above_tip_m"] = (None if row["membrane_z_max_m"] is None
                                         else row["membrane_z_max_m"] + .020)
        row["surface_excess_length_m"] = surface.get(row["t_s"])
    closure = [row for row in positions if row["phase"] == "close"]
    with (directory / "capture/capture-audit.csv").open(newline="") as stream:
        native = list(csv.DictReader(stream))
    loaded = sum(int(row["left_inner_candidates"]) > 0 and int(row["right_inner_candidates"]) > 0
                 and min(float(row["left_grip_N"]), float(row["right_grip_N"])) > .05
                 and float(row["tension_N"]) > .001 for row in native)
    return data | dict(closure_frames=len(closure), material_present_frames=sum(
        row["aperture_reference_length_m"] > 1e-9 for row in closure),
        peak_material=max(closure, key=lambda row: row["aperture_reference_length_m"]),
        peak_extra_length=max(closure, key=lambda row: row["surface_excess_length_m"]),
        positive_bilateral_samples=loaded,
        first_lift=next((row for row in positions if row["phase"] == "lift"), None)), positions


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("cases", choices=CASES, nargs="*")
    parser.add_argument("--output", type=Path, default=RUN / "pressure-comparison.json")
    args = parser.parse_args()
    report, rows = {}, []
    for name in args.cases or CASES:
        report[name], positions = summarize(name, CASES[name])
        rows.extend(positions)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    if rows:
        with args.output.with_suffix(".csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

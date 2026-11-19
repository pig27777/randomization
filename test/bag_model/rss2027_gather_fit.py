"""Measure a local PE gathering slope inside a recorded pressure window."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def fit(case):
    config = json.loads((case / "config.json").read_text())
    summary = json.loads((case / "summary.json").read_text())
    raw_window = summary.get("gather_pressure_window_s")
    with (case / "curve.csv").open() as stream:
        rows = [(float(row["t_s"]), 2 * float(row["gather_per_pad_m"]), float(row["fold_excess_length_m"]),
                 float(row["left_fz_N"]) + float(row["right_fz_N"]))
                for row in csv.DictReader(stream) if row["phase"] == "gather"]
    if not summary["numerically_valid"] or not rows or raw_window is None:
        return dict(qualifying_intervals=[], raw_pressure_window_s=raw_window, reason="no usable raw pressure window")
    time_s, inward, retained, pressure = np.asarray(rows).T
    peak = float(retained.max())
    target = config["press_force"]
    usable = (raw_window[0] - 1e-9 <= time_s) & (time_s <= raw_window[1] + 1e-9)
    usable &= (.8 * target <= pressure) & (pressure <= 1.2 * target)
    usable &= (.10 * peak <= retained) & (retained <= .70 * peak)
    indices = np.flatnonzero(usable)
    intervals = []
    for group in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        if not len(group):
            continue
        first, last = group[0], group[-1]
        if time_s[last] - time_s[first] < .020 - 1e-9:
            continue
        rise = float(retained[last] - retained[first])
        travel = float(inward[last] - inward[first])
        if rise < config["material"]["contact_radius_m"] or travel <= 0:
            continue
        x, y = inward[group], retained[group]
        slope, intercept = np.polyfit(x, y, 1)
        fraction = np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - y.mean()) ** 2)
        intervals.append(dict(start_s=float(time_s[first]), end_s=float(time_s[last]), sample_count=len(group),
                              inward_travel_change_m=travel, fold_excess_change_m=rise,
                              pressure_N=[float(pressure[group].min()), float(pressure[group].max())],
                              slope_fold_m_per_total_inward_m=float(slope), r_squared=float(1 - fraction)))
    return dict(qualifying_intervals=intervals, raw_pressure_window_s=raw_window, peak_fold_excess_m=peak)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = fit(args.case)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

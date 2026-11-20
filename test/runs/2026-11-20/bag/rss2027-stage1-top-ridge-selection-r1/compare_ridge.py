"""Compare motion and membrane aperture before and after pressure relief."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[4]
sources = ("rss2027-stage1-randomized-top-local-mesh-r1", "rss2027-stage1-top-aperture-height-r1",
           "rss2027-stage1-reference-curves-r1")
sys.path[:0] = [str(RUN.with_name(name)) for name in sources]
from export_curves import COLUMNS, export_case, load_json  # noqa: E402
from measure_height import replay  # noqa: E402
from measure_surface_excess import read_case  # noqa: E402


def fold_window(surface, band):
    time, force, gap, extra = np.array([[row[key] for key in ("t_s", "downforce_N", "actual_gap_m", "extra_length_m")]
                                      for row in surface]).T
    extra -= extra[0]
    if extra.max() <= 0.:
        return None
    eligible = (force >= band[0]) & (force <= band[1]) & (extra >= .10 * extra.max()) & (extra <= .70 * extra.max())
    spans = np.flatnonzero(np.diff(np.r_[False, eligible, False])).reshape(-1, 2)
    windows = [(a, b) for a, b in spans if time[b - 1] - time[a] >= .020 - 1e-9 and extra[b - 1] - extra[a] >= .00025]
    if not windows:
        return None
    a, b = max(windows, key=lambda pair: time[pair[1] - 1] - time[pair[0]])
    x = gap[0] - gap[a:b]
    y = extra[a:b]
    if np.any(np.diff(x) <= 0.):
        return None
    coefficient, intercept = np.polyfit(x, y, 1)
    residual = y - coefficient * x - intercept
    return dict(t_start_s=float(time[a]), t_last_s=float(time[b - 1]), samples=len(x),
                mean_pressure_N=float(force[a:b].mean()), fold_growth_m=float(y[-1] - y[0]),
                slope_m_per_m=float(coefficient),
                r_squared=float(1. - np.square(residual).sum() / np.square(y - y.mean()).sum()))


def summarize(name, path):
    trial = path / "recording/trial"
    result = load_json(trial / "result.json")
    summary = {"outcome": result["outcome"], "reason": result["reason"], "peaks": result["peaks"],
               "pressure": result["normal_press"]}
    if not (path / "capture/capture-audit.csv").exists():
        return summary
    record = export_case(name, "pe", trial.relative_to(ROOT / "test/runs"), {})
    positions = replay(record, {name: i for i, name in enumerate(COLUMNS)})
    surface = read_case(path)
    with (path / "capture/capture-audit.csv").open(newline="") as stream:
        aperture = list(csv.DictReader(stream))
    candidates = [row for row in aperture if int(row["left_inner_candidates"]) and int(row["right_inner_candidates"])]
    positive = [row for row in candidates if float(row["tension_N"]) > .001
                and min(float(row["left_grip_N"]), float(row["right_grip_N"])) > .05]
    return summary | dict(candidates_with_bilateral_inner_sources=len(candidates), positive_bilateral_samples=len(positive),
                          first_lift=next((row for row in positions if row["phase"] == "lift"), None),
                          peak_fold=max(surface, key=lambda row: row["extra_length_m"]),
                          peak_total=max(surface, key=lambda row: row["aperture_reference_length_m"]),
                          fold_window=fold_window(surface, result["normal_press"]["band_N"]))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    short, full, relieved = RUN, RUN / "long-approach", RUN / "closure-relief"
    traces = {name: load_json(path / "recording/trial/trace.json")
              for name, path in (("short", short), ("full", full))}
    if traces["short"] != traces["full"][:len(traces["short"])]:
        raise ValueError("The first and extended approaches disagree before the original cutoff")
    report = dict(short_prefix_frames=len(traces["short"]), baseline=summarize("ridge", full))
    if not args.baseline:
        event = load_json(relieved / "recording/closure-relief.json")
        relief_trace = load_json(relieved / "recording/trial/trace.json")
        before = {name: [row for row in trace if row["t"] <= event["time_s"]]
                  for name, trace in (("baseline", traces["full"]), ("relieved", relief_trace))}
        if before["baseline"] != before["relieved"]:
            raise ValueError("The approach and closure prefix changed before relief")
        report.update(relief_event=event, relief_prefix_frames=len(before["relieved"]),
                      relieved=summarize("ridge_relief", relieved))
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

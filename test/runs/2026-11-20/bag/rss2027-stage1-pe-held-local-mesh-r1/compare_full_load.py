"""Compare PE jaw-local refinement in the same complete YAM action."""

import argparse
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-kraft-load-ramp-clear-r1")))
from compare_load_ramps import compare, load, report  # noqa: E402


def read_settings(recording):
    result = load(recording / "trial/result.json")
    contract = result["config"].copy()
    for key in ("geometry", "initial_state", "wall_limit"):
        contract.pop(key)
    return contract, result["dimensions"]["film_vertices"]


def onset_measurements(case):
    event = case.get("first_sustained_bilateral_sliding")
    if event is None:
        return None
    names = ("applied_load_N", "actual_gap_m", "tension_N", "left_inner_N", "right_inner_N")
    measured = {key: float(event[key]) for key in names}
    measured["force_ratio"] = measured["tension_N"] / (measured["left_inner_N"] + measured["right_inner_N"])
    return measured


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    coarse = RUN.with_name("rss2027-stage1-material-load-ramps-r1") / "pe/recording"
    fine = RUN / "full-action/recording"
    reference, n_coarse = read_settings(coarse)
    same, n_fine = read_settings(fine)
    if reference != same or (n_coarse, n_fine) != (142, 189):
        raise ValueError("The physical input or intended mesh pair does not match")
    cases = {}
    for label, recording in (("coarse", coarse), ("fine", fine)):
        data, preload = report(recording)
        qualifying = [row for row in preload if row["bilateral_held"] == "True"]
        cases[label] = data | dict(qualified_preload_samples=len(qualifying), onset=onset_measurements(data))
    a, b = (cases[label]["onset"] for label in ("coarse", "fine"))
    onset = None
    if a is not None and b is not None:
        changes = {key: 100. * (b[key] - a[key]) / abs(a[key]) for key in ("applied_load_N", "actual_gap_m",
                                                                         "tension_N", "force_ratio")}
        onset = dict(observed_percent_change=changes, within_10pct=all(abs(x) <= 10. for x in changes.values()))
    summary = dict(mesh_vertices=[n_coarse, n_fine], cases=cases, onset=onset,
                   detachment=compare(cases["coarse"], cases["fine"]))
    with args.output.open("x") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

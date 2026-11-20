"""Evaluate the same unsaturated material-growth window on four whole bags."""

import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-top-ridge-selection-r1")))
from compare_ridge import fold_window, read_case  # noqa: E402


def measure(case):
    result = json.loads((case / "recording/trial/result.json").read_bytes())
    pressure = result["normal_press"]
    if result["warnings"] or result["peaks"]["penetration_mm"] > 1. or not pressure["target_reached"]:
        raise ValueError(f"Missing original pressure or numerical qualification in {case}")
    rows = read_case(case)
    if not rows:
        raise ValueError(f"Missing closure geometry in {case}")
    growth = max(row["extra_length_m"] - rows[0]["extra_length_m"] for row in rows)
    root = RUN.parents[4]
    return dict(case=case.relative_to(root).as_posix(), pressure_N=pressure["target_N"],
                closure_samples=len(rows), peak_growth_m=growth, window=fold_window(rows, pressure["band_N"]))


def main():
    pe = RUN.with_name("rss2027-stage1-randomized-top-pressure-sweep-r1")
    series = {"pe": (pe / "low/refined", RUN.with_name("rss2027-stage1-randomized-top-local-mesh-r1") / "refined",
                     pe / "high/refined")}
    for material in ("bubble", "kraft", "paperboard"):
        series[material] = (RUN / f"series/{material}-0p3N", RUN / material, RUN / f"series/{material}-0p8N")
    evidence = {material: [measure(case) for case in cases] for material, cases in series.items()}
    with (RUN / "gathering-windows.json").open("x") as stream:
        stream.write(json.dumps(evidence, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

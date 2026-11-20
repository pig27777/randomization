"""Compare whole-bag pressure targets using measured displacement and aperture material."""

import argparse
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-top-local-mesh-r1")))
from measure_surface_excess import read_case  # noqa: E402


def load(file):
    return json.loads(file.read_bytes())


def case_measurements(location):
    trial = location / "recording/trial"
    if not (trial / "result.json").is_file():
        check = next((location / phase / "summary.json" for phase in ("settle", "smoke")
                      if (location / phase / "summary.json").is_file()), None)
        if check is None:
            return dict(outcome="no_recorded_entry")
        result = load(check)
        return dict(outcome=result["outcome"], phase=result["phase"], peak_penetration_mm=result["peaks"]["penetration_mm"],
                    result=check.relative_to(RUN.parents[4]).as_posix())
    result = load(trial / "result.json")
    trace = load(trial / "trace.json")
    approach = [row for row in trace if row["phase"] == "approach"]
    timing = next((item for item in result["phases"] if item["phase"] == "approach"), None)
    summary = dict(outcome=result["outcome"], reason=result["reason"], normal_press=result["normal_press"],
                   peak_penetration_mm=result["peaks"]["penetration_mm"], warnings=result["warnings"],
                   approach=timing, tip_travel_before_closing_m=(approach[0]["tip_mid"][2]-approach[-1]["tip_mid"][2]
                                                               if approach else None))
    capture = location / "capture/capture-summary.json"
    if capture.exists():
        geometry = load(capture)
        rows = read_case(location)
        summary.update(first_lift=geometry["first_lift"],
                       peak_aperture_reference_length_m=geometry["peak_aperture_material_length_m"],
                       peak_aperture_excess_m=max((row["extra_length_m"] for row in rows), default=0.),
                       lift_peak_tension_N=geometry["lift_tension_peak_N"],
                       lift_min_support_N=geometry["lift_support_min_N"])
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = {}
    for material in ("bubble", "paperboard", "kraft"):
        options = {"0.3": RUN / "series" / f"{material}-0p3N", "0.5": RUN / material,
                   "0.8": RUN / "series" / f"{material}-0p8N"}
        original = load((RUN / material) / "recording/trial/result.json")["config"]
        for label, location in options.items():
            config = original if label == "0.5" else load(location / "config.json")
            if config != (original | {"press_target_N": float(label)}):
                raise ValueError(f"Only the requested pressure may differ for {material}")
        results[material] = {label: case_measurements(location) for label, location in options.items()}
    with args.output.open("x") as stream:
        stream.write(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

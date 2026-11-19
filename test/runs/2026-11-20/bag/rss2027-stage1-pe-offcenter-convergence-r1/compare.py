"""Compare the shared first 250 ms of the registered PE pull trials."""

import csv
import json
from pathlib import Path


def measure(case):
    config = json.loads((case / "config.json").read_text())
    summary = json.loads((case / "summary.json").read_text())
    with (case / "curve.csv").open() as stream:
        pull = [row for row in csv.DictReader(stream) if row["phase"] == "pull"]
    shape = config["initial_geometry"]
    settings = dict(step_s=config["dt"], grid_m=shape.get("aperture_pitch_m", shape["nominal_local_pitch_m"]),
                    grip_window_s=summary["longest_grip_hold_band_s"], total_pull_seconds=config["pull_seconds"])
    if not pull:
        return dict(**settings, valid_capture=False, capture_after_pull_s=None, captured_material_m=None,
                    peak_tension_N=None, window_final_tension_N=None, full_peak_tension_N=None,
                    release_after_pull_s=None, traction_slip_after_pull_s=None, traction_slip_force_N=None)
    start = float(pull[0]["t_s"])
    # Use the same 250 observations on the shared 1 ms time grid.
    end = start + .249 + 1e-8
    window = [row for row in pull if float(row["t_s"]) <= end]
    captured_at = summary["capture_t_s"]
    captured = (captured_at is not None and captured_at <= end and
                summary["longest_grip_hold_band_s"] >= .01 and summary["numerically_valid"])
    event = next((row for row in window if float(row["t_s"]) == captured_at), None)
    return dict(**settings, valid_capture=bool(captured),
                capture_after_pull_s=float(captured_at) - start if captured else None,
                captured_material_m=float(event["aperture_material_length_m"]) if captured else None,
                peak_tension_N=max(float(row["signed_pull_N"]) for row in window),
                window_final_tension_N=float(window[-1]["signed_pull_N"]),
                full_peak_tension_N=summary["peak_pull_N"],
                release_after_pull_s=(summary["release_t_s"] - start) if summary["release_t_s"] is not None else None,
                traction_slip_after_pull_s=(summary["traction_slip_candidate_t_s"] - start)
                if summary["traction_slip_candidate_t_s"] is not None else None,
                traction_slip_force_N=summary["traction_slip_candidate_force_N"])


def main():
    run = Path(__file__).parent
    original = run.parent / "rss2027-stage1-pe-offcenter-gather-r1/level-2"
    baseline = measure(original)
    assert baseline["valid_capture"]
    metrics = ("capture_after_pull_s", "captured_material_m", "peak_tension_N")
    full_metrics = ("full_peak_tension_N", "release_after_pull_s", "traction_slip_after_pull_s", "traction_slip_force_N")
    results = {}
    cases = {name: run / name for name in ("coarse-extended", "fine-extended", "half-step")}
    cases["refined-aperture"] = run.parent / "rss2027-stage1-pe-aperture-refinement-r1/refined-aperture"
    for name, case in cases.items():
        if not (case / "summary.json").exists():
            continue
        observation = measure(case)
        comparison = {key: (observation[key] / baseline[key] - 1.) * 100 for key in metrics} \
            if observation["valid_capture"] else None
        results[name] = dict(observation=observation, change_percent=comparison,
                             within_ten_percent=bool(comparison is not None and
                                                     max(map(abs, comparison.values())) <= 10.))
    full_comparisons = {}
    if "fine-extended" in results:
        reference = results["fine-extended"]["observation"]
        for name, entry in results.items():
            case = entry["observation"]
            if name != "fine-extended" and case["total_pull_seconds"] == reference["total_pull_seconds"]:
                full_comparisons[name] = {
                    key: dict(change_percent=(case[key] / reference[key] - 1.) * 100,
                              within_ten_percent=abs(case[key] / reference[key] - 1.) <= .10)
                    if case[key] is not None and reference[key] not in (None, 0.) else None
                    for key in full_metrics
                }
    print(json.dumps(dict(baseline=baseline, results=results, full_comparisons_vs_fine=full_comparisons), indent=2))


if __name__ == "__main__":
    main()

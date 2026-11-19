"""Compare the shared first 250 ms of the registered PE pull trials."""

import csv
import json
from pathlib import Path


def measure(case):
    config = json.loads((case / "config.json").read_text())
    summary = json.loads((case / "summary.json").read_text())
    with (case / "curve.csv").open() as stream:
        pull = [row for row in csv.DictReader(stream) if row["phase"] == "pull"]
    start = float(pull[0]["t_s"])
    # Use the same 250 observations on the shared 1 ms time grid.
    end = start + .249 + 1e-8
    window = [row for row in pull if float(row["t_s"]) <= end]
    captured_at = summary["capture_t_s"]
    captured = (captured_at is not None and captured_at <= end and
                summary["longest_grip_hold_band_s"] >= .01 and summary["numerically_valid"])
    event = next((row for row in window if float(row["t_s"]) == captured_at), None)
    return dict(step_s=config["dt"], grid_m=config["initial_geometry"]["nominal_local_pitch_m"],
                valid_capture=bool(captured), grip_window_s=summary["longest_grip_hold_band_s"],
                capture_after_pull_s=float(captured_at) - start if captured else None,
                captured_material_m=float(event["aperture_material_length_m"]) if captured else None,
                peak_tension_N=max(float(row["signed_pull_N"]) for row in window),
                window_final_tension_N=float(window[-1]["signed_pull_N"]),
                release_after_pull_s=(summary["release_t_s"] - start) if summary["release_t_s"] is not None else None,
                traction_slip=(summary["traction_slip_candidate_t_s"] - start)
                if summary["traction_slip_candidate_t_s"] is not None else None)


def main():
    run = Path(__file__).parent
    original = run.parent / "rss2027-stage1-pe-offcenter-gather-r1/level-2"
    baseline = measure(original)
    assert baseline["valid_capture"]
    metrics = ("capture_after_pull_s", "captured_material_m", "peak_tension_N")
    results = {}
    for name in ("coarse-extended", "fine-extended", "half-step"):
        case = run / name
        if not (case / "summary.json").exists():
            continue
        observation = measure(case)
        comparison = {key: (observation[key] / baseline[key] - 1.) * 100 for key in metrics} \
            if observation["valid_capture"] else None
        results[name] = dict(observation=observation, change_percent=comparison,
                             within_ten_percent=bool(comparison is not None and
                                                     max(map(abs, comparison.values())) <= 10.))
    print(json.dumps(dict(baseline=baseline, results=results), indent=2))


if __name__ == "__main__":
    main()

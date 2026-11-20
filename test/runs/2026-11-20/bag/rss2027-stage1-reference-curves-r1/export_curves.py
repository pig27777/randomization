"""Export the archived Stage 1 whole-bag force and displacement samples in SI units."""

import argparse
import csv
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[4]
RUNS = ROOT / "test/runs"
AXIS_AUDIT = RUN.with_name("rss2027-stage1-aperture-axis-r1")
CASES = (
    ("pe-empty-top", "pe", "2026-11-19/bag/rss2027-stage1-empty-yam-r4/recording/trial"),
    ("pe-compact-top", "pe", "2026-11-19/bag/rss2027-stage1-content-pair-r1/compact-offset/trial"),
    ("pe-broad-top", "pe", "2026-11-19/bag/rss2027-stage1-content-pair-r1/broad-offset/trial"),
    ("pe-matched-side", "pe", "2026-11-19/bag/rss2027-stage1-side-grasp-matched-pe-fine-step-r1/recording/trial"),
    ("pe-like-side-release", "pe_like", "2026-11-20/bag/rss2027-stage1-whole-bag-release-r3/recording/trial"),
    ("paperboard-side-release", "paperboard", "2026-11-20/bag/rss2027-stage1-paperboard-release-r1/recording/trial"),
    ("kraft-side-release", "kraft", "2026-11-20/bag/rss2027-stage1-kraft-release-r2/recording/trial"),
    ("bubble-supported-release", "bubble", "2026-11-20/bag/rss2027-stage1-bubble-short-hover-r1/recording/trial"),
    ("pe-randomized-004", "pe", "2026-11-20/bag/rss2027-stage1-randomized-top-relaxed-r1/sample-004/recording/trial"),
    ("pe-randomized-127", "pe", "2026-11-20/bag/rss2027-stage1-randomized-top-r1/sample-127/recording/trial"),
    ("pe-randomized-002", "pe", "2026-11-20/bag/rss2027-stage1-randomized-top-r1/sample-002/recording/trial"),
    ("bubble-unsupported-release", "bubble", "2026-11-20/bag/rss2027-stage1-bubble-free-release-floor-r1/recording/trial"),
    ("pe-coarse-aperture", "pe", "2026-11-20/bag/rss2027-stage1-randomized-top-local-mesh-r1/coarse/recording/trial"),
    ("pe-refined-aperture", "pe", "2026-11-20/bag/rss2027-stage1-randomized-top-local-mesh-r1/refined/recording/trial"),
    ("pe-refined-relief", "pe", "2026-11-20/bag/rss2027-stage1-randomized-top-relief-r1/recording/trial"),
    ("pe-top-0p3N", "pe", "2026-11-20/bag/rss2027-stage1-randomized-top-pressure-sweep-r1/low/refined/recording/trial"),
    ("pe-top-0p8N", "pe", "2026-11-20/bag/rss2027-stage1-randomized-top-pressure-sweep-r1/high/refined/recording/trial"),
    ("pe-top-ridge", "pe", "2026-11-20/bag/rss2027-stage1-top-ridge-selection-r1/long-approach/recording/trial"),
    ("pe-top-ridge-relief", "pe",
     "2026-11-20/bag/rss2027-stage1-top-ridge-selection-r1/closure-relief/recording/trial"),
    ("pe-top-ridge-earlier-lift", "pe",
     "2026-11-20/bag/rss2027-stage1-top-ridge-selection-r1/earlier-lift/recording/trial"),
    ("pe-top-ridge-faster-approach", "pe",
     "2026-11-20/bag/rss2027-stage1-top-ridge-selection-r1/faster-approach/recording/trial"),
    ("pe-top-anchored-sweep", "pe",
     "2026-11-20/bag/rss2027-stage1-top-ridge-selection-r1/anchored-sweep/recording/trial"),
    ("pe-top-seated-close", "pe",
     "2026-11-20/bag/rss2027-stage1-top-ridge-selection-r1/seated-close/recording/trial"),
)
COLUMNS = (
    "t_s", "phase", "downforce_N", "direct_finger_support_N", "actual_gap_m", "commanded_gap_m",
    "closure_displacement_m", "aperture_reference_length_m", "lift_displacement_m", "payload_com_z_m",
    "support_N", "signed_pull_N", "grip_each_N", "force_on_fingers_world_N", "left_inner_candidates",
    "right_inner_candidates", "native_bilateral_sliding", "penetration_mm",
)
RAMP_CASES = (
    ("kraft-fixed-fast", "kraft", "rss2027-stage1-kraft-load-ramp-r1"),
    ("kraft-fixed-slow", "kraft", "rss2027-stage1-kraft-load-ramp-slow-r1"),
    ("kraft-clear-fast", "kraft", "rss2027-stage1-kraft-load-ramp-clear-fast-r1"),
    ("kraft-clear-slow", "kraft", "rss2027-stage1-kraft-load-ramp-clear-r1"),
    ("kraft-clear-half-step", "kraft", "rss2027-stage1-kraft-load-ramp-half-step-r1"),
    ("pe-matched-load", "pe", "rss2027-stage1-material-load-ramps-r1/pe"),
    ("paperboard-legacy-edges", "paperboard", "rss2027-stage1-material-load-ramps-r1/paperboard"),
    ("paperboard-rounded-pad", "paperboard", "rss2027-stage1-material-load-ramps-r1/paperboard-rounded-pad"),
    ("bubble-127mm", "bubble", "rss2027-stage1-material-load-ramps-r1/bubble"),
    ("bubble-130mm", "bubble", "rss2027-stage1-material-load-ramps-r1/bubble-130mm"),
    ("bubble-early-withdrawal", "bubble", "rss2027-stage1-material-load-ramps-r1/bubble-early-withdrawal"),
)
RAMP_COLUMNS = (
    "t_s", "applied_load_N", "actual_gap_m", "tension_N", "support_N", "tip_z_m", "payload_com_z_m",
    "left_inner_N", "right_inner_N", "left_sliding_N", "right_sliding_N", "max_finger_contact_N",
)


def load_csv(path):
    with path.open(newline="") as stream:
        return {float(row["t_s"]): row for row in csv.DictReader(stream)}


def load_json(path):
    return json.loads(path.read_bytes())


def export_case(label, material, path, axis_index):
    trial = RUNS / path
    key = trial.relative_to(ROOT).as_posix()
    archive = trial.parent.parent if trial.parent.name == "recording" else trial.parent
    corrected = axis_index.get(key)
    resolved = AXIS_AUDIT / corrected["corrected_directory"] if corrected else archive
    aperture_path = resolved / "capture/capture-audit.csv"
    aperture = load_csv(aperture_path)
    traction_path = archive / "release-traction.csv"
    traction = load_csv(traction_path) if traction_path.exists() else {}
    release_path = resolved / "release-check.json"
    release = load_json(release_path) if release_path.exists() else None
    trace = load_json(trial / "trace.json")
    result = load_json(trial / "result.json")
    times = [row["t"] for row in trace]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError(f"Nonincreasing trace time for {label}")
    if (set(aperture) | set(traction)) - set(times):
        raise ValueError(f"Unmatched geometry or traction samples for {label}")

    samples = []
    close_origin = next((row["gap"] for row in trace if row["phase"] == "close"), None)
    lift_origin = next((row["tip_mid"][2] for row in trace if row["phase"] == "lift"), None)
    lift_phases = ("lift", "lift_hold", "open")
    for row in trace:
        time, phase = row["t"], row["phase"]
        a = aperture.get(time)
        grip = row["grip_each_N"]
        tension = -sum(force[2] for force in row["force_on_fingers_N"])
        if a and (float(a["actual_gap_m"]) != row["gap"] or float(a["left_grip_N"]) != grip[0]
                  or float(a["right_grip_N"]) != grip[1] or float(a["support_N"]) != row["support_N"]
                  or abs(float(a["tension_N"]) - tension) > 1e-12):
            raise ValueError(f"Recorded reactions or gap changed for {label} at {time}")
        native = traction.get(time)
        close_s = close_origin - row["gap"] if phase == "close" else None
        lift_s = row["tip_mid"][2] - lift_origin if phase in lift_phases and lift_origin is not None else None
        samples.append([time, phase, row.get("down_N"), row.get("direct_support_N"), row["gap"],
                        row["command_gap"], close_s, float(a["aperture_material_length_m"]) if a else None,
                        lift_s, row["com"][2] if result["config"]["mode"] != "empty" else None,
                        row["support_N"], tension, grip, row["force_on_fingers_N"],
                        int(a["left_inner_candidates"]) if a else None, int(a["right_inner_candidates"]) if a else None,
                        native["bilateral_sliding"] == "True" if native else None, row["penetration_mm"]])

    if release and release["first_final_contact_loss_sample"] is not None:
        release_time = float(release["first_final_contact_loss_sample"]["t_s"])
        if release_time not in aperture:
            raise ValueError(f"Final contact loss has no matching sample for {label}")
        hold = [row["support_N"] for row in trace if row["phase"] == "open" and row["t"] <= release_time]
        release["peak_support_through_detachment_N"] = max(hold)
    meta = dict(id=label, material=material, trial=key, aperture_source=aperture_path.relative_to(ROOT).as_posix(),
                contact_source=traction_path.relative_to(ROOT).as_posix() if traction else None,
                outcome=result["outcome"], reason=result["reason"], warnings=result["warnings"],
                normal_press=result.get("normal_press"), release=release, config=result["config"],
                samples=samples)
    relief = trial.parent / "closure-relief.json"
    if relief.exists():
        meta["closure_pressure_relief"] = load_json(relief)
    print(label, len(samples), len(aperture), len(traction), result["outcome"])
    return meta


def export_ramp(label, material, path):
    recording = RUN.parent / path / "recording"
    raw = load_csv(recording / "load-curve.csv")
    check = load_json(recording / "load-check.json")
    result = load_json(recording / "trial/result.json")
    if len(raw) != check["samples"]:
        raise ValueError(f"Ramp sample count disagrees for {label}")
    start = check["load_start_s"]
    origin = next((float(row["payload_com_z_m"]) for time, row in raw.items() if start is not None and time >= start), None)
    previous = None
    samples = []
    for time, row in raw.items():
        if previous is not None and abs(time - previous - .002) > result["config"]["dt"] / 2:
            raise ValueError(f"Incorrect time interval in {label}")
        expected = min(check["ramp_limit_N"], max(0., (time - start) * check["ramp_rate_N_s"])) if start is not None else 0.
        if abs(float(row["applied_load_N"]) - expected) > 1e-10:
            raise ValueError(f"Incorrect load at {time} in {label}")
        displacement = origin - float(row["payload_com_z_m"]) if origin is not None and time >= start else None
        samples.append([float(row[k]) for k in RAMP_COLUMNS] + [row["native_bilateral_sliding"] == "True",
                         row["bilateral_held"] == "True", displacement])
        previous = time
    return dict(id=label, material=material, trial=(recording / "trial").relative_to(ROOT).as_posix(),
                config=result["config"], outcome=result["outcome"], reason=result["reason"], warnings=result["warnings"],
                peak_penetration_mm=result["peaks"]["penetration_mm"], measurement=check, samples=samples)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=RUN / "stage1_whole_bag_data.json")
    args = parser.parse_args()
    axis_index = load_json(AXIS_AUDIT / "index.json")
    cases = [export_case(*case, axis_index) for case in CASES]
    data = dict(stage=1, complete=False, sample_columns=COLUMNS, cases=cases, notes={
        "aperture_reference_length_m": "Total clipped reference area divided by finger width; sums all layers.",
        "signed_pull_N": "Upward film load from the negated vertical component of both finger reactions.",
        "native_bilateral_sliding": "Per-sample contact criterion; a sustained event is required for slip fitting.",
        "scope": "Archived whole-bag measurements. Material slip displacement and an all-material fitted envelope are pending.",
    })
    data["load_ramps"] = dict(
        sample_columns=RAMP_COLUMNS + ("native_bilateral_sliding", "bilateral_held", "payload_load_displacement_m"),
        cases=[export_ramp(*case) for case in RAMP_CASES],
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()

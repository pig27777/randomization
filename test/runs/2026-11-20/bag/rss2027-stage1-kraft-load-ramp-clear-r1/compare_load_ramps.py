"""Audit force-ramp detachment, support and numerical convergence."""

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parents[3] / "bag_model"))
from physical_parcel_checks import film_payload_geometry  # noqa: E402

CASES = {
    "fixed_fast": "rss2027-stage1-kraft-load-ramp-r1",
    "fixed_slow": "rss2027-stage1-kraft-load-ramp-slow-r1",
    "clear_fast": "rss2027-stage1-kraft-load-ramp-clear-fast-r1",
    "clear_slow": "rss2027-stage1-kraft-load-ramp-clear-r1",
    "clear_half_step": "rss2027-stage1-kraft-load-ramp-half-step-r1",
}


def load(path):
    return json.loads(path.read_bytes())


def final_replay(recording):
    event = load(recording / "withdrawal.json")
    trial = recording / "trial"
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    data = mujoco.MjData(model)
    with np.load(trial / "trajectory.npz") as state:
        spec = int(state["integration_state_spec"])
        mujoco.mj_setState(model, data, state["final_integration_state"], spec)
    if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
        raise ValueError(f"Nonfinite recorded geometry for {recording}")
    mujoco.mj_fwdPosition(model, data)
    heights = [float(data.geom_xpos[model.geom(name).id, 2]) for name in ("table", "floor")]
    np.testing.assert_allclose(heights, [point[2] for point in event["after_m"]], atol=1e-12, rtol=0.)
    geom = film_payload_geometry(model, data, model.flex_elem.reshape(-1, 3))
    return dict(support_z_m=heights, floor_clearance_at_withdrawal_m=event["floor_clearance_m"],
                outside_corners=geom["outside_corners"], intersecting_triangles=geom["intersecting_triangles"])


def report(recording):
    result = load(recording / "trial/result.json")
    check = load(recording / "load-check.json")
    metadata = dict(dt_s=result["config"]["dt"], rate_N_s=check["ramp_rate_N_s"],
                    numerical_ok=check["numerical_ok"], peak_penetration_mm=result["peaks"]["penetration_mm"],
                    peak_tension_N=check["max_observed_tension_N"], outcome=result["outcome"], reason=result["reason"])
    if check["load_start_s"] is None:
        return metadata | dict(samples=check["samples"], ramp_started=False, unsupported_loss=False), []
    with (recording / "load-curve.csv").open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    times = np.array([float(row["t_s"]) for row in raw])
    np.testing.assert_allclose(np.diff(times), .002, atol=result["config"]["dt"] / 2, rtol=0.)
    measured = np.array([float(row["applied_load_N"]) for row in raw])
    expected = np.clip((times - check["load_start_s"]) * check["ramp_rate_N_s"], 0., check["ramp_limit_N"])
    np.testing.assert_allclose(measured, expected, atol=1e-10, rtol=0.)

    def sliding(row):
        return row["native_bilateral_sliding"] == "True"

    longest = 0.
    for active, sequence in itertools.groupby(raw, key=sliding):
        if active:
            trace = list(sequence)
            longest = max(longest, float(trace[-1]["t_s"]) - float(trace[0]["t_s"]))
    contacts = np.flatnonzero([float(row["max_finger_contact_N"]) > .05 for row in raw])
    after = int(contacts[-1] + 1) if len(contacts) else 0
    loss = raw[after] if 0 < after < len(raw) and "lost >50ms" in result["reason"] else None
    max_support = max((float(row["support_N"]) for row in raw[:after + 1]), default=None) if loss else None
    supports = final_replay(recording) if (recording / "withdrawal.json").exists() else None
    enclosed = supports is None or not (supports["outside_corners"] or supports["intersecting_triangles"])
    prefix = [row for row in raw if float(row["t_s"]) < check["load_start_s"]]
    return metadata | dict(samples=len(raw), ramp_started=True, withdrawal=supports,
                both_sliding_samples=sum(sliding(row) for row in raw), continuous_bilateral_s=longest,
                first_sustained_bilateral_sliding=check["first_sustained_bilateral_sliding"],
                final_loss_sample=loss, peak_support_through_loss_N=max_support,
                unsupported_loss=bool(loss and max_support <= .05 and check["numerical_ok"] and enclosed)), prefix


def compare(first, second):
    a, b = first.get("final_loss_sample"), second.get("final_loss_sample")
    if not a or not b:
        return dict(comparable=False)
    differences = {key: 100. * (float(b[key]) - float(a[key])) / abs(float(a[key]))
                   for key in ("applied_load_N", "actual_gap_m")}
    comparable = first["unsupported_loss"] and second["unsupported_loss"]
    return dict(comparable=comparable, observed_percent_change=differences,
                within_10pct=all(abs(value) <= 10. for value in differences.values()) if comparable else None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=RUN / "load-ramp-comparison.json")
    args = parser.parse_args()
    cases, prefixes = {}, {}
    for label, location in CASES.items():
        cases[label], prefixes[label] = report(RUN.with_name(location) / "recording")
    assert prefixes["fixed_fast"] == prefixes["fixed_slow"]
    assert prefixes["clear_fast"] == prefixes["clear_slow"]
    result = dict(cases=cases, clear_matched_preload_samples=len(prefixes["clear_fast"]),
                  clear_rate_comparison=compare(cases["clear_fast"], cases["clear_slow"]),
                  clear_step_comparison=compare(cases["clear_slow"], cases["clear_half_step"]))
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

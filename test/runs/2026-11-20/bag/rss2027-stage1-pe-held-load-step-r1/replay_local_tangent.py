"""Retest a measured withdrawal window from the same mid-slide integration state."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from measure_aperture_flow import RUN, record_flow, replay, summarize
from measure_tangent_flow import estimate_beta

COARSE = RUN / "tangent-prefix/dt-25us"
START_S = 1.142
END_S = 1.200
DT = .000025


def extract(archive, native):
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    step = round(START_S / DT)
    recorded = None

    def sample(robot, i, _row):
        nonlocal recorded
        if i != step - 1:
            return
        model, data = robot.sim.m, robot.sim.d
        recorded = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, recorded, spec)
        np.testing.assert_allclose(data.time - native["source_time_s"], START_S, atol=1e-9, rtol=0.)

    result, rows = replay.run(DT, START_S, "jaw-tangent", readout=sample)
    with np.load(COARSE / "native-load.npz") as original:
        np.testing.assert_array_equal(rows, original["rows"][:step])
    if recorded is None or result["outcome"] != "no_loss_before_limit":
        raise ValueError("The archive did not reach the designated fully enclosed state")
    np.savez_compressed(archive, integration_state=recorded, spec=spec, matched_prefix_steps=len(rows))
    return recorded, spec, len(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    replay.SOURCE = RUN.with_name("rss2027-stage1-material-load-ramps-r1") / "pe/recording"
    replay.CHECKPOINT = RUN / "held-state.npz"
    reference = json.loads((COARSE / "summary.json").read_bytes())
    native = reference["native_load"]
    original_scene = replay.held_scene
    state, spec, count = extract(args.output / "mid-slide-state.npz", native)
    digest = hashlib.sha256(state.tobytes()).hexdigest()
    direction = np.asarray(native["force_direction"])
    ratio = reference["material_flow"]["candidate_force_ratio"]
    results = []

    def mid_slide_scene():
        robot, _i, _fingerprint = original_scene()
        mujoco.mj_setState(robot.sim.m, robot.sim.d, state, spec)
        return robot, count, digest

    for label, dt in (("25", DT), ("12p5", DT / 2.)):
        with patch.object(replay, "held_scene", mid_slide_scene):
            load, rows, transitions = record_flow(dt, END_S - START_S, "jaw-tangent", force_vector=direction,
                                                  initial_load_N=START_S * native["rate_N_s"])
        flow = summarize(load, rows, transitions, force_ratio=ratio)
        beta = estimate_beta(load, flow)
        report = dict(native_load=load, material_flow=flow, beta_candidate=beta)
        with (args.output / f"dt-{label}us.json").open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(args.output / f"dt-{label}us.npz", columns=np.asarray(replay.FIELDS), rows=rows)
        results.append(dict(dt_s=dt, peak_penetration_mm=load["peak_penetration_mm"],
                            outcome=load["outcome"], geometry=load["end_health"]["payload_geometry"],
                            longest_positive_bilateral_s=flow["longest_bilateral_positive_overload_s"],
                            net_withdrawal_during_slip_m=flow["signed_withdrawal_during_bilateral_slip_m"],
                            fitted=beta))
        print(label, load["outcome"], load["peak_penetration_mm"], beta, flush=True)
    relative = None
    if all(row["fitted"] for row in results):
        coarse, fine = [row["fitted"]["beta_m_per_N_s"] for row in results]
        relative = fine / coarse - 1.
    report = dict(original_start_s=START_S, original_end_s=END_S, prefix_steps=count,
                  native_state_sha256=digest, common_onset_force_ratio=ratio, trials=results,
                  fitted_beta_relative_change=relative, beta_within_10pct=relative is not None and abs(relative) <= .10)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

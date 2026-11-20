"""Verify the common PE tangent-load input and compare both time resolutions."""

import json
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent
PREFIX = RUN / "tangent-prefix"


def read_trial(label):
    case = PREFIX / label
    summary = json.loads((case / "summary.json").read_bytes())
    load, flow = summary["native_load"], summary["material_flow"]
    if load["supported_force_peak_N"] or load["outcome"] != "no_loss_before_limit":
        raise ValueError(f"{label} did not complete the common unsupported horizon")
    geometry = load["end_health"]["payload_geometry"]
    if load["peak_penetration_mm"] > 1. or geometry["outside_corners"] or geometry["intersecting_triangles"]:
        raise ValueError(f"{label} did not preserve the numerical and geometric conditions")
    with np.load(case / "native-load.npz") as archive:
        columns = list(archive["columns"])
        samples = archive["rows"]
    np.testing.assert_allclose(samples[-1, 0], 1.29, atol=1e-9, rtol=0.)
    start = load["sliding_onset"]
    projected = abs(samples[1:, columns.index("added_along_closing_axis_N")] / samples[1:, columns.index("added_N")])
    metrics = dict(onset_s=start["time_s"], onset_added_N=start["added_N"], onset_tension_N=start["tension_N"],
                   onset_gap_m=start["actual_gap_m"], onset_force_ratio=flow["candidate_force_ratio"],
                   continuous_sliding_s=load["longest_bilateral_slide_s"],
                   sliding_net_withdrawal_m=flow["signed_withdrawal_during_bilateral_slip_m"],
                   beta_m_per_N_s=summary["beta_candidate"]["beta_m_per_N_s"])
    return load, samples, columns, dict(metrics=metrics, beta_window=summary["beta_candidate"],
                                        native_peak_penetration_mm=load["peak_penetration_mm"],
                                        peak_axial_load_fraction=float(projected.max()), samples=len(samples))


def main():
    coarse, coarse_rows, columns, coarse_summary = read_trial("dt-25us")
    fine, fine_rows, fine_columns, fine_summary = read_trial("dt-12p5us")
    if columns != fine_columns:
        raise ValueError("Native readout columns differ")
    for key in ("source_frame", "source_time_s", "source_state_sha256", "force_direction", "rate_N_s", "actuator_input"):
        if coarse[key] != fine[key]:
            raise ValueError(f"The requested common load changed {key}")
    with np.load(RUN / "tangent-flow/dt-25us/native-load.npz") as long:
        np.testing.assert_array_equal(long["rows"][:len(coarse_rows)], coarse_rows)
    np.testing.assert_allclose(coarse_rows[:, :2], fine_rows[::2, :2], atol=1e-9, rtol=0.)
    drift = abs(coarse_rows[:, columns.index("actual_gap_m")] - fine_rows[::2, columns.index("actual_gap_m")])
    changed = np.flatnonzero(drift > .00025)
    relative = {key: fine_summary["metrics"][key] / value - 1.
                for key, value in coarse_summary["metrics"].items()}
    report = dict(checkpoint_sha256=coarse["source_state_sha256"], verified_25us_prefix=len(coarse_rows),
                  simulated_duration_s=1.29, coarse=coarse_summary, half_step=fine_summary,
                  relative_change_at_half_step=relative, limit_fraction=.10,
                  first_encoder_gap_difference_over_0p25mm_s=float(coarse_rows[changed[0], 0]) if len(changed) else None,
                  accepted=all(abs(value) <= .10 for value in relative.values()))
    with (PREFIX / "comparison.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

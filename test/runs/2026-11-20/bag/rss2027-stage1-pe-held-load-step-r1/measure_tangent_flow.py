"""Apply a tangential load and measure net membrane withdrawal in the YAM jaws."""

import argparse
import json
from pathlib import Path

import numpy as np

from measure_aperture_flow import RUN, record_flow, replay, summarize


def estimate_beta(result, flow):
    if result["outcome"] not in ("detached", "no_loss_before_limit") or result["supported_force_peak_N"] > .05:
        return None
    samples = flow["transitions"]
    valid = [row.get("overload_integral_Ns", 0.) > 0. and row["net_material_withdrawal_m"] > 0. for row in samples]
    ranges = np.flatnonzero(np.diff(np.r_[False, valid, False])).reshape(-1, 2)
    windows = [samples[a:b] for a, b in ranges if samples[b - 1]["to_t_s"] - samples[a]["from_t_s"] >= .010 - 1e-9]
    if not windows:
        return None
    window = max(windows, key=lambda rows: rows[-1]["to_t_s"] - rows[0]["from_t_s"])
    overload = np.array([row["overload_integral_Ns"] for row in window])
    displacement = np.array([row["net_material_withdrawal_m"] for row in window])
    coefficient = float(overload @ displacement / (overload @ overload))
    return dict(beta_m_per_N_s=coefficient, from_s=window[0]["from_t_s"], to_s=window[-1]["to_t_s"],
                displacement_m=float(displacement.sum()), residual_rms_m=float(np.sqrt(np.mean(
                    (displacement - coefficient * overload)**2))))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--step-us", type=float, choices=(25., 12.5), default=25.)
    args = parser.parse_args()
    replay.SOURCE = RUN.with_name("rss2027-stage1-material-load-ramps-r1") / "pe/recording"
    replay.CHECKPOINT = RUN / "held-state.npz"
    args.output.mkdir(exist_ok=False)
    dt = args.step_us * 1e-6
    result, rows, samples = record_flow(dt, .02 if args.smoke else 2.4, "jaw-tangent")
    readout = summarize(result, rows, samples)
    estimate = estimate_beta(result, readout)
    report = dict(native_load=result, beta_candidate=estimate, material_flow=readout)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "native-load.npz", columns=np.asarray(replay.FIELDS), rows=rows)
    print(result["outcome"], result["peak_penetration_mm"], result["longest_bilateral_slide_s"], estimate, flush=True)


if __name__ == "__main__":
    main()

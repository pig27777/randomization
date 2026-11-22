"""Measure in-air sway in the archived C3/C6 Policy 1 replay."""

import argparse
import json
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / "configs/stage3_dr_and_corners.yaml").read_text())
    payload = json.loads((root / "configs/payload_template_X.yaml").read_text())
    metadata = json.loads((args.replay / "summary.json").read_text())
    with np.load(args.replay / "trajectory.npz") as recorded:
        trace, stages = recorded["trace"], recorded["stage"]
    if not np.isfinite(trace).all() or metadata["selected_success"] != [True, False]:
        raise ValueError("Expected a finite replay with C3 success and C6 failure")
    fps = metadata["fps"]
    time = np.arange(len(trace)) / fps
    window = time >= time[-1] - 3.
    assert np.all(stages[window] == 3)
    report, series = {}, {}
    for i, name in enumerate(("C3", "C6")):
        physical = cfg["corners"][name]
        settings = payload["defaults"] | payload["presets"][physical["content"]] | physical.get("payload", {})
        weights = np.array(settings["mass_weights"])
        xyz = (trace[:, i, :12].reshape(-1, 4, 3) * (weights / weights.sum())[None, :, None]).sum(1)
        jaw = trace[:, i, 20:26].reshape(-1, 2, 3).mean(1)
        speed = 1000. * fps * np.linalg.norm(np.diff(xyz[window], axis=0), axis=1)
        quiet = speed < 20.
        boundaries = np.flatnonzero(np.diff(np.r_[False, quiet, False]))
        longest = (boundaries[1::2] - boundaries[::2]).max(initial=0) / fps
        report[name] = dict(
            mass_kg=settings["total_mass"], window_s=[float(time[window][0]), float(time[-1])],
            clearance_range_mm=(1000. * np.array([trace[window,i,30].min(), trace[window,i,30].max()])).tolist(),
            parcel_span_xyz_mm=(1000. * np.ptp(xyz[window], axis=0)).tolist(),
            jaw_span_xyz_mm=(1000. * np.ptp(jaw[window], axis=0)).tolist(),
            speed_p50_p95_mm_s=np.quantile(speed, [.50, .95]).tolist(),
            bilateral_frames=int((trace[window, i, 31:33] > .5).all(1).sum()),
            samples=int(window.sum()), velocity_intervals=len(speed),
            above_speed_gate=int((~quiet).sum()), longest_sampled_quiet_s=longest,
            phase_b_max_clearance_mm=float(trace[stages == 2, i, 30].max() * 1000.),
        )
        centered = lambda v: 1000. * (v[window, 0] - v[window, 0].mean())
        series[name] = centered(xyz), centered(jaw), speed
    out = dict(checkpoint_sha256=metadata["checkpoint_sha256"], source_rows=metadata["rows"],
               fps=fps, corners=report)
    (args.output / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")

    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size=12)
    c6, jaw, speed = series["C6"]
    t = time[window]
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5.7), sharex=True, layout="constrained")
    fig.suptitle("C6 离台后相对夹爪摆动", fontproperties=font, fontsize=16)
    a.plot(t,c6,color="#dc7525",lw=2,label="C6 包裹质心")
    a.plot(t,jaw,color="#245aa0",lw=2,label="C6 夹爪中点")
    a.set_ylabel("横向位置相对均值 (mm)", fontproperties=font)
    a.legend(loc="upper right",prop=font,frameon=False,ncol=2)
    b.plot(.5*(t[1:]+t[:-1]),speed,color="#dc7525",lw=2,label="C6 质心速度")
    b.plot(.5*(t[1:]+t[:-1]),series["C3"][-1],color="#23906f",lw=1.5,label="C3 参考")
    b.axhline(20,color="#34383f",ls="--",lw=1,label="稳悬速度门槛")
    b.set_ylabel("区间平均速度 (mm/s)", fontproperties=font)
    b.set_xlabel("仿真时间 (s)", fontproperties=font)
    b.legend(loc="upper right",prop=font,frameon=False,ncol=3)
    b.set_ylim(bottom=0)
    b.xaxis.set_major_locator(MultipleLocator(.5))
    for axis in (a,b):
        axis.grid(alpha=.22)
        axis.spines[["top","right"]].set_visible(False)
    fig.savefig(args.output / "c6-sway.png", dpi=155)


if __name__ == "__main__":
    main()

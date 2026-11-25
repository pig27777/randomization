"""Plot acquired native forces for the fixture and Yam_B probes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
COLORS = {"left": "#376e8d", "center": "#197561", "right": "#ad5b3b"}


def format_axis(axis, xlabel, ylabel):
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=.18)
    axis.legend(frameon=False, fontsize=8)


def fixture_plot():
    sources = [p for p in sorted(ROOT.glob("a*/trace.npz")) if p.is_file()]
    film_trace = ROOT / "film-fixture/trace.npz"
    if film_trace.is_file():
        sources.append(film_trace)
    if not sources:
        return
    fig, axes = plt.subplots(len(sources), figsize=(8.2, 3.*len(sources)), squeeze=False, layout="constrained")
    for axis, p in zip(axes.flat, sources, strict=True):
        s = json.loads(p.with_name("summary.json").read_text())
        with np.load(p) as d:
            axis.plot(d["time"], d["jaw_normal"][:,0], color="#39698a", label="Left pad")
            axis.plot(d["time"], d["jaw_normal"][:,1], color="#975d38", label="Right pad")
            axis.plot(d["time"], np.max(d["peak_substep_N"],axis=1), color=".32", alpha=.65, label="Native block peak")
        axis.axhline(25., color="#a54d40", ls="--", lw=1., label="25 N hard limit")
        axis.axvspan(*s["hold_window_s"], facecolor="#478981", alpha=.10)
        title = "A2 film fixture, hardened pad contact" if p == film_trace else p.parent.name.upper()
        axis.set_title(f"{title} · {s['completed_s']:.3f} s recorded")
        axis.set_xlim(left=0, right=s["expected_s"])
        format_axis(axis, "Fixture simulation time (s)", "Pad normal force (N)")
    fig.savefig(ROOT/"grasp_a2_a3_forces.png", dpi=250)
    plt.close(fig)


def probe_plot():
    sources = [(case,p) for case in COLORS for p in sorted(ROOT.glob(f"*/{case}/summary.json"))]
    if not sources:
        return
    fig, (axis, release_axis) = plt.subplots(2, figsize=(8.2, 6.5), layout="constrained")
    comparison = {"first-left": ("Left, first profile", ".56", ":"),
                  "harder-hold-only": ("Left, second profile, hold only", ".38", "--")}
    for case, p in sources:
        s = json.loads(p.read_text())
        rows = [row for row in s["rows"] if row["block"]=="probe"]
        if not rows:
            continue
        times = np.array([r["t_s"] for r in rows])
        run = p.parent.parent.name
        label, color, line = comparison.get(run, (case.capitalize(), COLORS[case], "-"))
        axis.plot(times-times[0], [r["arm_peak_N"] for r in rows], color=color, ls=line, label=label)
        if case == "right" and s["gate_pass"]:
            forces = np.asarray([r["peak_pad_N"] for r in rows])
            release_axis.plot(times-times[0], forces[:,0], color=COLORS["left"], label="Left pad")
            release_axis.plot(times-times[0], forces[:,1], color=COLORS["right"], label="Right pad")
            trigger = s["native_soft_cutoff"]["time_s"] - times[0]
            release_axis.axvline(trigger, color=".36", ls="--", lw=1., label="Soft cutoff")
    axis.axhline(2., color=".36", ls="--", lw=1., label="2 N total-arm soft cutoff")
    axis.axhline(3.5, color="#a54d40", ls=":", lw=1., label="3.5 N total-arm hard limit")
    axis.set_title("Yam_B · native total arm contact")
    format_axis(axis, "Probe simulation time (s)", "Running peak normal force (N)")
    release_axis.set_title("Right position · unloading and return")
    format_axis(release_axis, "Probe simulation time (s)", "Pad peak per control frame (N)")
    fig.savefig(ROOT/"probe_3pos_forces.png", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    fixture_plot()
    probe_plot()

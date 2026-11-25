"""Plot measured C0 force evidence, including failed attempts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 9), layout="constrained")
    probe, grasp, arm = axes
    sources = [("Original-mask replay", ROOT / "probe/archived/summary.json")]
    sources += [(case, ROOT / "guarded-body" / case / "summary.json") for case in ("left", "center", "right")]
    sources.append(("left, 2 mm", ROOT / "probe-selective-limit/left/summary.json"))
    for name, file in sources:
        if not file.is_file():
            continue
        sample = json.loads(file.read_text())
        r = [s for s in sample["rows"] if s["phase"] == "probe"]
        if not r:
            continue
        start = r[0]["t_s"]
        probe.plot([x["t_s"] - start for x in r], [x["peak_substep_N"] for x in r], label=name)
    probe.axhline(2., color=".39", lw=1., ls="--", label="Soft cutoff")
    probe.axhspan(.5, 3.5, facecolor="#478981", alpha=.11, label="Peak acceptance band")
    probe.set(xlabel="Probe time (s)", ylabel="Running peak (N)", title="Yam_B probe · GPU substep force")

    for name, color in zip(("a1", "a2", "a3", "b1", "b2"),
                           ("#39698a", "#25978a", "#975d38", "#8b7399", "#ad922f"), strict=True):
        file = ROOT / name / "trace.npz"
        if file.is_file():
            with np.load(file) as data:
                grasp.plot(data["time"], np.max(data["jaw_normal"], axis=1), color=color, label=name.upper())
    grasp.axhspan(3., 15., facecolor="#478981", alpha=.13, label="Median acceptance band")
    grasp.axhline(25., color="#a95b46", lw=1., ls=":", label="Peak limit")
    grasp.set(xlabel="Fixture time (s)", ylabel="Larger pad force (N)", title="Opposed-pad fixture · sampled grip force")
    for name, path, color in (("Through", "arm/through.json", "#39698a"),
                              ("Film", "arm-body/film.json", "#975d38")):
        file = ROOT / path
        if file.is_file():
            rows = json.loads(file.read_text())["rows"]
            for side, index, style in (("left", 0, "-"), ("right", 1, "--")):
                arm.plot([r["time_s"] for r in rows], [r["pad_N"][index] for r in rows],
                         color=color, ls=style, label=f"{name}, {side}")
    arm.set(xlabel="Simulation time (s)", ylabel="Pad force (N)", title="Yam_B grasp attempts")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.18)
        axis.legend(frameon=False, fontsize=8, ncols=3)
    fig.savefig(ROOT / "gate1_probe_and_grasp_forces.png", dpi=285)
    plt.close(fig)


if __name__ == "__main__":
    main()

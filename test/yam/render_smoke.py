"""Render recorded Yam_B acceptance trials and plot the six-axis feedback."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from yam_platform.render_video import render


def replay(run, case, output, title, *, fixed_camera=None):
    with TemporaryDirectory(prefix="replay-", dir=output.parent) as scratch:
        scratch = Path(scratch)
        for name, source in (("model.xml", run / f"{case}.xml"),
                             ("trajectory.npz", run / f"{case}.npz")):
            (scratch / name).symlink_to(source.resolve())
        with np.load(run / f"{case}.npz") as trace:
            duration = float(trace["time"][-1])
        (scratch / "result.json").write_text(json.dumps({
            "config": {"category": case, "yaw": 0, "pinch_coeff": 0},
            "outcome": "pass", "sim_time_s": duration, "video_title": title, "follow_site": "grasp_site",
            "fixed_camera": fixed_camera,
        }))
        render(scratch, output, fps=25)
        meta = output.with_name(output.stem + "_render.json")
        data = json.loads(meta.read_text())
        data["source"] = str(run / f"{case}.npz")
        meta.write_text(json.dumps(data, indent=2) + "\n")
        output.with_name(output.stem + "_first.ppm").unlink()


def contact_curves(run, output):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.3), sharex=True, constrained_layout=True)
    colors = {"free": "#596478", "soft": "#16877e", "rigid": "#cf653b"}
    for case, color in colors.items():
        with np.load(run / f"{case}.npz") as trace:
            t = trace["time"]
            for j, ax in enumerate(axes.flat):
                ax.plot(t, trace["tau_feedback"][:, j], color=color, lw=1.65, label=case)
                ax.set_title(f"Joint {j + 1}", loc="left")
                ax.set_ylabel("Motor effort (N·m)")
    for ax in axes[-1]:
        ax.set_xlabel("Simulation time (s)")
    axes[0, 0].legend(ncol=3, frameon=True)
    fig.suptitle("Yam_B motor feedback during an identical downward probe", fontsize=15)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def grasp_curves(run, output):
    with np.load(run / "grasp.npz") as trace:
        fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True)
        t, obs = trace["time"], trace["observation"]
        axes[0].plot(t, 1000 * obs[:, 27], "--", color="#6d7590", label="Target")
        axes[0].plot(t, 1000 * obs[:, 24], color="#118079", label="Measured")
        axes[0].set_ylabel("Jaw opening (mm)")
        axes[0].legend(loc="upper right")
        axes[1].plot(t, obs[:, 26], color="#ad562d")
        axes[1].set_ylabel("Motor effort (N·m)")
        axes[2].plot(t, 1000 * trace["payload_xyz"][:, 2], color="#376da5")
        axes[2].set_ylabel("Payload centre z (mm)")
        axes[2].set_xlabel("Simulation time (s)")
        phase = trace["phase"]
        starts = np.r_[0, np.flatnonzero(phase[1:] != phase[:-1]) + 1]
        for index in starts:
            for ax in axes:
                ax.axvline(t[index], color="#b1bac2", alpha=.75, lw=1)
            axes[2].text(t[index] + .05, .93, phase[index].capitalize(),
                         transform=axes[2].get_xaxis_transform(), va="top", fontsize=8)
        fig.suptitle("Yam_B native-contact grasp and place", fontsize=15)
        fig.savefig(output, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact", type=Path, required=True)
    parser.add_argument("--grasp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--abc", action="store_true", help="Use the existing ABC top camera for grasp replay")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    contact_curves(args.contact, args.output / "contact-feedback.png")
    grasp_curves(args.grasp, args.output / "grasp-feedback.png")
    replay(args.grasp, "grasp", args.output / "rigid-grasp-place.mp4",
           "ABC workcell | Yam_B rigid grasp and place", fixed_camera="abc_top" if args.abc else None)
    replay(args.contact, "soft", args.output / "soft-contact.mp4", "Yam_B | Compliant contact probe")


if __name__ == "__main__":
    main()

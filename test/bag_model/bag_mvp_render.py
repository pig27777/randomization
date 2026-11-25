"""Render the four manipulation tests and measured probe and jaw reactions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import mujoco
import numpy as np

from bag_mvp import CORNERS, PAD, PAD_HALF

BOX_EDGES = np.asarray([(i, j) for i in range(8) for j in range(i + 1, 8)
                        if np.count_nonzero(CORNERS[i] != CORNERS[j]) == 1])


def box(pose, half):
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, pose[3:])
    return (CORNERS * half) @ mat.reshape(3, 3).T + pose[:3]


def series(path):
    trace = np.load(path / "trace.npz")
    data = {key: trace[key] for key in ("time", "q", "edges", "payload", "fixtures", "wrench", "jaw_normal", "floor")}
    trace.close()
    with (path / "config.json").open() as config:
        return data, json.load(config)


def video(output, paths, title, kind, fps):
    samples = [series(path) for path in paths]
    duration = max(d["time"][-1] for d, _ in samples)
    underside = kind == "through" and samples[0][1].get("variant") == "exposed-underface"
    projection = (0, 1) if kind == "contain" else (1, 2) if kind == "through" and not underside else (0, 2)
    fig, axes = plt.subplots(len(samples), 1, figsize=(8.0, 2.4 * len(samples)), squeeze=False)
    panels = []
    for ax, (d, cfg) in zip(axes[:, 0], samples):
        lines = []
        for color, width in (("#4185a6", .45), ("#bf6b29", 1.6), ("#303d47", 1.2), ("#303d47", 1.2)):
            line = LineCollection([], linewidth=width, colors=color)
            ax.add_collection(line)
            lines.append(line)
        p = 1000 * d["q"][:, :, projection]
        lo = p.min(axis=(0, 1)) - 14
        hi = p.max(axis=(0, 1)) + 14
        if kind in ("try", "through"):
            lo[1], hi[1] = min(-8, lo[1]), max(64, hi[1])
        if kind == "film":
            lo[1] = min(-18, lo[1])
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal",
               xlabel=("y" if projection[0] == 1 else "x") + " (mm)",
               ylabel=("y" if kind == "contain" else "z") + " (mm)")
        ax.grid(alpha=.15)
        if kind == "film" or underside:
            ax.plot([-400., cfg["support_edge"] * 1000], [0., 0.], color=".40", linewidth=2)
        elif kind != "contain":
            ax.axhline(0, color=".55", linewidth=1.2)
        probe = Circle((0., 0.), 10., edgecolor="#343d49", facecolor="none", linewidth=1.45, visible=kind == "try")
        ax.add_patch(probe)
        label = ax.text(.025, .93, "", transform=ax.transAxes, fontsize=8, va="top")
        panels.append((lines, probe, label))
    clock = fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    writer = FFMpegWriter(fps=fps, codec="h264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, output, dpi=110):
        n = round(duration * fps)
        for now in np.linspace(0., duration, n + 1):
            clock.set_text(f"{title}  |  {now:.2f} s")
            for (d, cfg), (lines, probe, label) in zip(samples, panels):
                i = int(np.argmin(abs(d["time"] - now)))
                poses = d["fixtures"][i]
                lines[0].set_segments(1000 * d["q"][i][d["edges"]][:, :, projection])
                payload = box(d["payload"][i], np.array(cfg["spec"]["size"]) / 2)
                lines[1].set_segments(1000 * payload[BOX_EDGES][:, :, projection])
                if kind == "try":
                    probe.center = 1000 * poses[0, list(projection)]
                    details = f"{d['wrench'][i,0,2]:.2f} N vertical"
                elif kind in ("film", "through"):
                    for line, pose in zip(lines[2:], poses[2:4]):
                        pad = box(pose, [PAD_HALF, *PAD / 2])
                        line.set_segments(1000 * pad[BOX_EDGES][:, :, projection])
                    details = f"{d['jaw_normal'][i,0]:.1f} / {d['jaw_normal'][i,1]:.1f} N jaws"
                else:
                    details = f"{cfg['spec']['mass']:.2f} kg, support x = {poses[1,0]*1000:.1f} mm"
                label.set_text(details if kind == "contain" else f"{cfg['case']}  ·  {details}")
            writer.grab_frame()
    plt.close(fig)


def plots(output, tries, grasps):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for path in tries:
        d, cfg = series(path)
        t = d["time"]
        x = d["fixtures"][:, 0, 0] * 1000
        f = d["wrench"][:, 0, 2]
        sweep = (t >= .62) & (t <= 1.54)
        label = cfg["case"].removeprefix("try-").capitalize()
        axes[0].plot(t, f, label=label)
        axes[1].plot(x[sweep], f[sweep], label=label)
    axes[0].set(xlabel="Time (s)", ylabel="Vertical reaction (N)")
    axes[1].set(xlabel="Probe x during sweep (mm)", ylabel="Vertical reaction (N)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "active-try.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9., 5.2), sharex="row")
    for row, path in enumerate(grasps):
        d, cfg = series(path)
        t = d["time"]
        a, b = axes[row]
        a.plot(t, d["jaw_normal"][:, 0], label="Left jaw")
        a.plot(t, d["jaw_normal"][:, 1], label="Right jaw")
        b.plot(t, (d["floor"] - d["floor"][0]) * 1000)
        b.axhline(10., color=".5", linestyle=":")
        a.set(title=cfg["case"], ylabel="Normal force (N)")
        b.set(ylabel="Payload lowest point rise (mm)")
        a.legend(fontsize=8)
        for ax in (a, b):
            ax.set(xlabel="Time (s)")
            ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "grasp-forces.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--film", default="film")
    parser.add_argument("--through", default="through")
    args = parser.parse_args()
    out = args.run
    cases = [
        ("contain", "Containment under support motion", ("contain-200", "contain-350")),
        ("try", "Identical press and sweep", ("try-left", "try-center", "try-right")),
        ("film", "Seam-film pinch and release", (args.film,)),
        ("through", "Through-film grasp and release", (args.through,)),
    ]
    for kind, title, ids in cases:
        video(out / f"{kind}.mp4", [out / k for k in ids], title, kind, fps=30)
    plots(out, [out / k for k in cases[1][2]], [out / args.film, out / args.through])


if __name__ == "__main__":
    main()

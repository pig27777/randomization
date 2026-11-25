"""Render measured PE benchmark trajectories and matched coupon comparisons."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
import numpy as np

from pe_physics import PAIRS, save_json


def available(run, case, key):
    return f"{case}/{key}" in run.files


def curve(ax, run, case, key, label, *, factor=1., column=None):
    values = run[f"{case}/{key}"]
    if column is not None:
        values = values[:, column]
    ax.plot(run[f"{case}/time"], factor * values, label=label)


def save_figure(fig, output):
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def mesh_collection(ax, positions, edges, projection):
    lines = LineCollection(1000 * positions[edges][:, :, projection], linewidth=.38, colors="#18648a")
    ax.add_collection(lines)
    ax.autoscale()
    ax.set_aspect("equal")
    return lines


def mesh_panel(output, runs, case, name, key, select=-1):
    eligible = [(backend, run) for backend, run in runs if available(run, case, "q")]
    if not eligible:
        return
    fig, axes = plt.subplots(len(eligible), 2, squeeze=False, figsize=(8.5, 3.4 * len(eligible)))
    for row, (backend, run) in enumerate(eligible):
        state = run[f"{case}/{key}"]
        q = state[select] if key == "q" else state
        for col, projection in enumerate(((0, 1), (0, 2))):
            ax = axes[row, col]
            mesh_collection(ax, q, run[f"{case}/edges"], projection)
            ax.set(xlabel="x (mm)", ylabel=f"{'y' if col == 0 else 'z'} (mm)",
                   title=f"{backend} · {'top' if col == 0 else 'side'}")
    save_figure(fig, output / name)


def fall_plot(output, runs):
    eligible = [(name, run) for name, run in runs if available(run, "B01", "com")]
    if not eligible:
        return
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.2), sharex=True)
    for label, run in eligible:
        curve(axes[0, 0], run, "B01", "com", label, factor=1000, column=2)
        curve(axes[0, 1], run, "B01", "rms_speed", label, factor=1000)
        for key, desc in (("kinetic", "kinetic"), ("gravity_energy", "gravity"), ("bending_energy", "bending")):
            curve(axes[1, 0], run, "B01", key, f"{label} {desc}", factor=1000)
        force = -run["B01/wrench"][:, -2, 2]
        axes[1, 1].plot(run["B01/time"], force, label=label)
    for ax, label in zip(axes.ravel(), ("COM height (mm)", "RMS speed (mm/s)", "Film energy (mJ)", "Table load (N)")):
        ax.set(ylabel=label, xlabel="Time (s)")
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    axes[0, 1].axhline(5., color=".5", linestyle=":")
    save_figure(fig, output / "free-fall.png")


def stretch(output, runs):
    cases = ("B02-X", "B02-Y")
    eligible = [(name, run) for name, run in runs if any(available(run, k, "strain") for k in cases)]
    if not eligible:
        return {}
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6))
    metrics = {}
    for axis, case in enumerate(cases):
        metrics[case] = {}
        for label, run in eligible:
            if not available(run, case, "strain"):
                continue
            eps = run[f"{case}/strain"]
            force = -run[f"{case}/clamp_force"][:, axis]
            axes[0, axis].plot(eps * 100, force, label=label, linewidth=1.25)
            idx = int(np.argmin(abs(run[f"{case}/time"] - .47)))
            stretch = run[f"{case}/edge_stretch"][idx]
            axes[1, axis].hist(stretch, bins=24, histtype="step", label=label, linewidth=1.25)
            time = run[f"{case}/time"]
            samples = np.array([np.argmin(abs(time - t)) for t in (.15, .31, .47)])
            if time[-1] >= .60:
                metrics[case][label] = dict(
                    engineering_strain=eps[samples].tolist(),
                    clamp_force_N=force[samples].tolist(),
                    apparent_axial_stiffness_N=float(np.dot(eps[samples], force[samples]) /
                                                     np.dot(eps[samples], eps[samples])))
        axes[0, axis].set(xlabel="Engineering strain (%)", ylabel="Clamp reaction (N)", title=f"{case[-1]} loading")
        axes[1, axis].set(xlabel="Edge length ratio at peak load", ylabel="Edges")
        for ax in axes[:, axis]:
            if ax.lines or ax.patches:
                ax.legend()
            ax.grid(alpha=.2)
        data = metrics[case]
        if "Flex CPU" in data and "Newton VBD CUDA" in data:
            a = np.asarray(data["Flex CPU"]["engineering_strain"])
            b = np.asarray(data["Newton VBD CUDA"]["engineering_strain"])
            data["relative_strain_error_at_matched_command"] = (np.abs(a - b) / np.abs(a)).tolist()
    save_figure(fig, output / "stretch.png")
    return metrics


def bend(output, runs):
    case = "B03-A"
    eligible = [(name, run) for name, run in runs if available(run, case, "tip_z")]
    if not eligible:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for label, run in eligible:
        curve(ax, run, case, "tip_z", label, factor=1000)
    ax.set(xlabel="Time (s)", ylabel="Tip height (mm)")
    ax.legend()
    ax.grid(alpha=.2)
    save_figure(fig, output / "cantilever.png")


def contact(output, runs):
    cases = ("B04-PLATES", "B04-DOUBLE", "B04-U", "B04-ACCORDION")
    eligible = [(name, run) for name, run in runs if any(available(run, k, "pair_normal") for k in cases)]
    if not eligible:
        return
    fig, axes = plt.subplots(len(cases), 2, figsize=(8.5, 9.5), sharex="row")
    for row, case in enumerate(cases):
        for label, run in eligible:
            if not available(run, case, "pair_normal"):
                continue
            curve(axes[row, 0], run, case, "pair_normal", label, column=PAIRS.index("film_self"))
            curve(axes[row, 1], run, case, "self_crossings", label)
        for ax, ylabel in zip(axes[row], ("Self-contact normal sum (N)", "Crossing triangle pairs")):
            ax.set(xlabel="Time (s)", ylabel=ylabel, title=case)
            if ax.lines:
                ax.legend(fontsize=7)
            ax.grid(alpha=.2)
    save_figure(fig, output / "self-contact.png")


def fall_video(output, run, backend):
    case = "B01"
    q = run[f"{case}/q"]
    edges = run[f"{case}/edges"]
    time = run[f"{case}/time"]
    fps = 50
    frames = np.unique([np.argmin(abs(time - t)) for t in np.arange(0., time[-1] + .5 / fps, 1 / fps)])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    lines = []
    for i, projection in enumerate(((0, 1), (0, 2))):
        ax = axes[i]
        lines.append(mesh_collection(ax, q[0], edges, projection))
        mins = 1000 * q[:, :, projection].min(axis=(0, 1)) - 8
        maxs = 1000 * q[:, :, projection].max(axis=(0, 1)) + 8
        ax.set(xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]), xlabel="x (mm)",
               ylabel=f"{'y' if i == 0 else 'z'} (mm)", title="Top" if i == 0 else "Side")
    axes[1].axhline(0., color=".55", linewidth=1)
    title = fig.suptitle("")
    fig.tight_layout()
    writer = FFMpegWriter(fps=fps, codec="h264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, output, dpi=105):
        for index in frames:
            for line, projection in zip(lines, ((0, 1), (0, 2))):
                line.set_segments(1000 * q[index][edges][:, :, projection])
            title.set_text(f"{backend}   ·   B01   ·   {time[index]:.3f} s")
            writer.grab_frame()
    plt.close(fig)


class BackendTrace:
    def __init__(self, archive, backend):
        self.archive = archive
        self.prefix = f"{backend}/"
        self.files = [key[len(self.prefix):] for key in archive.files if key.startswith(self.prefix)]

    def __getitem__(self, label):
        return self.archive[self.prefix + label]


def containment(output, runs):
    cases = ("B05-A-200", "B05-A-350", "B05-B", "B05-C", "B05-D")
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5))
    for label, run in runs:
        for case in cases:
            if not available(run, case, "payload_floor"):
                continue
            ax = axes[0] if case.startswith("B05-A") else axes[1]
            curve(ax, run, case, "payload_floor", f"{label} {case}", factor=1000.)
    for ax in axes:
        if ax.lines:
            ax.legend(fontsize=7)
        ax.grid(alpha=.2)
        ax.set(xlabel="Time (s)", ylabel="Lowest payload corner (mm)")
    save_figure(fig, output / "payload-containment.png")


def probe(output, runs):
    cases = ("B06-S0-P1", "B06-S2-P0", "B06-S2-P1")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharey=True)
    found = False
    for ax, (backend, run) in zip(axes, runs):
        for case in cases:
            if not available(run, case, "wrench"):
                continue
            found = True
            height = run[f"{case}/fixtures"][:, 0, 2]
            target = run[f"{case}/fixture_target"][:, 0, 2]
            displacement = 1000 * (target[0] - height)
            force = run[f"{case}/wrench"][:, 0, 2]
            keep = abs(displacement) < 40.
            ax.plot(displacement[keep], force[keep], label=case)
        ax.axhline(5., color=".45", linestyle=":", label="5 N")
        ax.set(xlabel="Probe descent (mm)", ylabel="Vertical wrench (N)", title=backend)
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    if found:
        save_figure(fig, output / "press-diagnostic.png")
    else:
        plt.close(fig)


def gentle_probe(output, path):
    with np.load(path / "raw_trajectories.npz") as archive:
        time, commanded_z, normal = archive["B06-S2-P1/probe_microsteps"].T
    active = np.flatnonzero(normal > .01)
    if not len(active):
        return
    peak = int(np.argmax(normal))
    event = time[peak]
    selection = (time >= event - .004) & (time <= event + .010)
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    milliseconds = 1000 * (time[selection] - event)
    ax.plot(milliseconds, normal[selection], color="#18648a", label="Measured reaction")
    ax.axhline(5., color=".35", linestyle=":", label="Limit")
    ax.axvline(0., color=".6", linestyle="--")
    ax.set(xlabel="Time from force peak (ms)", ylabel="Vertical reaction (N)")
    other = ax.twinx()
    other.plot(milliseconds, 1000 * commanded_z[selection], color="#b17624", alpha=.85)
    other.set_ylabel("Commanded probe height (mm)", color="#a2651c")
    ax.legend(loc="upper right")
    ax.grid(alpha=.2)
    save_figure(fig, output / "press-gentle.png")


def calibration_checks(runs, stretching, heldout):
    result = dict(reference="Uncalibrated engineering PE profile",
                  status="Simulation cross-check at available fixtures",
                  stretching=stretching, bending={}, compression={},
                  friction_slip="pending physical gate")
    for label, run in runs:
        if available(run, "B03-A", "tip_z"):
            z = run["B03-A/tip_z"]
            t = run["B03-A/time"]
            trough = int(np.argmin(z))
            result["bending"][label] = dict(trough_time_s=float(t[trough]), trough_tip_m=float(z[trough]),
                                             final_tip_m=float(z[-1]), final_rms_m_s=float(run["B03-A/rms_speed"][-1]))
        if available(run, "B04-DOUBLE", "wrench"):
            time = run["B04-DOUBLE/time"]
            hold = (time >= .30) & (time <= .40)
            force = run["B04-DOUBLE/wrench"][hold, 1, 2]
            result["compression"][label] = dict(window_s=[.30, .40],
                                                 platen_force_median_N=float(np.median(force)),
                                                 platen_force_p95_N=float(np.quantile(force, .95)))
    if heldout is not None:
        row = json.loads((heldout / "summary.json").read_text())[0]
        baseline = stretching["B02-Y"]["Newton VBD CUDA"]["engineering_strain"]
        reference = np.asarray(stretching["B02-Y"]["Flex CPU"]["engineering_strain"])
        calibrated = np.array([pt["strain"] for pt in row["gauge_response"][:3]])
        result["heldout_Y"] = dict(scale=row["config"]["vbd_extensional_scale"],
                                  fitting_case="B02-X", anchor_strains=calibrated.tolist(),
                                  rms_strain_error_before=float(np.sqrt(np.mean((baseline - reference)**2))),
                                  rms_strain_error_after=float(np.sqrt(np.mean((calibrated - reference)**2))),
                                  peak_edge_ratio=row["maximum_edge_ratio"], status=row["status"])
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("raw", type=Path, help="Consolidated raw_trajectories.npz")
    parser.add_argument("--vbd-calibration", type=Path)
    parser.add_argument("--gentle-probe", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.raw) as archive:
        runs = [(name, BackendTrace(archive, key)) for name, key in
                (("Flex CPU", "flex"), ("Newton VBD CUDA", "vbd"))]
        mesh_panel(args.output, runs, "B01", "initial-geometry.png", "current")
        mesh_panel(args.output, runs, "B01", "final-geometry.png", "q")
        mesh_panel(args.output, runs, "B03-B", "drape-geometry.png", "q")
        mesh_panel(args.output, runs, "B04-U", "fold-geometry.png", "q")
        fall_plot(args.output, runs)
        bend(args.output, runs)
        contact(args.output, runs)
        containment(args.output, runs)
        probe(args.output, runs)
        if args.gentle_probe is not None:
            gentle_probe(args.output, args.gentle_probe)
        calibration = stretch(args.output, runs)
        save_json(args.output / "material_calibration.json",
                  calibration_checks(runs, calibration, args.vbd_calibration))
        for name, run in runs:
            if available(run, "B01", "q"):
                slug = "flex" if name == "Flex CPU" else "vbd"
                fall_video(args.output / f"{slug}-free-fall.mp4", run, name)


if __name__ == "__main__":
    main()

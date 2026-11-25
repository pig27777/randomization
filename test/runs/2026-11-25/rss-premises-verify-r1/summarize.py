"""Summarize only observed Stage 1 data and render a diagnostic figure."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
CASES = ("1A", "1B", "1C-empty", "1C-500", "1C-1000")
ARCHIVES = ("entry", "loaded-preflight", "film-pinch", "through-bag", "grasp-reset", "grasp-height",
            "joint-load", "joint-load-retry", "through-approach", "film-10g", "through-10g", "joint-10g")


def trial_summary(path):
    trial = json.loads(path.read_text())
    rows = trial.pop("rows")
    payload_mass = trial.pop("mass_kg")
    with np.load(path.with_suffix(".npz")) as trace:
        triangles = trace["film"][0][trace["faces"]].astype(float)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    film_area = .5 * np.linalg.norm(normal, axis=1).sum()
    material = trial["material"]
    default_density = material["density_kg_m3"] * material["thickness_m"]
    areal_density = trial.get("film_areal_density_kg_m2", default_density)
    film_mass = trial.get("particle_film_mass_kg", film_area * areal_density)
    geometry = [r["geometry"] for r in rows if r["geometry"]]
    clearance = [g["clearance_m"] for g in geometry]
    hold_start = trial.get("lift_end_s", 2.4) if trial.get("trial") == "grasp" else None
    hold = [r for r in rows if r["time_s"] >= hold_start] if hold_start is not None else []
    return dict(
        **trial, payload_mass_kg=payload_mass, film_mass_kg=float(film_mass),
        whole_bag_mass_kg=float(payload_mass + film_mass), reference_film_area_m2=float(film_area), data=str(path.relative_to(ROOT)), frames=str(path.with_suffix(".npz").relative_to(ROOT)),
        valid_steps=len(rows) - 1, observed_end_s=rows[-1]["time_s"],
        all_states_finite=all(r["finite"] for r in rows),
        max_edge_ratio=max(r["max_edge_ratio"] for r in rows if r["max_edge_ratio"] is not None),
        max_loaded_arm_core=max(r["direct_arm_core_loaded"] for r in rows),
        max_loaded_arm_table=max((r.get("loaded_arm_table", 0) for r in rows)),
        max_intersecting_triangles=max((g["intersecting_triangles"] for g in geometry), default=None),
        max_outside_corners=max((g["outside_corners"] for g in geometry), default=None),
        max_core_intrusion_m=max((g["depth_upper_m"] for g in geometry), default=None),
        max_core_clearance_m=max(clearance, default=None),
        max_film_clearance_m=max((r["min_film_z_m"] for r in rows if r["min_film_z_m"] is not None), default=None),
        hold_end_s=hold[-1]["time_s"] if hold else None,
        hold_min_core_clearance_m=min((r["geometry"]["clearance_m"] for r in hold if r["geometry"]), default=None),
        hold_min_film_clearance_m=min((r["min_film_z_m"] for r in hold if r["min_film_z_m"] is not None), default=None),
    )


def diagnostic_plot(trials):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    for item in trials:
        rows = json.loads((ROOT / item["data"]).read_text())["rows"]
        t = [r["time_s"] for r in rows]
        variant = " approach" if item.get("through_approach") else " height retry" if item.get("height_retry") else ""
        label = f'{item["case"]}{variant}'
        axes[0].plot(t, [r["max_edge_ratio"] for r in rows], label=label)
        if item["payload_mass_kg"]:
            axes[1].plot(t, [1000 * r["geometry"]["clearance_m"] if r["geometry"] else np.nan
                             for r in rows], label=label)
    axes[0].axhline(1.05, color="#a23f3a", lw=1, ls="--", label="Gate 1A limit")
    axes[0].set(ylabel="Maximum film edge ratio", xlabel="Simulation time (s)")
    axes[1].set(ylabel="Payload clearance (mm)", xlabel="Simulation time (s)")
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        if ax.lines:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Stage 1: measured dynamics before the suspension gate")
    for fmt in ("pdf", "png"):
        fig.savefig(ROOT / f"fig_stage1_diagnostic.{fmt}", dpi=280)
    plt.close(fig)



CORNERS = np.array(list(np.ndindex(2, 2, 2))) * 2 - 1
BOX_FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))


def box(actor, axis, half, color, alpha):
    vertices = Rotation.from_quat(actor[3:]).apply(CORNERS * half) + actor[:3]
    collection = Poly3DCollection(1000 * vertices[np.array(BOX_FACES)],
                                  facecolor=color, edgecolor=color, linewidth=.5, alpha=alpha)
    axis.add_collection3d(collection)


def contact_plot(items):
    fig, axes = plt.subplots(len(items), 2, figsize=(9, 3.35 * len(items)),
                             subplot_kw={"projection": "3d"}, squeeze=False, layout="constrained")
    for item, pair in zip(items, axes, strict=True):
        with np.load(ROOT / item["frames"]) as trace:
            for index, ax in zip((0, -1), pair, strict=True):
                film, bodies = trace["film"][index], trace["bodies"][index]
                t = trace["steps"][index] * item["dt_s"]
                visible = Poly3DCollection(1000 * film[trace["faces"]], facecolor="#6ca9c8",
                                           edgecolor="#507e94", linewidth=.12, alpha=.28)
                ax.add_collection3d(visible)
                payload = int(trace["core_id"])
                if payload >= 0:
                    box(bodies[payload], ax, trace["half"], "#d5803f", .8)
                for i, body in enumerate(trace["pad_body_ids"]):
                    pose = bodies[body]
                    r = Rotation.from_quat(pose[3:])
                    c = pose[:3] + r.apply(trace["pad_local_centers"][i])
                    q = (r * Rotation.from_quat(trace["pad_local_rotation"][i])).as_quat()
                    box(np.r_[c, q], ax, trace["pad_half_sizes"][i], "#17324e", .9)
                x, y = 1000 * film[:, :2].T
                ax.set(xlim=(x.min() - 4, x.max() + 4), ylim=(y.min() - 4, y.max() + 4),
                       zlim=(0, max(48, 1000 * film[:, 2].max() + 10)),
                       xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
                       title=f'{item["case"]} | {t:.4f} s')
                ax.set_box_aspect([4.0, 3., 1.25])
                ax.view_init(elev=28, azim=-112)
    fig.suptitle("Stage 1: simulated contact geometry at start and termination")
    fig.legend([Line2D([], [], color=c, lw=4) for c in ("#6ca9c8", "#d5803f", "#17324e")],
               ["Film", "Payload", "Yam_B pads"], loc="outside lower center", ncols=3, frameon=False)
    for fmt in ("pdf", "png"):
        fig.savefig(ROOT / f"fig_stage1_contact_diagnostic.{fmt}", dpi=280)
    plt.close(fig)


def main():
    trials = [
        trial_summary(path) for archive in ARCHIVES for case in CASES
        if (path := ROOT / archive / f"{case}.json").is_file() and path.with_suffix(".npz").is_file()
    ]
    if not trials:
        raise ValueError("No complete Stage 1 logs found")
    (ROOT / "stage1_audit.json").write_text(json.dumps({"trials": trials}, indent=2, allow_nan=False) + "\n")
    grasp = [t for t in trials if t.get("trial") == "grasp" and t.get("planned_reset_verified")]
    corrected = [t for t in grasp if t.get("specified_film_mass_g") == 10.]
    diagnostic_plot(corrected or grasp or trials)
    names = ("film-10g/1A.json", "through-10g/1B.json") if corrected else (
        "grasp-reset/1A.json", "through-approach/1B.json")
    geometry = [t for t in grasp if t["data"] in names]
    if geometry:
        contact_plot(geometry)


if __name__ == "__main__":
    main()

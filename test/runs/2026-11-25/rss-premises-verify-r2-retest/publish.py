"""Summarize actual physics windows and draw diagnostic figures from archived states."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
R1 = ROOT.parent / "rss-premises-verify-r1"
CASES = ("1B-600", "1B-200", "1C-empty", "1C-150", "1C-250")
CORNER_SIGNS = np.array(list(np.ndindex(2, 2, 2))) * 2 - 1


def read(path):
    return json.loads(path.read_text())


def longest(rows, test, stride_s):
    groups = []
    current = []
    for row in rows:
        if not test(row) or (current and row["t_s"] - current[-1]["t_s"] > stride_s + 1.e-8):
            if current:
                groups.append(current)
            current = []
        if test(row):
            current.append(row)
    if current:
        groups.append(current)
    return max(groups, key=lambda g: g[-1]["t_s"] - g[0]["t_s"], default=[])


def duration(group):
    return group[-1]["t_s"] - group[0]["t_s"] if group else 0.


def summarize(case):
    files = {"1B-600": "newton/1B.json", "1B-200": "newton-light-b/1B.json",
             "1C-empty": "newton/1C-empty.json", "1C-150": "newton-light-150/1C-150.json",
             "1C-250": "newton-light-250/1C-250.json"}
    p = ROOT / files[case]
    if not p.is_file():
        return None
    through = case.startswith("1B")
    data = read(p)
    rows = data.pop("rows")
    geo = [row["geometry"] for row in rows if row["geometry"] is not None]
    # Contact radius plus broad-phase gap bounds the near-table band for this model.
    floor_band = .0005
    stride_s = data["dt_s"] * data["audit_stride"]
    hold_rows = [row for row in rows if data["lift_end_s"] <= row["t_s"] <= (data["release_begin_s"] or 4.6)]
    def clearance(row):
        core = row["geometry"]["clearance_m"] if row["geometry"] else None
        return row["film_min_m"], core
    def fully_up(row):
        film, core = clearance(row)
        minimum = .10 if through else floor_band
        return (film is not None and film > floor_band and
                (not data["payload_mass_g"] or core is not None and core >= minimum))
    def loaded(row):
        return all(force[0] > 0 for force in row.get("pad_N_normal_tangent", [[0., 0.], [0., 0.]]))
    core_raised = longest(rows, lambda r: r["geometry"] is not None and
                          r["geometry"]["clearance_m"] >= .002, stride_s)
    suspended = longest(hold_rows, lambda r: fully_up(r) and loaded(r), stride_s)
    clear_hold = [clearance(r) for r in hold_rows]
    steady = None
    if not through and duration(suspended) >= 2. - 1.e-8:
        endpoint = suspended[-1]["t_s"]
        window = [r for r in suspended if endpoint - 1. < r["t_s"] <= endpoint and "F_z_est_N" in r]
        if len(window) >= 24:
            steady = dict(window_s=[endpoint - 1., endpoint], samples=len(window),
                          F_z_est_N=float(np.mean([r["F_z_est_N"] for r in window])),
                          q_error_mean_rad=np.mean([r["error"][:6] for r in window], axis=0).tolist(),
                          applied_effort_mean_Nm=np.mean([r["applied_effort"][:6] for r in window], axis=0).tolist())
    release_drop = None
    release = data["release_begin_s"]
    if release and not data["failure"] and rows[-1]["t_s"] > release:
        begin = min(rows, key=lambda r: abs(r["t_s"] - release))
        release_drop = 1000. * (begin["geometry"]["clearance_m"] - geo[-1]["clearance_m"])
    forces = [r for r in rows if "pad_N_normal_tangent" in r]
    after_r1 = .972 if case == "1B-600" else None
    steps = data["steps"] * data["dt_s"] * 25.
    data.update(
        source=str(p.relative_to(ROOT)), end_s=rows[-1]["t_s"],
        control_steps_per_wall_s=steps / data["wall_s"],
        control_steps_per_wall_s_excluding_geometry_readback=steps / (data["wall_s"] - data["geometry_wall_s"]),
        all_finite=all(r["finite"] for r in rows),
        min_payload_clearance_mm=min((g["clearance_m"]*1000. for g in geo), default=None),
        max_payload_clearance_mm=max((g["clearance_m"]*1000. for g in geo), default=None),
        max_film_clearance_mm=max((r["film_min_m"]*1000. for r in rows if r["film_min_m"] is not None), default=None),
        min_hold_payload_clearance_mm=min((c*1000. for _, c in clear_hold if c is not None), default=None),
        min_hold_film_clearance_mm=min((f*1000. for f, _ in clear_hold if f is not None), default=None),
        max_nom_pen_um=max((g["nominal_pen_um"] for g in geo), default=None),
        max_inset_pen_um=max((g["inset_pen_um"] for g in geo), default=None),
        peak_inset_triangles=max((g["inset_triangles"] for g in geo), default=None),
        inset_frames=sum(g["inset_triangles"] > 0 for g in geo) if geo else None,
        max_corner_outside_um=max((g["outside_max_um"] for g in geo), default=None),
        bilateral_loaded_hold_s=duration(suspended), core_above_2_mm_s=duration(core_raised),
        hold_audit_frames=len(hold_rows), release_drop_mm=release_drop, steady=steady,
        peak_pad_normal_tangent_N=np.max([r["pad_N_normal_tangent"] for r in forces], axis=0).tolist() if forces else None,
        peak_pad_loaded_vertices=np.max([r["pad_loaded_vertices"] for r in forces], axis=0).tolist() if forces else None,
        stop_pad_normal_tangent_N=rows[-1].get("pad_N_normal_tangent"),
        r1_stop_s=after_r1, additional_sim_s=rows[-1]["t_s"] - after_r1 if after_r1 is not None else None,
    )
    return data


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"{name}.{ext}", dpi=270)
    plt.close(fig)


def jaw_plot(jaws):
    fig, axes = plt.subplots(3, 2, figsize=(9.4, 7.4), sharex="col", layout="constrained")
    for col, run in enumerate(jaws):
        data = run["frames"]
        t = np.array([r["t_s"] for r in data])
        ax = axes[:, col]
        ax[0].plot(t, [r["jaw_gap_mm"] for r in data], c="#2b5b83", label="Measured jaw gap")
        ax[0].axhline(run["double_layer_criterion_mm"], ls="--", c="#8e431f", label="Double-layer reference")
        ax[0].set_yscale("log")
        ax[0].set(ylabel="Gap (mm)", title=run["case"])
        for i, pad in enumerate(("Left", "Right")):
            color = ("#227c78", "#b55b2a")[i]
            ax[1].plot(t, [r["inward_force_N"][i] for r in data], c=color, label=pad + " axial")
            ax[1].plot(t, [r["tangent_force_N"][i] for r in data], c=color, ls="--", label=pad + " tangential")
        ax[1].set(ylabel="Net body force (N)")
        for field, label in (("mid_z_mm", "Pad center"), ("film_min_z_mm", "Film minimum"),
                             ("film_max_z_mm", "Film maximum")):
            ax[2].plot(t, [r[field] for r in data], label=label)
        ax[2].set(xlabel="Simulation time (s)", ylabel="Height (mm)")
        for a in ax:
            a.grid(alpha=.2)
            a.axvline(.4, c=".55", lw=.7)
            a.axvline(2.4, c=".55", lw=.7)
            a.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 1]:
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, "fig_jaw_diagnosis")


def polygon(ax, center, quat, half, color):
    world = Rotation.from_quat(quat).apply(CORNER_SIGNS * half) + center
    projected = world[:, [0, 2]] * 1.e3
    hull = ConvexHull(projected)
    ax.fill(*projected[hull.vertices].T, color=color, alpha=.6)


def frame_plot():
    light = ROOT / "newton-light-b/1B.npz"
    sources = (
        (R1 / "film-10g/1A.npz", "1A", [.4, 1.04, 2.4]),
        (R1 / "joint-10g/1C-empty.npz", "Empty", [.4, 2.4, 3.2]),
        (ROOT / "newton/1B.npz", "1B 600 g", [.68, .96, .98]),
    )
    if light.is_file():
        with np.load(light) as data:
            end = float(data["steps"][-1] * .0005)
        sources += ((light, "1B 200 g", [.68, max(.68, end - .08), end]),)
    fig, axes = plt.subplots(len(sources), 3, figsize=(10.4, 2.4 * len(sources)),
                             squeeze=False, sharex=True, sharey="row",
                             layout="constrained")
    for (path, name, frame_times), axes_row in zip(sources, axes, strict=True):
        with np.load(path) as data:
            triangles = data["faces"]
            edges = np.unique(np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                                      triangles[:, [2, 0]])), axis=1), axis=0)
            times = data["steps"] * .0005
            for ax, t in zip(axes_row, frame_times, strict=True):
                i = np.abs(times - t).argmin()
                film = data["film"][i]
                bodies = data["bodies"][i]
                ax.add_collection(LineCollection(1.e3 * film[edges][:, :, [0, 2]], colors="#4789a4",
                                                  linewidths=.33, alpha=.23))
                for p, bid in enumerate(data["pad_body_ids"]):
                    r = Rotation.from_quat(bodies[bid, 3:])
                    world = r.apply(data["pad_local_centers"][p]) + bodies[bid, :3]
                    quat = (r * Rotation.from_quat(data["pad_local_rotation"][p])).as_quat()
                    polygon(ax, world, quat, data["pad_half_sizes"][p], "#223154")
                core = int(data["core_id"])
                if core >= 0:
                    polygon(ax, bodies[core, :3], bodies[core, 3:], data["half"], "#c07831")
                ax.set(title=f"{name} | {times[i]:.3f} s", xlim=(-240, 180),
                       ylim=(-8, 65 if name.startswith("1B") else 290))
                ax.axhline(0, color=".3", lw=.75)
                ax.grid(alpha=.12)
    for ax in axes[-1]:
        ax.set_xlabel("World X (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("World Z (mm)")
    fig.suptitle("Simulated geometry | film (blue), pads (navy), payload (orange)")
    save(fig, "fig_retest_keyframes")


def flex_plot():
    paths = {case: ROOT / "flex-cpu" / case / "trace.npz" for case in
             ("try-left", "try-center", "try-right", "film", "through")}
    if not all(path.is_file() for path in paths.values()):
        return
    fig, (ax, lift) = plt.subplots(1, 2, figsize=(9.6, 3.5), layout="constrained")
    for case, color in zip(("try-left", "try-center", "try-right"),
                           ("#2b729c", "#388769", "#b2652e"), strict=True):
        with np.load(paths[case]) as data:
            m = (data["time"] >= .62) & (data["time"] <= 1.54)
            ax.plot(data["fixtures"][m, 0, 0] * 1000.,
                    np.maximum(0., data["wrench"][m, 0, 2]), label=case.removeprefix("try-").title(), c=color)
    for case, color in zip(("film", "through"), ("#347e9d", "#b87341"), strict=True):
        with np.load(paths[case]) as data:
            lift.plot(data["time"], (data["floor"] - data["floor"][0])*1000., c=color, label=case.title())
    ax.set(xlabel="Probe position (mm)", ylabel="Probe reaction (N)", title="Three core locations")
    lift.set(xlabel="Simulation time (s)", ylabel="Change in payload clearance (mm)", title="Lift and release")
    for panel in (ax, lift):
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(alpha=.18)
        panel.legend(frameon=False)
    fig.suptitle("MuJoCo Flex fixture data, 50 μs CPU step")
    save(fig, "fig_flex_reference")


def main():
    trials = {case: summarize(case) for case in CASES}
    empty = trials["1C-empty"]
    force_delta = {}
    for case in CASES[-2:]:
        run = trials[case]
        val = run["steady"] if run else None
        ref = empty["steady"] if empty else None
        force_delta[case] = val["F_z_est_N"] - ref["F_z_est_N"] if val and ref else None
    jaws = read(ROOT / "jaw_diagnosis.json")
    flex = {}
    for name in ("try-left", "try-center", "try-right", "film", "through"):
        p = ROOT / "flex-cpu" / name / "summary.json"
        flex[name] = read(p) if p.is_file() else None
    gate_1b = trials["1B-200"]
    known_kill = bool((gate_1b and (gate_1b["solid_direct_count"] or gate_1b["peak_inset_triangles"]
                       or gate_1b["bilateral_loaded_hold_s"] < 1.5)) or
                      (empty and empty["bilateral_loaded_hold_s"] < 2.))
    reason = None
    if known_kill:
        if gate_1b and gate_1b["failure"]:
            reason = gate_1b["failure"]
        else:
            reason = "no valid empty-bag suspension window"
    result = dict(
        verdict="KILL" if known_kill else None,
        reason=reason, newton=trials, joint_static_delta_N=force_delta,
        lightweight_expected_deltas_N={"1C-150": 1.4715, "1C-250": 2.4525},
        abandoned_heavy_joint_cases_g=[500, 1000],
        both_mass_deltas_computed=all(x is not None for x in force_delta.values()),
        mass_revisions=1, jaw_mesh_tuning_rounds_executed=0, rollback_engine="MuJoCo/MJWarp 4.0.1",
        flex_cpu_50us=flex, gpu_count_on_ubun=1, eight_gpu_run=False,
        jaw_diagnosis_file="jaw_diagnosis.json",
    )
    (ROOT / "retest_summary.json").write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    jaw_plot(jaws)
    frame_plot()
    flex_plot()


if __name__ == "__main__":
    main()

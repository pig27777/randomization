"""Gate A on Stage 2 probes and frozen Stage 1 evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.low_order_parcel.build_stage1_envelopes import OUTPUT as CONFIG, ROOT
from src.low_order_parcel.lumped_contact_proxy import ContactInputs, ContactState as C
from src.low_order_parcel.macro_parcel_sim import GRAVITY, MacroParcelSim

CURVES = ROOT / "test/runs/2026-11-20/bag/rss2027-stage1-reference-curves-r1/stage1_whole_bag_data.json"
BASELINE_S = (539.12, 579.86, 4263.54)


def longest(mask, dt):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return float(np.max(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1), initial=0) * dt)


def both_captured(trace):
    return (trace["contact_state"][:, :, C.CAPTURE] == 1.).all(axis=1)


def airborne(trace):
    return (trace["support_N"] == 0.) & (trace["table_clearance_m"] > .001)


def first_index(mask):
    return int(np.flatnonzero(mask)[0]) if mask.any() else None


def empty_result(trace):
    tension = float(trace["tension_N"].max())
    capture = float(trace["contact_state"][:, :, C.CAPTURE].max())
    return dict(peak_tension_N=tension, peak_capture=capture, pass_gate=tension == 0. and capture == 0.)


def side_metrics(trace, nominal, reference_k, dt):
    captured = both_captured(trace)
    stable = airborne(trace) & captured & (np.linalg.norm(trace["velocity_m_s"], axis=1) < .020)
    stable_s = longest(stable, dt)
    weight = nominal["mass_kg"] * GRAVITY
    pull = trace["elastic_pull_N"]
    loading = (trace["time_s"] > .5) & (trace["support_N"] > 0.) & captured & (pull > .15 * weight) & (pull < .8 * weight)
    separation = trace["pad_position_m"][:, :, 2].mean(axis=1) - trace["com_m"][:, 2]
    slope = float(np.polyfit(separation[loading], pull[loading], 1)[0]) if loading.sum() >= 3 else None
    in_band = slope is not None and .75 * reference_k[0] <= slope <= 1.25 * reference_k[1]
    return dict(hover_s=stable_s, final_clearance_m=float(trace["table_clearance_m"][-1]),
                stiffness_N_per_m=slope, stiffness_samples=int(loading.sum()), stage1_stiffness_N_per_m=reference_k,
                pass_gate=stable_s >= .20 and in_band)


def feed_metrics(trace, dt):
    observed = trace["contact_state"][:, :, C.CAPTURE] == 1.
    falling = np.diff(observed.astype(int), axis=0, prepend=0) < 0
    first_drop = first_index(falling.any(axis=1))
    order = np.flatnonzero(falling[first_drop]).tolist() if first_drop is not None else []
    return dict(bilateral_unsupported_s=longest(airborne(trace) & both_captured(trace), dt),
                slipped_m=trace["contact_state"][-1, :, C.SLIP_CUM].tolist(), first_release_fingers=order,
                **empty_result(trace))


def slope_between(t, values, low, high):
    fit = (t >= low) & (t < high)
    return float(np.polyfit(t[fit], values[fit], 1)[0]) if fit.sum() >= 3 else None


def asymmetric_metrics(trace):
    t = trace["time_s"]
    slipping = trace["slip_rate_mps"] > 1e-7
    first_slips = [first_index(slipping[:, i]) for i in range(2)]
    captures = trace["contact_state"][:, :, C.CAPTURE] == 1.
    first_drops = [first_index(np.diff(captures[:, i].astype(int), prepend=0) < 0) for i in range(2)]
    if any(i is None for i in (*first_slips, *first_drops)):
        return dict(pass_gate=False, first_slip_samples=first_slips, first_release_samples=first_drops)
    first_slip = min(first_slips)
    first_drop = min(first_drops)
    initiating_pad = int(np.argmin(first_slips))
    hold = (t >= trace["action_change_s"] - .20) & (t < trace["action_change_s"])
    torque = trace["wrench"][:, 3:] - trace["wrench"][hold, 3:].mean(axis=0)
    torque_marker = float(np.linalg.norm(torque[first_slip:first_drop], axis=1).max(initial=0.))
    pull = -trace["wrench"][:, 2]
    slope_prev = slope_between(t, pull, t[first_slip] - .080, t[first_slip])
    slope_after = slope_between(t, pull, t[first_drop] - .030, t[first_drop] + .030)
    return dict(first_slip_s=t[first_slips].tolist(), first_release_s=t[first_drops].tolist(),
                initiating_finger=initiating_pad, torque_precursor_Nm=torque_marker,
                pull_slope_pre_slip_N_per_s=slope_prev, pull_slope_at_first_release_N_per_s=slope_after,
                stage1_markers="A sustained unilateral path or the first finger losing load precedes full detachment",
                pass_gate=first_slips[0] != first_slips[1] and first_drops[0] != first_drops[1]
                and first_slip < first_drop and initiating_pad == int(np.argmin(first_drops)) and torque_marker >= .0001
                and slope_prev is not None and slope_after is not None and slope_after < slope_prev)


def detachment_metrics(trace, dt, observable, reference):
    t = trace["time_s"]
    change = trace["action_change_s"]
    active = trace["contact_state"][:, :, C.CAPTURE] == 1.
    index = first_index((t > change) & ~active.any(axis=1))
    slipping = trace["slip_rate_mps"].max(axis=1) > 1e-7
    if index is None:
        return dict(pass_gate=False, detached=False, stage1_reference=reference)
    preceding = (t > change) & (t < t[index])
    slip_s = longest(preceding & slipping, dt)
    max_support = float(trace["support_N"][preceding].max(initial=0.))
    norm = np.linalg.norm(trace["wrench"][:, :3], axis=1)
    peak_pulse = float(np.diff(norm[(t >= change) & (t <= t[index])]).max(initial=0.))
    measured = float(trace[observable][index])
    residual = min((abs(measured - x) for x in reference), default=None)
    return dict(detached=True, end_s=float(t[index]), value=measured, stage1_reference=reference,
                absolute_difference_to_nearest_reference=residual, slip_s=slip_s,
                max_support_before_release_N=max_support, largest_positive_step_N=peak_pulse,
                pass_gate=bool(both_captured(trace)[t <= change][-1]) and slip_s >= 3 * dt
                and max_support == 0. and peak_pulse <= .5)


def gather_slope(sim):
    p = sim.proxy.params
    pads = p.S_bag.new_zeros((1, 2, 3))
    pair = torch.ones_like(p.mu_eff)
    action = ContactInputs(pads, pads, pads, pads, pads[:, 0], .8 * pair, .0075 * pair, pair * 0., pair * 0.)
    state = sim.proxy.initial_state()
    distance, slack = [], []
    for idx in range(round(.100 / sim.dt)):
        state = sim.proxy.step(state, action).state
        if idx * sim.dt >= .020:
            distance.append(2. * (idx + 1) * .0075 * sim.dt)
            slack.append(state[:, :, C.SLACK].sum().item())
    return float(np.polyfit(distance, slack, 1)[0])


def verify(config, dt):
    materials = config["materials"]
    archived = json.loads(CURVES.read_bytes())
    release_refs = {row["material"]: row["release"]["first_final_contact_loss_sample"]["actual_gap_m"]
                    for row in archived["cases"] if row["release"]}
    release_refs["pe"] = release_refs["pe_like"]
    top, hover, release, load = {}, {}, {}, {}
    figures = {key: {} for key in ("top", "hover", "feed", "release", "load")}
    for material, entry in materials.items():
        nominal = entry["nominal"]
        sim = MacroParcelSim.from_nominal(nominal, dt)
        for pressure in (.3, .5, .8):
            trace = sim.run_top_press(pressure).channels
            top[f"{material}/{pressure:g} N"] = empty_result(trace)
            if pressure == .8:
                figures["top"][material] = trace
        result = sim.run_side_grasp_and_hover()
        figures["hover"][material] = result.channels
        hover[material] = side_metrics(result.channels, nominal, entry["anchors"]["k_N_per_m"], dt)
        hover[material]["wall_s"] = result.elapsed_s

        grip, span = nominal["probe_grip_N"], nominal["probe_opening_span_m"]
        trace = sim.run_controlled_release(grip_N=grip, opening_span_m=span).channels
        figures["release"][material] = trace
        release[material] = detachment_metrics(trace, dt, "gap_m", [float(release_refs[material])])
        axis = "tangential" if material == "bubble" else "vertical"
        trace = sim.run_suspended_load_ramp(grip_N=grip, axis=axis).channels
        figures["load"][material] = trace
        reference = entry["anchors"]["release_applied_load_by_axis"][axis]
        if material == "bubble":
            slow, fast = reference  # The frozen manifest lists the 5 and 20 N/s runs in this order.
            reference = [slow]
        load[material] = detachment_metrics(trace, dt, "external_load_N", [row["added_N"] for row in reference])
        load[material]["direction"] = axis
        load[material]["rate_N_per_s"] = 5.
        if material == "bubble":
            load[material]["stage1_20_N_per_s_reference"] = fast["added_N"]

    pe = materials["pe"]["nominal"]
    low_slack_m = config["geometry"]["opposing_axis_budget_by_group_m"]["0.90"][0]
    starved = MacroParcelSim.from_nominal(pe, dt, slack_m=low_slack_m).run_side_grasp_and_hover(feed_depth_m=.035)
    low_slack = dict(slack_budget_m=low_slack_m, capture_threshold_m=pe["proxy"]["s_crit"], **empty_result(starved.channels))
    sim = MacroParcelSim.from_nominal(pe, dt)
    feed = {}
    for depth in (.0015, .0075, .01125, .015531134179):
        label = f"{depth * 1000:.2f} mm"
        trace = figures["hover"]["pe"] if depth > .015 else sim.run_side_grasp_and_hover(feed_depth_m=depth).channels
        figures["feed"][label] = trace
        feed[label] = feed_metrics(trace, dt)
    empty, shallow, intermediate, full = feed.values()
    feed_trend = empty["pass_gate"] and 0. < shallow["bilateral_unsupported_s"] < intermediate["bilateral_unsupported_s"]
    feed_trend = feed_trend and intermediate["bilateral_unsupported_s"] < full["bilateral_unsupported_s"]
    feed_trend = feed_trend and shallow["first_release_fingers"] == [1] and len(intermediate["first_release_fingers"]) == 1
    for row in feed.values():
        del row["pass_gate"]  # Zero-capture is only the expected outcome at the shallowest depth.

    ecc = MacroParcelSim.from_nominal(pe, dt, com_offset=(.003, 0., 0.))
    asymmetric = ecc.run_suspended_load_ramp(grip_N=pe["probe_grip_N"]).channels
    figures["asymmetric"] = asymmetric
    asym = asymmetric_metrics(asymmetric)
    alpha = gather_slope(sim)
    slope_pass = .60 <= alpha <= .75
    release_load_pass = all(row["pass_gate"] for row in (*release.values(), *load.values()))
    regimes = {
        "1_top_press": dict(cases=top, stage1_bilateral_positive_pull=False,
                            pass_gate=all(row["pass_gate"] for row in top.values())),
        "2_low_slack": low_slack,
        "3_side_hover": dict(materials=hover, pass_gate=all(row["pass_gate"] for row in hover.values())),
        "4_feed_depth": dict(cases=feed, stage1_feed_diagnostics=config["pe_feed_depth_diagnostics"], pass_gate=feed_trend),
        "5_unilateral_release": asym,
        "6_gather_and_detachment": dict(pe_slope_m_per_m=alpha, pe_reference_interval=[.60, .75],
                                       controlled_opening=release, load_ramp=load, pass_gate=slope_pass and release_load_pass),
    }
    timings = {mat: dict(wall_s=row["wall_s"], multiplier=[ref / row["wall_s"] for ref in BASELINE_S])
               for mat, row in hover.items()}
    return dict(machine=platform.node(), torch_version=torch.__version__, device="cpu", batch=1, dt_s=dt,
                config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                side_episode_s=4.77, reference_warp_wall_s=BASELINE_S, cpu_side_episode_timings=timings,
                regimes=regimes, gate_A_pass=all(row["pass_gate"] for row in regimes.values())), figures


def plot(figures, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
    for key, title, xlabel, ylabel, getter, ax in (
        ("top", "Top press (0.8 N)", "Time (s)", "Pull (N)",
         lambda d: (d["time_s"], d["tension_N"].sum(axis=1)), axes[0, 0]),
        ("hover", "Side grasp", "Time (s)", "Clearance (mm)",
         lambda d: (d["time_s"], d["table_clearance_m"] * 1000.), axes[0, 1]),
        ("feed", "PE insertion depth", "Time (s)", "Captured fingers",
         lambda d: (d["time_s"], d["contact_state"][:, :, C.CAPTURE].sum(axis=1)), axes[1, 0]),
        ("release", "Controlled opening", "Jaw opening (mm)", "Tension (N)",
         lambda d: (d["gap_m"] * 1000., d["tension_N"].sum(axis=1)), axes[2, 0]),
        ("load", "Suspended load ramp", "Applied load (N)", "Tension (N)",
         lambda d: (d["external_load_N"], d["tension_N"].sum(axis=1)), axes[2, 1]),
    ):
        for label, data in figures[key].items():
            selected = data["time_s"] >= data["action_change_s"] if key in ("release", "load") else slice(None)
            x, y = getter(data)
            ax.plot(x[selected], y[selected], label=label)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize="small")
        ax.grid(alpha=.2)
    d = figures["asymmetric"]
    ax = axes[1, 1]
    active = (d["time_s"] >= d["action_change_s"]) & (d["time_s"] <= d["action_change_s"] + 2.)
    t = d["time_s"][active] - d["action_change_s"]
    ax.plot(t, d["tension_N"][active, 0], label="left tension")
    ax.plot(t, d["tension_N"][active, 1], label="right tension")
    twin = ax.twinx()
    twin.plot(t, d["wrench"][active, 4] * 1000., "C2--", label="wrist torque")
    twin.set_ylabel("Wrist torque (mN·m)")
    ax.set(title="Eccentric PE load", xlabel="Time from ramp (s)", ylabel="Tension (N)")
    ax.legend(loc="upper left", fontsize="small")
    twin.legend(loc="upper right", fontsize="small")
    ax.grid(alpha=.2)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "test/runs/stage2_freeze")
    parser.add_argument("--dt", type=float, default=.002, choices=(.001, .002))
    args = parser.parse_args()
    if any((args.output / filename).exists() for filename in ("alignment_report.json", "alignment_waveforms.png")):
        parser.error(f"An alignment result already exists in {args.output}")
    torch.set_num_threads(1)
    with torch.inference_mode():
        report, figures = verify(json.loads(CONFIG.read_bytes()), args.dt)
    args.output.mkdir(parents=True, exist_ok=True)
    plot(figures, args.output / "alignment_waveforms.png")
    (args.output / "alignment_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, row in report["regimes"].items():
        print(f"{name}: {'PASS' if row['pass_gate'] else 'FAIL'}")
    print(args.output / "alignment_report.json")
    return 0 if report["gate_A_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

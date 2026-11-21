"""Scan the frozen PE/Stage 2.5 physical boundaries on three fixed grids."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.low_order_parcel.lumped_contact_proxy import ContactState, LumpedContactProxy, ProxyParams
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim, ParcelAction
from src.low_order_parcel.unified_payload import DT, PayloadParamsX, UnifiedPayloadTemplate
from test_unified_payload import CONFIG, STAGE2


def wide_grid(a, b, device):
    x, y = np.meshgrid(a, b)
    return [torch.as_tensor(v.ravel()[:, None], dtype=torch.float32, device=device) for v in (x, y)]


def build(preset, x, **changes):
    nominal = STAGE2["materials"]["pe"]["nominal"]["proxy"]
    params = ProxyParams.from_scalars(nominal, dtype=x.dtype, device=x.device)
    proxy = LumpedContactProxy(ProxyParams(**{k: v.expand(len(x), -1) for k, v in vars(params).items()}), DT)
    changes.setdefault("bag_half_extents", x.new_tensor([[.085, .105, .050]]).expand(len(x), -1))
    payload = PayloadParamsX.from_presets(CONFIG, [preset] * len(x), dtype=x.dtype, device=x.device, **changes)
    return MacroParcelSim(proxy, payload=UnifiedPayloadTemplate(payload))


def top_action(i, like, z, press_z, feedback, previous):
    time = i * DT
    if time < 1.:
        z = z + DT * (.006 * (feedback - 20.)).clamp(-.13, .065)
        press_z = z
    else:
        lift = min(1., (time - 1.) / 1.3)
        z = press_z + .5 * (1. - math.cos(math.pi * lift)) * (.315 - press_z)
    close = min(1., max(0., (time - .30) / .65))
    side = like.new_tensor([[-1., 1.]])
    x = ((.056 - .043 * close) * side).expand(len(like), -1)
    zeros = torch.zeros_like(x)
    pos = torch.stack((x, zeros, z.expand_as(x)), dim=-1)
    velocity = torch.zeros_like(pos) if previous is None else (pos - previous) / DT
    action = ParcelAction(pos, velocity, torch.full_like(x, 6.), (-velocity[:, :, 0] * side).clamp_min(0.),
                          zeros, zeros, pos[:, 0] * 0., z * 0.)
    return action, z, press_z


def side_action(i, z, previous):
    t = min(i * DT, .34)
    side = z.new_tensor([[-1., 1.]])
    x = ((.030 - .025 * t / .34) * side).expand(len(z), -1)
    y = torch.full_like(x, -.112 + .12 * t)
    pos = torch.stack((x, y, z.expand_as(x)), dim=-1)
    v = torch.zeros_like(pos) if previous is None else (pos - previous) / DT
    return ParcelAction(pos, v, torch.full_like(x, 6.), (-v[:, :, 0] * side).clamp_min(0.),
                        (y + .105).clamp_min(0.), torch.ones_like(x), pos[:, 0] * 0., z * 0.)


def phase_b_actions(a, masses):
    column = build("rigid_box", masses, total_mass=masses)
    # The shared probe takes one feed depth and sums its two pad contributions.
    # Build one mass column per feed depth, then restore meshgrid row order.
    profiles = [column.probe_actions("side", feed_depth_m=float(depth))[3] for depth in a]
    return ParcelAction(*(torch.stack(values, dim=2).flatten(1, 2)
                          for values in zip(*profiles, strict=True)))


def longest_updates(state, result, current, best):
    both = (state.contacts[:, :, ContactState.CAPTURE] > .5).all(dim=1)
    unsupported = (result.table_clearance[:, 0] > .001) & (result.support_N[:, 0] == 0.)
    stable = both & unsupported & (state.velocity.norm(dim=1) < .020)
    current = torch.where(stable, current + 1, 0)
    return current, torch.maximum(current, best)


@torch.inference_mode()
def scan(phase, a, b, device):
    x, y = wide_grid(a, b, device)
    if phase == "A":
        sim = build("fluffy_pillow", x, fill_ratio=x, max_indent_ratio=y, total_mass=torch.full_like(x, .24))
        steps = 2300
    elif phase == "B":
        sim = build("rigid_box", x, total_mass=y)
        actions = phase_b_actions(a, y[::len(a)])
        steps = len(actions.pinch_position)
    else:
        sim = build("rigid_box", x, table_friction=x, total_mass=torch.full_like(x, .065))
        steps = 245
    state = sim.initial_state()
    z = torch.full_like(x, .115)
    feedback = torch.zeros_like(z)
    previous = None
    press_z = z
    capture = torch.zeros((len(x), 2), dtype=torch.bool, device=device)
    bilateral = torch.zeros(len(x), dtype=torch.bool, device=device)
    clearance = torch.zeros_like(x[:, 0])
    feed = torch.zeros_like(clearance)
    elapsed = torch.zeros_like(clearance, dtype=torch.long)
    best = torch.zeros_like(elapsed)
    nonfinite = torch.zeros_like(elapsed, dtype=torch.bool)
    for i in range(steps):
        if phase == "A":
            action, z, press_z = top_action(i, x, z, press_z, feedback, previous)
        elif phase == "B":
            action = ParcelAction(*(value[i] for value in actions))
        else:
            action = side_action(i, 2. * sim.payload.half_height + y, previous)
        previous = action.pinch_position
        result = sim.step(state, action)
        state = result.state
        feedback = result.wrist_wrench[:, 2:3]
        clearance = torch.maximum(clearance, result.table_clearance[:, 0])
        active = state.contacts[:, :, ContactState.CAPTURE] > .5
        capture |= active
        bilateral |= active.all(dim=1)
        feed = torch.maximum(feed, state.contacts[:, :, ContactState.FEED_DEPTH].amin(dim=1))
        elapsed, best = longest_updates(state, result, elapsed, best)
        nonfinite |= ~torch.isfinite(state.payload.pos).all(dim=(1, 2))
    arrays = dict(clearance_m=clearance, max_net_feed_m=feed, captured_left=capture[:, 0],
                  captured_right=capture[:, 1], bilateral_capture=bilateral, hover_steps=best, nonfinite=nonfinite)
    return {key: value.cpu().numpy().reshape(len(b), len(a)) for key, value in arrays.items()}


def check(phase, a, b, grids):
    left, right = grids["captured_left"], grids["captured_right"]
    both = grids["bilateral_capture"]
    hover = grids["hover_steps"]
    result = {"nonfinite_cells": int(grids["nonfinite"].sum())}
    if phase == "A":
        success = hover >= round(.20 / DT)
        result |= dict(hard_captures=int((left[0] | right[0]).sum()),
                       hard_lifts=int((grids["clearance_m"][0] > 1e-8).sum()),
                       reverse_edges=int((success[:-1] & ~success[1:]).sum()),
                       successful_cells=int(success.sum()), fill_columns_without_success=int((~success.any(0)).sum()))
        result["passed"] = not any(result[k] for k in ("nonfinite_cells", "hard_captures", "hard_lifts",
                                                       "reverse_edges", "fill_columns_without_success"))
    elif phase == "B":
        success = hover >= round(.20 / DT)
        result |= dict(heavier_longer_edges=int((np.diff(hover, axis=0) > 0).sum()),
                       deeper_loses_success_edges=int((success[:, :-1] & ~success[:, 1:]).sum()),
                       successful_cells=int(success.sum()),
                       max_heavier_advantage_s=float(np.diff(hover, axis=0).max() * DT))
        result["passed"] = not any(result[k] for k in ("nonfinite_cells", "heavier_longer_edges",
                                                       "deeper_loses_success_edges")) and bool(success.any())
    else:
        low = np.ix_(b <= -.005, a <= .20)
        high = np.ix_(b >= .005, a <= .20)
        feed = grids["max_net_feed_m"]
        h_min = max(np.atleast_1d(STAGE2["materials"]["pe"]["nominal"]["proxy"]["h_min"]))
        result |= dict(low_region_cells=int(both[low].size), low_region_any_captures=int((left[low] | right[low]).sum()),
                       low_region_net_feed_above_gate=int((feed[low] >= h_min).sum()),
                       raised_region_cells=int(both[high].size), raised_region_no_bilateral_capture=int((~both[high]).sum()),
                       raised_region_net_feed_below_gate=int((feed[high] < h_min).sum()))
        result["passed"] = not any(result[k] for k in ("nonfinite_cells", "low_region_any_captures",
                                                       "low_region_net_feed_above_gate", "raised_region_no_bilateral_capture",
                                                       "raised_region_net_feed_below_gate"))
    return result


def output_phase(output, phase, a, b, grids, names):
    with (output / f"phase-{phase}.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*names, *grids, "hover_s"])
        for j, y in enumerate(b):
            for i, x in enumerate(a):
                writer.writerow([x, y, *(grid[j, i] for grid in grids.values()), DT * grids["hover_steps"][j, i]])
    fields = {"A": [("Maximum table clearance (mm)", 1000. * grids["clearance_m"]),
                    ("Stable bilateral capture", grids["hover_steps"] * DT >= .20)],
              "B": [("Longest stable unsupported hover (s)", grids["hover_steps"] * DT)],
              "C": [("Maximum bilateral net feed (mm)", 1000. * grids["max_net_feed_m"]),
                    ("Bilateral capture", grids["bilateral_capture"])]}
    fig, axes = plt.subplots(1, len(fields[phase]), figsize=(6.3 * len(fields[phase]), 4.6), squeeze=False,
                             layout="constrained")
    for ax, (title, values) in zip(axes[0], fields[phase], strict=True):
        image = ax.pcolormesh(a, b, values, shading="nearest", cmap="viridis")
        fig.colorbar(image, ax=ax)
        ax.set(xlabel=names[0], ylabel=names[1], title=title)
        if phase == "B":
            ax.set_yscale("log")
    fig.savefig(output / f"phase-{phase}.png", dpi=170)
    plt.close(fig)


def main(output, n, device):
    output.mkdir(parents=True, exist_ok=False)
    axes = dict(A=(np.linspace(.25, .95, n), np.linspace(0., .78, n), ("fill_ratio", "indent_ratio")),
                B=(np.linspace(.001, .025, n), np.geomspace(.03, 2., n), ("feed_m", "mass_kg")),
                C=(np.linspace(.04, .95, n), np.linspace(-.020, .015, n), ("table_mu", "entry_height_m")))
    report = {}
    for phase, (a, b, names) in axes.items():
        grids = scan(phase, a, b, device)
        report[phase] = check(phase, a, b, grids)
        output_phase(output, phase, a, b, grids, names)
        print(phase, json.dumps(report[phase]), flush=True)
    (output / "phase-summary.json").write_text(json.dumps(dict(grid=n, device=device, phases=report), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.grid < 2:
        parser.error("--grid must be at least 2")
    torch.set_num_threads(1)
    main(args.output, args.grid, args.device)

"""Apply a measured downward load to an already suspended whole bag in MuJoCo."""

import argparse
import csv
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path[:0] = [str(RUN.parents[3] / "bag_model"), str(RUN.with_name("rss2027-stage1-bubble-free-release-r1"))]
import physical_parcel_yam as backend  # noqa: E402
from record_physical_parcel import penetration_peak, release_contacts  # noqa: E402
from record_support_withdrawal import FLOOR, FLOOR_RETRACTION_M, STATE_SPEC, SUPPORT  # noqa: E402
from record_support_withdrawal import integration_state, support_pose, world_with_movable_support  # noqa: E402
from rss2027_micropatch import sustained_starts  # noqa: E402
from rss2027_release_traction import inner_loads  # noqa: E402
from yam_platform.protocol import run, tips  # noqa: E402

BASE_HOLD_S = .20
LOAD_RATE_N_S = 20.
MAX_LOAD_N = 12.
SAMPLE_PERIOD_S = .002


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=RUN / "config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--load-rate", type=float, default=LOAD_RATE_N_S, help="applied force ramp in N/s")
    parser.add_argument("--withdraw-supports", action="store_true", help="lower the unloaded shelf and floor at loading")
    parser.add_argument("--withdraw-before-preload", action="store_true",
                        help="withdraw at the first unloaded bilateral hold, then start the preload clock")
    parser.add_argument("--disable-legacy-pad-edges", action="store_true",
                        help="test only the standard rounded pad without overlapping legacy edges")
    args = parser.parse_args()
    config = backend.Config(**json.loads(args.config.read_bytes()))
    if config.mode != "rigid" or config.pinch_coeff or not config.pickup_only or config.pickup_release_seconds:
        parser.error("the ramp requires an unassisted pickup of a single rigid payload")
    if config.lift_hold_seconds <= BASE_HOLD_S:
        parser.error("the hold must cover preloading and the applied-force ramp")
    if not np.isfinite(args.load_rate) or args.load_rate <= 0.:
        parser.error("--load-rate must be finite and positive")
    if args.withdraw_supports and config.support_height <= 0.:
        parser.error("support withdrawal requires the raised shelf and separate floor")
    if args.withdraw_before_preload and not args.withdraw_supports:
        parser.error("--withdraw-before-preload requires --withdraw-supports")
    if args.smoke:
        config = replace(config, smoke_seconds=.02)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "recorder_source.py").write_bytes(Path(__file__).read_bytes())
    hold_seen = False
    support_clear = not args.withdraw_supports
    load_start = stable_since = next_readout = None
    rows, raw, peaks = [], [], []
    previous_phase = None
    scene = backend.scene

    def withdraw(sim, time, force):
        table, floor = support_pose(sim, SUPPORT), support_pose(sim, FLOOR)
        clearance = float(sim.d.flexvert_xpos[:, 2].min() - sim.m.flex_radius[0] - floor[2])
        failure, health = backend.check_state(sim)
        if (failure or force["support_N"] != 0. or clearance <= 0.
                or health["support_geometry"]["intersecting_triangles"]):
            return False
        before = [table.tolist(), floor.tolist()]
        table[2] -= FLOOR_RETRACTION_M
        floor[2] -= FLOOR_RETRACTION_M
        event = dict(time_s=time, support_N=force["support_N"], floor_clearance_m=clearance,
                     support_geometry=health["support_geometry"], before_m=before, after_m=[table.tolist(), floor.tolist()])
        with (args.output / "withdrawal.json").open("x") as stream:
            stream.write(json.dumps(event, indent=2) + "\n")
        return True

    def with_load(cfg):
        robot = scene(cfg)
        sim = robot.sim
        if args.disable_legacy_pad_edges:
            names = [f"{side}_pad_edge_{edge}" for side in ("left", "right") for edge in ("a", "b")]
            indices = [sim.m.geom(name).id for name in names]
            disabled = {name: [int(sim.m.geom_contype[i]), int(sim.m.geom_conaffinity[i])]
                        for name, i in zip(names, indices, strict=True)}
            if not all(a or b for a, b in disabled.values()):
                raise ValueError("Expected all four upstream pad edges to have enabled collision")
            sim.m.geom_contype[indices] = 0
            sim.m.geom_conaffinity[indices] = 0
            (args.output / "disabled-legacy-pad-edges.json").write_text(json.dumps(disabled, indent=2) + "\n")
        step = sim.step
        payload = int(sim.payload_bodies[0])

        def load_step():
            nonlocal load_start, stable_since, next_readout, support_clear
            if not hold_seen:
                step()
                return
            time = float(sim.d.time)
            load = min(MAX_LOAD_N, max(0., args.load_rate * (time - load_start))) if load_start is not None else 0.
            # The shared controller clears external forces before calling step.
            sim.d.xfrc_applied[payload, 2] = -load
            measured_gap = float(robot.gap())
            step()
            if next_readout is not None and time < next_readout - cfg.dt / 2:
                return
            next_readout = time + SAMPLE_PERIOD_S
            # The split step leaves constraint observations from its start pose.
            force = backend.measure(robot)
            contacts = release_contacts(robot)
            normal, sliding = inner_loads(contacts)
            tension = -sum(reaction[2] for reaction in force["force_on_fingers_N"])
            held = min(normal) > .001 and tension > .001 and force["support_N"] <= .05
            stable_since = (time if stable_since is None else stable_since) if held else None
            if args.withdraw_before_preload and held and not support_clear and withdraw(sim, time, force):
                support_clear = True
                stable_since = time + cfg.dt
            if load_start is None and stable_since is not None and time - stable_since >= BASE_HOLD_S - cfg.dt / 2:
                if support_clear or (not args.withdraw_before_preload and withdraw(sim, time, force)):
                    load_start = time + cfg.dt
                    print(json.dumps(dict(load_start_s=load_start)), flush=True)
            candidate = bool(held and load > 0. and (sliding >= .5 * normal).all())
            rows.append(dict(t_s=time, applied_load_N=load, actual_gap_m=measured_gap, tension_N=tension,
                             left_inner_N=float(normal[0]), right_inner_N=float(normal[1]),
                             left_sliding_N=float(sliding[0]), right_sliding_N=float(sliding[1]),
                             native_bilateral_sliding=candidate, bilateral_held=bool(held), support_N=force["support_N"],
                             tip_z_m=float(tips(robot)[:, 2].mean()), payload_com_z_m=float(sim.com()[2]),
                             max_finger_contact_N=max(force["contact_magnitude_each_N"])))
            raw.append(dict(t_s=time, contacts=contacts))

        sim.step = load_step
        return robot

    def observe(_robot, row):
        nonlocal hold_seen, previous_phase
        if row["phase"] != previous_phase:
            previous_phase = row["phase"]
            print(json.dumps(dict(t_s=row["t"], phase=previous_phase, support_N=row["support_N"])), flush=True)
        hold_seen = row["phase"] == "lift_hold"

    def observe_peak(robot, phase):
        peaks.append(penetration_peak(robot, phase))

    hooks = dict(scene=with_load)
    if args.withdraw_supports:
        hooks.update(parcel_world=partial(world_with_movable_support, lower_floor=True),
                     record_state=integration_state, STATE_SPEC=STATE_SPEC)
    with patch.multiple(backend, create=True, **hooks):
        result = run(backend, config, args.output / "trial", on_sample=observe, on_penetration_peak=observe_peak)
    events = sustained_starts(np.array([row["t_s"] for row in rows]),
                              [row["native_bilateral_sliding"] for row in rows])
    metric = dict(outcome=result["outcome"], numerical_ok=not result["warnings"] and result["peaks"]["penetration_mm"] <= 1.,
                  preloading_s=BASE_HOLD_S, ramp_rate_N_s=args.load_rate, ramp_limit_N=MAX_LOAD_N,
                  load_start_s=load_start, samples=len(rows),
                  max_added_load_N=max((row["applied_load_N"] for row in rows), default=0.),
                  max_observed_tension_N=max((row["tension_N"] for row in rows), default=0.),
                  first_sustained_bilateral_sliding=rows[events[0]] if events else None)
    if args.withdraw_before_preload:
        metric["withdrawal_before_preload"] = True
    for name, data in (("summary.json", result), ("load-check.json", metric), ("penetration-peaks.json", peaks),
                       ("load-contacts.json", raw)):
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")
    if rows:
        with (args.output / "load-curve.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(metric), flush=True)


if __name__ == "__main__":
    main()

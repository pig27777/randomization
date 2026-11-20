"""Repeat the PE load test with feedback on measured inner normal force."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path[:0] = [str(RUN.with_name(name)) for name in ("rss2027-stage1-kraft-held-load-step-r1",
                                                     "rss2027-stage1-material-load-ramps-r1")]
import replay_held_load as replay  # noqa: E402
from analyze_pe_parameters import intervals, local_contact_motion  # noqa: E402
from yam_platform.protocol import tips  # noqa: E402

NORMAL_N = 5.
BAND_N = (4.5, 5.5)
RATE_N_S = 5.
FIELDS = ("time_s", "added_N", "tension_N", "support_N", "gap_m", "gripper_command_m", "left_normal_N",
          "right_normal_N", "left_speed_m_s", "right_speed_m_s", "sliding", "in_force_band", "penetration_mm",
          "tip_payload_separation_m")
COLUMN = {name: i for i, name in enumerate(FIELDS)}


def candidate(rows):
    time = rows[:, COLUMN["time_s"]]
    normal = rows[:, [COLUMN["left_normal_N"], COLUMN["right_normal_N"]]].sum(axis=1)
    tension = rows[:, COLUMN["tension_N"]]
    eligible = rows[:, COLUMN["sliding"]].astype(bool) & rows[:, COLUMN["in_force_band"]].astype(bool)
    events = [(i, j) for i, j in intervals(eligible) if time[j - 1] - time[i] >= .010 - 1e-9]
    if not events:
        longest = max((time[j - 1] - time[i] for i, j in intervals(eligible)), default=0.)
        return dict(longest_sliding_in_normal_band_s=float(longest))
    first, end = events[0]
    ratio = tension[first] / normal[first]
    overload = tension - ratio * normal
    overload[first] = 0.
    windows = [(i, j) for i, j in intervals(eligible & (overload > 0.) & (time > time[first]))
               if time[j - 1] - time[i] >= .010 - 1e-9]
    report = dict(first_slip_time_s=float(time[first]), first_slip_force_N=float(tension[first]),
                  force_ratio=float(ratio), first_slip_duration_s=float(time[end - 1] - time[first]),
                  beta_candidate_m_per_N_s=None)
    if windows:
        a, b = max(windows, key=lambda pair: time[pair[1] - 1] - time[pair[0]])
        velocity = rows[a:b, [COLUMN["left_speed_m_s"], COLUMN["right_speed_m_s"]]].mean(axis=1)
        x = overload[a:b]
        coeff = float((x @ velocity) / (x @ x))
        report.update(beta_candidate_m_per_N_s=coeff, beta_interval_s=[float(time[a]), float(time[b - 1])],
                      beta_rms_residual_m_s=float(np.sqrt(np.mean((velocity - coeff * x)**2))))
    return report


def run(dt, duration, is_smoke):
    robot, frame, fingerprint = replay.held_scene()
    sim = robot.sim
    sim.m.opt.timestep = dt
    fixed_arm = sim.d.ctrl[robot.act].copy()
    actuator = robot.gripper
    low, high = sim.m.actuator_ctrlrange[actuator]
    payload = int(sim.payload_bodies[0])
    filtered = NORMAL_N
    start = float(sim.d.time)
    stable_since = load_start = loss_since = None
    outcome = "smoke_complete" if is_smoke else "load_budget"
    rows = []
    for step in range(round(duration / dt)):
        t = float(sim.d.time) - start
        loading = load_start is not None
        load = float(np.clip(RATE_N_S * (t - load_start), 0., 12.)) if loading else 0.
        gap, command = robot.gap(), sim.d.ctrl[actuator]
        sim.d.xfrc_applied[payload, 2] = -load
        sim.step()
        measured = replay.backend.measure(robot)
        normal, sliding, speed = local_contact_motion(replay.release_contacts(robot))
        tension = -sum(force[2] for force in measured["force_on_fingers_N"])
        pressure = normal.sum()
        in_band = BAND_N[0] <= pressure <= BAND_N[1]
        supported = measured["support_N"]
        depth = measured["penetration_mm"]
        separation = float(tips(robot)[:, 2].mean() - sim.com()[2])
        bilateral = normal.min() > .001 and tension > .001 and supported <= .05
        slipping = bool(load > 0. and bilateral and (sliding >= .5 * normal).all())
        rows.append((t, load, tension, supported, gap, command, *normal, *speed, slipping, in_band, depth, separation))
        if depth > 1. or sim.d.warning.number.any():
            outcome = "native_penetration_or_warning"
            break
        if step % round(.002 / dt) == 0:
            failure, _ = replay.backend.check_state(sim)
            if failure:
                outcome = failure
                break
        if loading:
            lost = max(measured["contact_magnitude_each_N"]) <= .05
            loss_since = (t if loss_since is None else loss_since) if lost else None
            if loss_since is not None and t - loss_since >= .05 - dt / 2:
                outcome = "detached"
                break
            if t - load_start >= 2.45:
                break
        else:
            stable_since = (t if stable_since is None else stable_since) if bilateral and in_band else None
            if stable_since is not None and t - stable_since >= .20 - dt / 2:
                load_start = t + dt
            elif t >= .60:
                outcome = "no_qualified_preload"
                break
        filtered += dt / (.005 + dt) * (pressure - filtered)
        speed_command = np.clip(.0005 * (filtered - NORMAL_N), -.004, .004)
        sim.d.ctrl[actuator] = np.clip(command + dt * speed_command, low, high)
    np.testing.assert_array_equal(sim.d.ctrl[robot.act], fixed_arm)
    replay.backend.refresh(sim)
    failure, health = replay.backend.check_state(sim)
    if failure:
        outcome = failure
    rows = np.asarray(rows)
    report = dict(dt_s=dt, source_frame=frame, source_sha256=fingerprint, normal_target_N=NORMAL_N,
                  normal_band_N=BAND_N, load_start_s=load_start, outcome=outcome, samples=len(rows),
                  peak_penetration_mm=float(rows[:, COLUMN["penetration_mm"]].max()),
                  peak_support_N=float(rows[:, COLUMN["support_N"]].max()),
                  force_range_N=(rows[:, 6:8].sum(axis=1).min().item(), rows[:, 6:8].sum(axis=1).max().item()),
                  final_health=health, effective_parameters=candidate(rows))
    return report, rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    replay.SOURCE = RUN.with_name("rss2027-stage1-material-load-ramps-r1") / "pe/recording"
    replay.CHECKPOINT = RUN.with_name("rss2027-stage1-pe-held-load-step-r1") / "held-state.npz"
    args.output.mkdir(exist_ok=False)
    for name, dt in (("25", .000025), ("12p5", .0000125)):
        report, rows = run(dt, .02 if args.smoke else 3.06, args.smoke)
        stem = args.output / f"dt-{name}us"
        np.savez_compressed(stem.with_suffix(".npz"), columns=np.asarray(FIELDS), rows=rows)
        stem.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

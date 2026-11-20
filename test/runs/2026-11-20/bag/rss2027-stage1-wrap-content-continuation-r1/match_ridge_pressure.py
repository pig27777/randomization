"""Compare two gathering meshes at the same observed normal pressure and travel."""

import argparse
from itertools import product
import json
from pathlib import Path

import mujoco
import numpy as np

from record_local_top import protocol
from replay_ridge_mesh import CASE, DT, load_pair
from replay_ridge_window import aperture_observation, backend

TARGET_N = .5
PRELOAD_LIMIT_S = 1.25
CLOSURE_S = .20
SPEED_M_S = .008


class CenteredPress(protocol.NormalPress):
    proportional_gain = .012
    integral_gain = .020
    integral_limit_N_s = .08
    start_band_N = (.48, .52)
    feed_margin_N = .05

    def __init__(self, target_N, surface_z):
        super().__init__(target_N, surface_z)
        self.integral_N_s = 0.

    def move(self, z, tip_z, force_N, support_N, dt, approach):
        original_z = super().move(z, tip_z, force_N, support_N, dt, approach)
        if support_N > self.support_limit_N:
            self.integral_N_s = 0.
            return original_z
        error_N = self.filtered_N - self.target_N
        limit = self.integral_limit_N_s
        integral = min(limit, max(-limit, self.integral_N_s + error_N * dt))
        proportional = self.proportional_gain * error_N
        velocity = proportional + self.integral_gain * integral
        if -.002 <= velocity <= .004 or (velocity > .004 and error_N < 0.) or (velocity < -.002 and error_N > 0.):
            self.integral_N_s = integral
        velocity = proportional + self.integral_gain * self.integral_N_s
        return z + dt * min(.004, max(-.002, velocity))

    def closure_rate(self, nominal_m_s, force_N, support_N):
        margin = min(force_N - .4, .6 - force_N, self.filtered_N - .4, .6 - self.filtered_N)
        return nominal_m_s * min(1., max(0., margin / self.feed_margin_N))


class OpeningServo:
    start_tolerance_m = .00010
    proportional_gain = 1.5
    integral_gain_per_s = 8.
    limit_m = .004

    def __init__(self, actual_m, command_m):
        self.reference_m = actual_m
        self.integral_m = min(self.limit_m, max(-self.limit_m, command_m - actual_m))

    def command(self, actual_m, dt):
        error_m = self.reference_m - actual_m
        limit = self.limit_m
        proportional = self.proportional_gain * error_m
        integral = min(limit, max(-limit, self.integral_m + self.integral_gain_per_s * error_m * dt))
        correction = proportional + integral
        if abs(correction) <= limit or correction * error_m < 0.:
            self.integral_m = integral
        correction = proportional + self.integral_m
        return self.reference_m + min(limit, max(-limit, correction))


def run(robot, dt, preload_limit_s, closure_s, *, centered_control=False, matched_opening=False):
    sim = robot.sim
    m, d = sim.m, sim.d
    m.opt.timestep = dt
    robot.control_period_steps = round(.010 / dt)
    feedforward = d.qfrc_bias[robot.dofs] - d.qfrc_gravcomp[robot.dofs]
    robot.qtarget = d.ctrl[robot.act] - feedforward / m.actuator_gainprm[robot.act, 0]
    robot.kin.qpos[:] = d.qpos
    robot.kin.qpos[robot.qadr] = robot.qtarget
    mujoco.mj_fwdPosition(m, robot.kin)
    target = robot.kin.site_xpos[robot.site].copy()
    gap = 2. * d.ctrl[robot.gripper]
    opening = OpeningServo(robot.gap(), gap) if matched_opening else None
    initial_opening = opening.reference_m if opening else None
    press_type = CenteredPress if centered_control else protocol.NormalPress
    press = press_type(TARGET_N, protocol.tips(robot).mean(axis=0)[2])
    start_low, start_high = CenteredPress.start_band_N if centered_control else (.4, .6)
    load_start = None
    consecutive = 0
    stride = round(.020 / dt)
    closure_stride = round(.010 / dt) if centered_control else stride
    closure_steps = round(closure_s / dt)
    preload_steps = round(preload_limit_s / dt)
    preload, samples = [], []
    pressure_streak_start = None
    outcome = "preload timeout"
    for step in range(preload_steps + closure_steps + 1):
        feedback = backend.measure(robot)
        down = max(0., sum(force[2] for force in feedback["force_on_fingers_N"]))
        support = feedback["direct_support_N"]
        if feedback["penetration_mm"] > 1. or d.warning.number.any():
            outcome = "numerical gate"
            break
        if step % stride == 0:
            failure, health = backend.check_state(sim)
            if failure or health["payload_geometry"]["intersecting_triangles"]:
                outcome = "geometry gate"
                break
        qualified = .4 <= down <= .6 and support <= .05
        tip_z = protocol.tips(robot).mean(axis=0)[2]
        target[2] = press.move(target[2], tip_z, down, support, dt, approach=False)
        in_band = qualified and .4 <= press.filtered_N <= .6
        if load_start is None:
            start_qualified = qualified and start_low <= press.filtered_N <= start_high
            if opening:
                start_qualified = (start_qualified and start_low <= down <= start_high
                                   and abs(robot.gap() - opening.reference_m) <= opening.start_tolerance_m)
            if step % stride == 0:
                preload.append(dict(time_s=float(d.time), pressure_N=down, direct_support_N=support))
                if centered_control:
                    preload[-1]["filtered_pressure_N"] = press.filtered_N
                if opening:
                    preload[-1]["actual_gap_m"] = robot.gap()
                if step % (stride * 25) == 0:
                    print(f"{m.nflexvert} nodes at {step * dt:.3f} s preload, {down:.6f} N", flush=True)
            consecutive = consecutive + 1 if start_qualified else 0
            if consecutive >= round(press.hold_s / dt):
                load_start = step
            elif step >= preload_steps:
                break
        if load_start is not None:
            elapsed = step - load_start
            if centered_control:
                if not in_band:
                    pressure_streak_start = None
                elif pressure_streak_start is None:
                    pressure_streak_start = float(d.time)
            if elapsed % closure_stride == 0:
                observation = aperture_observation(robot, feedback)
                if centered_control:
                    observation.update(filtered_pressure_N=press.filtered_N,
                                       pressure_streak_start_s=pressure_streak_start)
                    if elapsed % (closure_stride * 20) == 0:
                        print(f"{m.nflexvert} nodes at {elapsed * dt:.3f} s closure, {down:.6f} N", flush=True)
                if opening:
                    observation["reference_gap_m"] = opening.reference_m
                samples.append(observation)
            if elapsed == closure_steps:
                outcome = "complete"
                break
            if in_band:
                rate = press.closure_rate(SPEED_M_S, down, support)
                if opening:
                    error_N = max(abs(down - TARGET_N), abs(press.filtered_N - TARGET_N))
                    rate *= min(1., max(0., (.055 - error_N) / .035))
                    opening.reference_m -= rate * dt
                else:
                    gap -= rate * dt
        if opening:
            gap = opening.command(robot.gap(), dt)
        robot.command(target, gap)
        sim.step()
    result = dict(outcome=outcome, preload_s=load_start * dt if load_start is not None else None,
                  preload=preload, samples=samples)
    if opening:
        result["preload_gap_reference_m"] = initial_opening
    return result


def pressure_spans(rows, *, absolute_gap=False):
    origin = 0. if absolute_gap else rows[0]["actual_gap_m"]
    inward = origin - np.asarray([row["actual_gap_m"] for row in rows])
    streaks = [row.get("pressure_streak_start_s", 0.) for row in rows]
    eligible = [.4 <= row["pressure_N"] <= .6 and row["direct_support_N"] <= .05 and streak is not None
                for row, streak in zip(rows, streaks, strict=True)]
    for start, end in np.flatnonzero(np.diff(np.r_[False, eligible, False])).reshape(-1, 2):
        pressure_interruptions = np.not_equal(streaks[start + 1:end], streaks[start:end - 1])
        breaks = np.flatnonzero((np.diff(inward[start:end]) <= 0.) | pressure_interruptions) + start + 1
        bounds = np.r_[start, breaks, end]
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            if b - a >= 3:
                yield inward[a:b], rows[a:b]


def fit_common(runs, *, absolute_gap=False):
    if any(run["outcome"] != "complete" for run in runs):
        return None
    intervals = []
    for traces in product(*(pressure_spans(run["samples"], absolute_gap=absolute_gap) for run in runs)):
        low, high = max(x[0] for x, _ in traces), min(x[-1] for x, _ in traces)
        if high > low:
            intervals.append((low, high, traces))
    if not intervals:
        return None
    low, high, traces = max(intervals, key=lambda interval: interval[1] - interval[0])
    positions = np.linspace(low, high, 5)
    fits = []
    for x, data in traces:
        values = np.interp(positions, x, [row["extra_length_m"] for row in data])
        reference = np.interp(positions, x, [row["reference_excess_m"] for row in data])
        pressure = np.interp(positions, x, [row["pressure_N"] for row in data])
        times = np.interp((low, high), x, [row["time_s"] for row in data])
        duration = float(times[1] - times[0])
        growth = float(values[-1] - values[0])
        reference_growth = float(reference[-1] - reference[0])
        fits.append(dict(duration_s=duration, growth_m=growth, pressure_N=float(pressure.mean()),
                         slope_m_per_m=float(np.polyfit(positions, values, 1)[0]),
                         eligible=duration >= .020 - 1e-9 and growth >= .00025,
                         reference_growth_m=reference_growth,
                         reference_slope_m_per_m=float(np.polyfit(positions, reference, 1)[0]),
                         reference_eligible=duration >= .020 - 1e-9 and reference_growth >= .00025))
    changes = {}
    for field in ("growth_m", "slope_m_per_m", "pressure_N"):
        if fits[0][field]:
            changes[field] = fits[1][field] / fits[0][field] - 1.
    reference_changes = {}
    for field in ("reference_growth_m", "reference_slope_m_per_m"):
        if fits[0][field]:
            reference_changes[field] = fits[1][field] / fits[0][field] - 1.
    passed = (len(changes) == 3 and all(fit["eligible"] for fit in fits)
              and all(abs(change) <= .10 for change in changes.values()))
    reference_passed = (len(reference_changes) == 2 and all(fit["reference_eligible"] for fit in fits)
                        and all(abs(change) <= .10 for change in reference_changes.values())
                        and abs(changes["pressure_N"]) <= .10)
    interval = (dict(common_absolute_gap_interval_m=[float(-high), float(-low)]) if absolute_gap
                else dict(common_travel_interval_m=[float(low), float(high)]))
    return dict(**interval, fits=fits, relative_change=changes, passed=passed,
                reference_relative_change=reference_changes, reference_passed=reference_passed)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--step-us", type=float, choices=(12.5, 25.), default=DT * 1e6)
    parser.add_argument("--preload-limit-s", type=float, default=PRELOAD_LIMIT_S)
    parser.add_argument("--closure-s", type=float, default=CLOSURE_S)
    parser.add_argument("--centered-control", action="store_true")
    parser.add_argument("--matched-opening", action="store_true", help="track the same physical opening from preload")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not np.isfinite(value) or value < .020 for value in (args.preload_limit_s, args.closure_s)):
        parser.error("Preload and closure budgets must be finite and at least 20 ms")
    if args.matched_opening and not args.centered_control:
        parser.error("Matching actual openings requires --centered-control")
    case = args.case.resolve()
    dt = args.step_us / 1e6
    coarse, refined, *_ = load_pair(case=case, timestep=dt)
    runs = [run(robot, dt, args.preload_limit_s, args.closure_s, centered_control=args.centered_control,
                matched_opening=args.matched_opening)
            for robot in (coarse, refined)]
    comparison = fit_common(runs)
    result = dict(dt_s=dt, mesh_vertices=[coarse.sim.m.nflexvert, refined.sim.m.nflexvert],
                  pressure_target_N=TARGET_N, preload_limit_s=args.preload_limit_s, closure_s=args.closure_s,
                  runs=runs, comparison=comparison)
    if args.centered_control:
        result["controller"] = dict(proportional_gain_m_per_N_s=CenteredPress.proportional_gain,
                                    integral_gain_m_per_N_s2=CenteredPress.integral_gain,
                                    integral_limit_N_s=CenteredPress.integral_limit_N_s,
                                    start_band_N=CenteredPress.start_band_N,
                                    feed_margin_N=CenteredPress.feed_margin_N,
                                    closure_sample_s=.010)
        result["absolute_gap_comparison"] = fit_common(runs, absolute_gap=True)
    if args.matched_opening:
        result["controller"].update(opening_proportional_gain=OpeningServo.proportional_gain,
                                    opening_integral_gain_per_s=OpeningServo.integral_gain_per_s,
                                    opening_command_limit_m=OpeningServo.limit_m,
                                    opening_start_tolerance_m=OpeningServo.start_tolerance_m,
                                    closing_rate_center_half_width_N=.020,
                                    closing_rate_zero_error_N=.055)
    output = args.output or case / "local-mesh-window/matched-pressure.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print([trial["outcome"] for trial in runs], comparison, flush=True)


if __name__ == "__main__":
    main()

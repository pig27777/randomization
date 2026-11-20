"""Pull the verified paperboard package with a compliant displacement drive."""

import argparse
import json
from unittest.mock import patch

import numpy as np

from replay_paperboard_tangent import estimate_beta, record_flow, replay, scene, setup, summarize

SPEED_M_S = .008
ACCELERATION_S = .10
STIFFNESS_N_M = 2_000.
LIMIT_N = 12.


class PullHead:
    def __init__(self):
        self.origin = None
        self.curve = []

    def apply(self, robot, time, direction):
        position = float(robot.sim.com() @ direction)
        if self.origin is None:
            self.origin = position
        acceleration_time = min(time, ACCELERATION_S)
        desired = SPEED_M_S * (time - acceleration_time + .5 * acceleration_time**2 / ACCELERATION_S)
        actual = position - self.origin
        load = float(np.clip(STIFFNESS_N_M * (desired - actual), 0., LIMIT_N))
        self.curve.append((time, desired, actual, load))
        return load


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--extend", action="store_true")
    args = parser.parse_args()
    case = setup("paperboard")
    reference = json.loads((case / "recording/summary.json").read_bytes())["native_load"]
    baseline = case / "displacement-control"
    output = case / "displacement-extended" if args.extend else baseline
    output.mkdir(exist_ok=False)
    for name, duration in (("smoke", .020), ("recording", 5.0 if args.extend else 3.0)):
        driver = PullHead()
        with patch.object(replay, "held_scene", scene):
            native, rows, transitions = record_flow(reference["dt_s"], duration, "jaw-tangent",
                                                    force_vector=np.asarray(reference["force_direction"]),
                                                    load_profile=driver.apply)
        command = np.asarray(driver.curve)
        np.testing.assert_array_equal(command[:, 0], rows[:, 0])
        np.testing.assert_array_equal(command[:, 3], rows[:, 1])
        np.testing.assert_allclose(command[0], 0., rtol=0., atol=1e-12)
        matched_steps = None
        if args.extend:
            with np.load(baseline / name / "native-load.npz") as archived:
                matched_steps = len(archived["rows"])
                np.testing.assert_array_equal(rows[:matched_steps], archived["rows"])
                np.testing.assert_array_equal(command[:matched_steps], archived["drive"])
        flow = summarize(native, rows, transitions)
        beta = estimate_beta(native, flow)
        directory = output / name
        directory.mkdir(exist_ok=False)
        np.savez_compressed(directory / "native-load.npz", columns=np.asarray(replay.FIELDS), rows=rows,
                            drive_columns=np.asarray(("time_s", "command_travel_m", "payload_travel_m", "force_N")),
                            drive=command)
        report = dict(native_load=native, material_flow=flow, beta_candidate=beta,
                      matched_baseline_steps=matched_steps,
                      drive=dict(speed_m_s=SPEED_M_S, acceleration_seconds=ACCELERATION_S,
                                 stiffness_N_m=STIFFNESS_N_M, maximum_force_N=LIMIT_N))
        (directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(name, native["outcome"], native["peak_penetration_mm"], native["longest_bilateral_slide_s"], beta, flush=True)
        if name == "smoke" and (native["outcome"] != "no_loss_before_limit"
                                or native["supported_force_peak_N"] > .05 or np.min(rows[:, 5:7]) <= .001):
            break


if __name__ == "__main__":
    main()

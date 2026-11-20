"""Check suspension while holding a recorded YAM actuator command."""

import json
from pathlib import Path

import numpy as np

from replay_refined import backend, integration_state, refined_scene
from scan_preload import contact_loads

RUN = Path(__file__).resolve().parent
BASE_HOLD_S = .20


def check(dt, output, scene):
    robot, frame, fingerprint = scene()
    sim = robot.sim
    sim.m.opt.timestep = dt
    command = sim.d.ctrl.copy()
    start = float(sim.d.time)
    run_start = None
    longest = 0.
    rows = []
    status = "no_continuous_suspension"
    for index in range(round(.45 / dt)):
        time = float(sim.d.time) - start
        gap = robot.gap()
        sim.step()
        contacts = contact_loads(robot)
        rows.append((time, gap, *contacts["inner_N"], contacts["tension_N"],
                     contacts["support_N"], contacts["penetration_mm"], contacts["qualified"]))
        if sim.d.warning.number.any() or contacts["penetration_mm"] > 1.:
            status = "numerical_gate"
            break
        if index % round(.002 / dt) == 0:
            failure, _ = backend.check_state(sim)
            if failure:
                status = failure
                break
        if contacts["qualified"]:
            run_start = time if run_start is None else run_start
            longest = max(longest, time - run_start)
            if longest >= BASE_HOLD_S - 1e-9:
                status = "suspension_established"
                break
        else:
            run_start = None
    np.testing.assert_array_equal(sim.d.ctrl, command)
    np.testing.assert_array_equal(sim.d.xfrc_applied, 0.)
    data = np.asarray(rows)
    stem = output / ("dt-" + f"{dt * 1e6:g}".replace(".", "p") + "us")
    names = ("time_s", "actual_gap_m", "left_inner_N", "right_inner_N", "tension_N",
             "support_N", "penetration_mm", "bilateral")
    np.savez_compressed(stem.with_suffix(".npz"), rows=data, columns=np.asarray(names),
                        final_state=integration_state(sim))
    result = dict(dt_s=dt, source_frame=frame, source_sha256=fingerprint, outcome=status, physics_steps=len(rows),
                  peak_tension_N=float(data[:, 4].max()), peak_support_N=float(data[:, 5].max()),
                  peak_penetration_mm=float(data[:, 6].max()), positive_bilateral_steps=int(data[:, 7].sum()),
                  longest_bilateral_s=longest)
    with stem.with_suffix(".json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


def main(run_dir=RUN, timesteps=(.00005, .000025), scene=refined_scene):
    output = run_dir / "free-relaxation"
    output.mkdir(exist_ok=False)
    for dt in timesteps:
        check(dt, output, scene)


if __name__ == "__main__":
    main()

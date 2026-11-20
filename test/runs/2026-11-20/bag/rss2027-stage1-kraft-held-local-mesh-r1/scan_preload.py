"""Compare native bilateral suspension at the same saved preload positions."""

import json
from pathlib import Path

import mujoco
import numpy as np

from replay_refined import backend, original_scene, refined_scene, replay

RUN = Path(__file__).resolve().parent


def contact_loads(robot):
    measured = backend.measure(robot)
    normal, _ = replay.inner_loads(replay.release_contacts(robot))
    tension = -sum(force[2] for force in measured["force_on_fingers_N"])
    return dict(inner_N=normal.tolist(), axial_feedback_N=measured["grip_each_N"], tension_N=tension,
                penetration_mm=measured["penetration_mm"], support_N=measured["support_N"],
                qualified=bool(normal.min() > .001 and tension > .001 and measured["support_N"] <= .05
                               and measured["penetration_mm"] <= 1.))


def main():
    trial = replay.SOURCE / "trial"
    trace = json.loads((trial / "trace.json").read_bytes())
    event = json.loads((replay.SOURCE / "withdrawal.json").read_bytes())
    robot, last, _ = original_scene()
    sim = robot.sim
    sim.m.opt.timestep = .00005
    frames = [i for i, row in enumerate(trace) if i <= last and trace[last]["t"] - row["t"] <= .2 + 1e-12
              and row["phase"] == "lift_hold"]
    records = []
    with np.load(trial / "trajectory.npz") as archive:
        for index in frames:
            state = archive["integration_state"][index]
            mujoco.mj_setState(sim.m, sim.d, state, int(archive["integration_state_spec"]))
            np.testing.assert_array_equal(sim.d.qpos, archive["qpos"][index])
            backend.refresh(sim)
            failure, health = backend.check_state(sim)
            eligible = (not failure and backend.measure(robot)["support_N"] == 0.
                        and not health["support_geometry"]["intersecting_triangles"])
            if not eligible:
                records.append(dict(frame=index, time_s=trace[index]["t"], support_removal_eligible=False))
                continue
            for body, expected, target in zip((replay.SUPPORT, replay.FLOOR), event["before_m"], event["after_m"], strict=True):
                pose = replay.support_pose(sim, body)
                np.testing.assert_allclose(pose, expected, atol=1e-12, rtol=0.)
                pose[:] = target
            backend.refresh(sim)
            coarse_loads = contact_loads(robot)
            fine, _, _ = refined_scene(lambda: (robot, index, ""))
            records.append(dict(frame=index, time_s=trace[index]["t"], support_removal_eligible=True,
                                original=coarse_loads, refined=contact_loads(fine)))
    output = RUN / "matched-preload-contacts.json"
    with output.open("x") as stream:
        stream.write(json.dumps(records, indent=2, allow_nan=False) + "\n")
    print("paired", sum(row.get("refined", {}).get("qualified", False)
                        and row.get("original", {}).get("qualified", False) for row in records), "of", len(records))


if __name__ == "__main__":
    main()

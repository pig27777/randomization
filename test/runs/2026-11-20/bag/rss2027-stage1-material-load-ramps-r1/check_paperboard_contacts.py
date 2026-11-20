"""Account for paperboard suspension force by the actual YAM collider."""

from functools import partial
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-kraft-held-load-step-r1")))
from replay_held_load import backend, jaw_axis, world_with_movable_support  # noqa: E402


def main():
    trial = RUN / "paperboard/recording/trial"
    cfg = backend.Config(**json.loads((trial / "result.json").read_bytes())["config"])
    with patch.object(backend, "parcel_world", partial(world_with_movable_support, lower_floor=True)):
        robot = backend.scene(cfg)
    sim = robot.sim
    trace = json.loads((trial / "trace.json").read_bytes())
    hold = [i for i, row in enumerate(trace) if row["phase"] == "lift_hold"]
    records = []
    with np.load(trial / "trajectory.npz") as archive:
        for j in (hold[0], hold[len(hold) // 2], hold[-1]):
            mujoco.mj_setState(sim.m, sim.d, archive["integration_state"][j], int(archive["integration_state_spec"]))
            backend.refresh(sim)
            axis, _ = jaw_axis(robot)
            contributions = [{}, {}]
            local = np.empty(6)
            for i, contact in enumerate(sim.d.contact):
                if contact.efc_address < 0 or 0 not in contact.flex:
                    continue
                mujoco.mj_contactForce(sim.m, sim.d, i, local)
                force = contact.frame.reshape(3, 3).T @ local[:3]
                for side, geoms in enumerate(robot.all_finger_sets):
                    for slot, g in enumerate(contact.geom):
                        if g not in geoms:
                            continue
                        name = sim.m.geom(g).name
                        body = sim.m.body(sim.m.geom_bodyid[g]).name
                        item = contributions[side].setdefault(name, dict(body=body, axial_N=0., contacts=0))
                        item["axial_N"] += float((2 * side - 1) * (2 * slot - 1) * (force @ axis))
                        item["contacts"] += 1
            split = [max(0., sum(item["axial_N"] for item in side.values())) for side in contributions]
            np.testing.assert_allclose(split, trace[j]["grip_each_N"], atol=1e-12, rtol=0.)
            np.testing.assert_array_equal(backend.measure(robot)["grip_each_N"], trace[j]["grip_each_N"])
            records.append(dict(time_s=trace[j]["t"], actual_gap_m=trace[j]["gap"], right=contributions[1],
                                left=contributions[0], measured_grip_N=trace[j]["grip_each_N"]))
    with (RUN / "paperboard/contact-geometries.json").open("x") as stream:
        stream.write(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

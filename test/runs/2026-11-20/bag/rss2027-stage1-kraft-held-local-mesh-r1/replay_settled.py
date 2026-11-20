"""Replay both steps from the one refined Kraft state that already settled."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from replay_refined import refined_scene, replay

RUN = Path(__file__).resolve().parent
PREPARATION = RUN / "free-relaxation/dt-50us"


def settled_scene():
    robot, frame, _ = refined_scene()
    sim = robot.sim
    info = json.loads(PREPARATION.with_suffix(".json").read_bytes())
    if info["outcome"] != "suspension_established" or info["peak_support_N"] > .05:
        raise ValueError("The refined source must have completed the bilateral suspension gate")
    previous_control = sim.d.ctrl.copy()
    with np.load(PREPARATION.with_suffix(".npz")) as archive:
        state = archive["final_state"]
        mujoco.mj_setState(sim.m, sim.d, state, int(mujoco.mjtState.mjSTATE_INTEGRATION))
    np.testing.assert_array_equal(sim.d.ctrl, previous_control)
    return robot, frame, hashlib.sha256(state.tobytes()).hexdigest()


if __name__ == "__main__":
    with patch.object(replay, "held_scene", settled_scene):
        replay.main()

"""Test whether measured jaw opening preserves the observed bubble-bag preload."""

import argparse
from pathlib import Path

import numpy as np

from replay_bubble import RUN, replay
from settle_refined import check

APERTURE_M = .005


def adjust(robot, gap, dt):
    sim = robot.sim
    actuator = robot.gripper
    low, high = sim.m.actuator_ctrlrange[actuator]
    speed = np.clip(20. * (APERTURE_M - gap), -.010, .010)
    sim.d.ctrl[actuator] = np.clip(sim.d.ctrl[actuator] + speed * dt, low, high)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    replay.SOURCE = RUN.with_name("rss2027-stage1-material-load-ramps-r1") / "bubble-early-withdrawal/recording"
    replay.CHECKPOINT = RUN / "held-state.npz"
    args.output.mkdir(exist_ok=False)
    for timestep in (.0002, .0001):
        check(timestep, args.output, replay.held_scene, adjust_gripper=adjust)


if __name__ == "__main__":
    main()

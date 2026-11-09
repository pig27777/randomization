"""I-11: why rigid_tet_check.py saw finger_off at 95% while test/contact_transfer failed it (42%, bouncing).

Runs test/contact_transfer/benchmark.py's own experiment and metrics, changing one setting at a time
from its formal configuration toward rigid_tet_check.py's. Results go to results_unified/i11/ so the
benchmark folder and its report are left untouched.

Usage: i11_finger_off.py [step ...]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

HERE = NEWTON
BENCH = HERE.parents[1] / 'test' / 'contact_transfer'
sys.path.insert(0, str(BENCH))
import warp as wp  # noqa: E402


NEWTON = Path(__file__).resolve().parents[2] / "contents/newton"
sys.path.insert(0, str(NEWTON))

import benchmark as bm  # noqa: E402

OUT = HERE / 'results_unified' / 'i11'
BASE = bm.Config(case='finger_off', dt=.001, iterations=10, ke=3000., kd=3., friction=0., guided=True, seconds=3.)
STEPS = {
    'formal': {},
    'free_tool': dict(guided=False),
    'dt2ms': dict(dt=.002),
    'friction08': dict(friction=.8),
    'all_three': dict(guided=False, dt=.002, friction=.8),
    # kinematic tool held at the height all_three settled to: contact alone, no tool-soft coupling iteration
    'position': dict(drive='position', tool_z_mm=56.84),
    'position_friction08': dict(drive='position', tool_z_mm=56.84, friction=.8),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('steps', nargs='*', default=list(STEPS))
    args = ap.parse_args()
    wp.config.quiet = True
    for name in args.steps:
        bm.run(replace(BASE, tag=f'i11_{name}', **STEPS[name]), OUT)


if __name__ == '__main__':
    main()

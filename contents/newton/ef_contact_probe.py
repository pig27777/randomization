"""Edge/face contact chatter: minimal repro with a kinematic 20 mm finger held between slab nodes.

The tool is prescribed (test/contact_transfer/benchmark.py --drive position), so only the
contact path and the soft solve act. Each variant changes one setting; a dense window samples
every substep to get the oscillation's peak-to-peak and dominant frequency.
A substep-alternating (2-cycle) oscillation shows up at 1 / (2 dt).
Results go to results_unified/ef_probe/.

Usage: ef_contact_probe.py [variant ...]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'test' / 'contact_transfer'))
import warp as wp  # noqa: E402

import benchmark as bm  # noqa: E402

OUT = HERE / 'results_unified' / 'ef_probe'
# 57.102 mm: the settled tool-centre height the benchmark's own ctl_position runs used
BASE = bm.Config(case='finger_off', dt=.001, iterations=10, ke=3000., kd=3., friction=0., drive='position',
                 tool_z_mm=57.102, seconds=2., dense_s=.05)
VARIANTS = {
    'base': {},
    'on_node': dict(case='finger'),
    'iter40': dict(iterations=40),
    'iter160': dict(iterations=160),
    'ke300': dict(ke=300., kd=.3),
    'ke100': dict(ke=100., kd=.1),
    'kd0': dict(kd=0.),
    'kd30': dict(kd=30.),
    'dt0p25': dict(dt=.00025),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variants', nargs='*', default=list(VARIANTS))
    args = ap.parse_args()
    wp.config.quiet = True
    for name in args.variants:
        bm.run(replace(BASE, tag=f'ef_{name}', **VARIANTS[name]), OUT)


if __name__ == '__main__':
    main()

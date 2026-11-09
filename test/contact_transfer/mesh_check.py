"""How much of the plate's coarse-to-fine sink change comes from the tet mesh rather than the contact?

Runs test/contact_transfer/benchmark.py's direct pressure load (no tool, no contact) and the plate with
planar quadrature contact, each on the 12x10x4 and 24x20x8 meshes, with one shared protocol.
Run with Newton 1.7.0 (/home/ubuntu/Dev/Envs/newton_latest). Results go to results_unified/mesh_check/.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

HERE = NEWTON
sys.path.insert(0, str(HERE.parents[1] / 'test' / 'contact_transfer'))
import warp as wp  # noqa: E402


NEWTON = Path(__file__).resolve().parents[2] / "contents/newton"
sys.path.insert(0, str(NEWTON))

import benchmark as bm  # noqa: E402

OUT = HERE / 'results_unified' / 'mesh_check'
BASE = bm.Config(dt=.001, iterations=20, seconds=8.)
MESHES = {'coarse': (12, 10, 4), 'fine': (24, 20, 8)}

if __name__ == '__main__':
    wp.config.quiet = True
    for case in sys.argv[1:] or ['direct_plate', 'plate']:
        for mesh, cells in MESHES.items():
            # planar quadrature only applies with a tool; the direct load has no contact at all
            bm.run(replace(BASE, case=case, cells=cells, tag=f'mesh_{mesh}', planar_quadrature=case == 'plate'), OUT)

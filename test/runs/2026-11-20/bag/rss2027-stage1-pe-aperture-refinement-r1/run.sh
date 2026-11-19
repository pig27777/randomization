#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
fixture=test/bag_model/rss2027_micropatch.py
python=.venv-physical-parcel/bin/python
cp -- "$fixture" "$run_dir/fixture_source.py"
args=(--match-whole-bag-pe --slack .008 --press-force .5 --gather-seconds .914
      --refine-contact both --refine-levels 2 --refine-aperture --hold-seconds .25
      --hold-downforce .05 --grip-target 1.0 --pull-seconds .65 --closed-gap .00110
      --coupon-offset-x .000625)
"$python" -B "$fixture" "${args[@]}" --smoke --output "$run_dir/smoke"
"$python" -B - "$run_dir" <<'PY'
import json
import math
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1])
fine = root.parent / 'rss2027-stage1-pe-offcenter-convergence-r1/fine-extended/config.json'
old = json.loads(fine.read_text())
smoke = root / 'smoke'
current = json.loads((smoke / 'config.json').read_text())
for key in ('reference_span_m', 'projected_span_m'):
    assert math.isclose(current['initial_geometry'][key], old['initial_geometry'][key], rel_tol=1e-12), key
key = 'total_film_mass_kg'
assert math.isclose(current['material'][key], old['material'][key], rel_tol=1e-12), key
with np.load(smoke / 'coupon_nodes.npz') as model:
    mesh = model['reference_m']
    for nodes, bound in ((mesh[:, 0] - current['coupon_offset_x'], .0075), (mesh[:, 1], .010)):
        coordinates = np.unique(nodes)
        part = coordinates[np.abs(coordinates) <= bound + 1e-12]
        assert len(part) > 1 and np.allclose(np.diff(part), .000625, atol=1e-12, rtol=0.)
PY
"$python" -B "$fixture" "${args[@]}" --output "$run_dir/refined-aperture"

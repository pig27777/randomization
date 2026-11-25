#!/usr/bin/env bash
set -euo pipefail
run="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$run/../../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
flock /tmp/claude-gpu.lock bash -s -- "$run" <<'LOCKED'
set -euo pipefail
run=$1
python=.venv-physical-parcel/bin/python
old=test/runs/2026-11-26/rss-exp-ab-r1
"$python" -B - "$run" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
pilot = {row["action"]: row for p in (root / "pilot-pressure-guard").glob("B-*.json")
         if (row := json.loads(p.read_text()))}
if len(pilot) != 9:
    raise SystemExit(f"Pilot incomplete: {len(pilot)}/9")
for action, row in sorted(pilot.items()):
    print(action, row["state"], row["physics_validity"], row["success"],
          row["bilateral_close_fraction"], row["invalid_reasons"], flush=True)
if not all(pilot[g]["success"] for g in ("g4", "g5", "g6")):
    raise SystemExit("Pilot diagonal hold/release gate failed")
if not all(pilot[g]["physics_validity"] and (pilot[g]["bilateral_close_fraction"] or 0) > 0
           for g in ("g1", "g2", "g3")):
    raise SystemExit("Pilot cross-core bilateral contact gate failed")
PY
date -Iseconds
sha256sum test/bag_model/rss_exp_ab.py test/bag_model/rss_exp_ab_analysis.py test/bag_model/rss_exp_ab_rework.py
"$python" -B test/bag_model/rss_exp_ab.py "$run/episodes" --phase formal --experiment B --workers 22 >> "$run/formal-light.log" 2>&1
if [[ ! -d "$run/heavy_episodes" ]]; then
    "$python" -B test/bag_model/rss_exp_ab_rework.py "$run" --source "$old" --part B >> "$run/analysis-b.log" 2>&1
fi
if [[ -d "$run/heavy_episodes" ]] || "$python" -B -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["light"]["heavy_control_triggered"])' "$run/expB_matrix_r2.json"; then
    date -Iseconds
    "$python" -B test/bag_model/rss_exp_ab.py "$run/heavy_episodes" --phase heavy --experiment B --workers 22 >> "$run/formal-heavy.log" 2>&1
    "$python" -B test/bag_model/rss_exp_ab_rework.py "$run" --source "$old" --part B >> "$run/analysis-b.log" 2>&1
fi
date -Iseconds
LOCKED

#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
base=$(dirname "$run_dir")
overlay="$base/rss2027-stage1-warp-film-normal-r1"
python=.venv-physical-parcel/bin/python
record=test/bag_model/record_physical_parcel.py
env PYTHONPATH="$overlay/isolated-package" "$python" -B "$record" \
    --engine warp --record-penetration-peaks --config "$run_dir/smoke.json" --output "$run_dir/smoke" \
    > "$run_dir/smoke.log" 2>&1
"$python" -B - "$run_dir/smoke/summary.json" <<'PY'
import json
from pathlib import Path
import sys

r = json.loads(Path(sys.argv[1]).read_text())
samples = r['gpu_samples']
if (r['dimensions']['film_vertices'] != 200 or r['outcome'] != 'incomplete' or r['warnings'] or
        r['peaks']['penetration_mm'] > 1. or r['sim_time_s'] < .02 or
        not samples or any(s['overflow_bits'] for s in samples)):
    raise SystemExit(f"The 200-node device entry failed: {r['reason']}")
PY
env PYTHONPATH="$overlay/isolated-package" "$python" -B "$record" \
    --engine warp --record-penetration-peaks --config "$run_dir/config.json" --output "$run_dir/recording"
"$python" -B "$overlay/analyze_trial.py" "$run_dir" \
    --cpu "$base/rss2027-stage1-pickup-200-cpu-r2/recording/trial/result.json"

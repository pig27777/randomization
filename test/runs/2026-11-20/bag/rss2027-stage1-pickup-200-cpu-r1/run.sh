#!/usr/bin/env bash
set -euo pipefail
run_dir=${1:-$(dirname "$0")}
python=.venv-physical-parcel/bin/python
record=test/bag_model/record_physical_parcel.py
"$python" -B test/bag_model/prepare_physical_parcel.py \
    --model-options "$(cat "$run_dir/prepare.json")" --seconds .50 \
    --output "$run_dir/preparation" > "$run_dir/preparation.log" 2>&1
"$python" -B "$record" --record-penetration-peaks --config "$run_dir/smoke.json" \
    --output "$run_dir/smoke" > "$run_dir/smoke.log" 2>&1
"$python" -B - "$run_dir/smoke/trial/result.json" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
if (result['dimensions']['film_vertices'] != 200 or result['outcome'] != 'incomplete' or
        result['warnings'] or result['peaks']['penetration_mm'] > 1. or result['sim_time_s'] < .02):
    raise SystemExit(f"200-node entry failed: {result['reason']}")
PY
"$python" -B "$record" --record-penetration-peaks --config "$run_dir/config.json" \
    --output "$run_dir/recording" > "$run_dir/recording.log" 2>&1
"$python" -B test/bag_model/rss2027_yam_capture_audit.py "$run_dir/recording/trial" "$run_dir/capture"

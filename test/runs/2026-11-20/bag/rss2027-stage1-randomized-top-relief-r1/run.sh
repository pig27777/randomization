#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
python=.venv-physical-parcel/bin/python
record="$run_dir/record_relief.py"
"$python" -B "$record" --config "$run_dir/smoke.json" --output "$run_dir/smoke" \
    --record-penetration-peaks > "$run_dir/smoke.log" 2>&1
"$python" -B - "$run_dir/smoke/summary.json" <<'PY'
import json
from pathlib import Path
import sys

r = json.loads(Path(sys.argv[1]).read_text())
if (r['dimensions']['film_vertices'] != 318 or r['outcome'] != 'smoke_only' or r['warnings'] or
        r['peaks']['penetration_mm'] > 1. or r['sim_time_s'] < .02):
    raise SystemExit(f"Closure relief entry failed: {r['reason']}")
PY
"$python" -B "$record" --config "$run_dir/config.json" --output "$run_dir/recording" --record-penetration-peaks
if [ -s "$run_dir/recording/closure-relief.json" ]; then
    "$python" -B test/bag_model/rss2027_yam_capture_audit.py "$run_dir/recording/trial" "$run_dir/capture"
fi

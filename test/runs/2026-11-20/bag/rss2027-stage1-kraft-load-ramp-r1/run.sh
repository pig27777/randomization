#!/usr/bin/env bash
set -euo pipefail
source_dir=$(dirname "$0")
run_dir=${1:-$source_dir}
if (( $# )); then shift; fi
python=.venv-physical-parcel/bin/python
record="$source_dir/record_load_ramp.py"
"$python" -B "$record" --config "$run_dir/config.json" --smoke --output "$run_dir/smoke" "$@" > "$run_dir/smoke.log" 2>&1
"$python" -B - "$run_dir/smoke/summary.json" <<'PY'
import json
from pathlib import Path
import sys

r = json.loads(Path(sys.argv[1]).read_text())
if (r['dimensions']['film_vertices'] != 142 or r['outcome'] != 'smoke_only' or r['warnings'] or
        r['peaks']['penetration_mm'] > 1. or r['sim_time_s'] < .02):
    raise SystemExit(f"Load-ramp entry failed: {r['reason']}")
PY
"$python" -B "$record" --config "$run_dir/config.json" --output "$run_dir/recording" "$@"
if [ -s "$run_dir/recording/load-curve.csv" ]; then
    "$python" -B test/bag_model/rss2027_yam_capture_audit.py "$run_dir/recording/trial" "$run_dir/capture"
fi

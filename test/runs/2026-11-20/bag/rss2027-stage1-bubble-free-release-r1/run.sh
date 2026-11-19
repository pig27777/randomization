#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
python=.venv-physical-parcel/bin/python
analysis=$(dirname "$run_dir")/rss2027-stage1-whole-bag-release-r1/analyze_release.py
"$python" -B "$run_dir/record_support_withdrawal.py" --smoke --output "$run_dir/smoke" > "$run_dir/smoke.log" 2>&1
"$python" -B - "$run_dir/smoke/summary.json" <<'PY'
import json
from pathlib import Path
import sys

r = json.loads(Path(sys.argv[1]).read_text())
if (r['dimensions']['film_vertices'] != 142 or r['outcome'] != 'smoke_only' or r['warnings'] or
        r['peaks']['penetration_mm'] > 1. or r['sim_time_s'] < .02):
    raise SystemExit(f"Support withdrawal entry failed: {r['reason']}")
PY
"$python" -B "$run_dir/record_support_withdrawal.py" --output "$run_dir/recording"
if [ -s "$run_dir/recording/withdrawal.json" ]; then
    "$python" -B test/bag_model/rss2027_yam_capture_audit.py "$run_dir/recording/trial" "$run_dir/capture"
    "$python" -B "$analysis" "$run_dir"
    "$python" -B test/bag_model/rss2027_release_traction.py "$run_dir"
fi

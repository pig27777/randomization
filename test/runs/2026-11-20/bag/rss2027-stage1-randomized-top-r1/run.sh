#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
python=.venv-physical-parcel/bin/python
record=test/bag_model/record_physical_parcel.py
audit=test/bag_model/rss2027_yam_capture_audit.py
for id in 004 127 002; do
    case_dir="$run_dir/sample-$id"
    if ! "$python" -B "$record" --config "$case_dir/smoke.json" --output "$case_dir/smoke" \
        --record-penetration-peaks > "$case_dir/smoke.log" 2>&1; then
        printf 'Sample %s failed to start, see %s/smoke.log\n' "$id" "$case_dir"
        continue
    fi
    if ! "$python" -B - "$case_dir/smoke/summary.json" <<'PY'
import json
from pathlib import Path
import sys

r = json.loads(Path(sys.argv[1]).read_text())
if (r['dimensions']['film_vertices'] != 142 or r['outcome'] != 'smoke_only' or r['warnings'] or
        r['peaks']['penetration_mm'] > 1. or r['sim_time_s'] < .02):
    raise SystemExit(f"First 20 ms failed the entry check: {r['reason']}")
PY
    then
        continue
    fi
    if ! "$python" -B "$record" --config "$case_dir/config.json" --output "$case_dir/recording" \
        --record-penetration-peaks > "$case_dir/run.log" 2>&1; then
        printf 'Sample %s aborted, see %s/run.log\n' "$id" "$case_dir"
        continue
    fi
    if "$python" -B - "$case_dir/recording/trial/trace.json" <<'PY'
import json
from pathlib import Path
import sys

rows = json.loads(Path(sys.argv[1]).read_text())
if not any(row['phase'] in ('close', 'lift', 'lift_hold') for row in rows):
    raise SystemExit('No closure or lift to replay; pressure trace archived')
PY
    then
        "$python" -B "$audit" "$case_dir/recording/trial" "$case_dir/capture"
    fi
done

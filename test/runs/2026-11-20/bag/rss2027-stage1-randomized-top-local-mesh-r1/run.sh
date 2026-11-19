#!/usr/bin/env bash
set -euo pipefail
source_dir=$(dirname "$0")
run_dir=${1:-$source_dir}
if (( $# )); then shift; fi
if (( $# == 0 )); then set -- refined coarse; fi
python=.venv-physical-parcel/bin/python
record="$source_dir/refine_parcel.py"
for variant in "$@"; do
    case_dir="$run_dir/$variant"
    "$python" -B "$record" --config "$case_dir/smoke.json" --output "$case_dir/smoke" \
        --record-penetration-peaks > "$case_dir/smoke.log" 2>&1
    if ! "$python" -B - "$case_dir/smoke/summary.json" "$source_dir/preparation.json" "$variant" <<'PY'
import json
from pathlib import Path
import sys

r = json.loads(Path(sys.argv[1]).read_text())
prep = json.loads(Path(sys.argv[2]).read_text())
vertices = prep['refined_vertices'] if sys.argv[3] == 'refined' else prep['original_vertices']
if (r['dimensions']['film_vertices'] != vertices or r['outcome'] != 'smoke_only' or r['warnings'] or
        r['peaks']['penetration_mm'] > 1. or r['sim_time_s'] < .02):
    raise SystemExit(f"{sys.argv[3]} entry failed: {r['reason']}")
PY
    then
        continue
    fi
    "$python" -B "$record" --config "$case_dir/config.json" --output "$case_dir/recording" \
        --record-penetration-peaks > "$case_dir/run.log" 2>&1
    if "$python" -B - "$case_dir/recording/trial/trace.json" <<'PY'
import json
from pathlib import Path
import sys

rows = json.loads(Path(sys.argv[1]).read_text())
if not any(row['phase'] in ('close', 'lift', 'lift_hold') for row in rows):
    raise SystemExit('No closure or lift to replay; pressure trace archived')
PY
    then
        "$python" -B test/bag_model/rss2027_yam_capture_audit.py "$case_dir/recording/trial" "$case_dir/capture"
    fi
done

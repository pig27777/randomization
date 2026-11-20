#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
case_dir=$1
recorder=${2:-"$run_dir/../rss2027-stage1-randomized-top-local-mesh-r1/refine_parcel.py"}
python=.venv-physical-parcel/bin/python
"$python" -B "$recorder" --config "$case_dir/smoke.json" --output "$case_dir/smoke" \
    --record-penetration-peaks > "$case_dir/smoke.log" 2>&1
jq -e '.dimensions.film_vertices==142 and .outcome=="smoke_only" and .warnings=={} and .peaks.penetration_mm<=1 and .sim_time_s>=.02' \
    "$case_dir/smoke/summary.json" >/dev/null
"$python" -B "$recorder" --config "$case_dir/config.json" --output "$case_dir/recording" \
    --record-penetration-peaks --record-capture-forces > "$case_dir/run.log" 2>&1
trial="$case_dir/recording/trial"
if jq -e 'any(.[]; .phase == "close")' "$trial/trace.json" >/dev/null; then
    "$python" -B test/bag_model/rss2027_yam_capture_audit.py "$trial" "$case_dir/capture"
fi

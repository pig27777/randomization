#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
recorder="$run_dir/../rss2027-stage1-wrap-content-continuation-r1/record_prepared_top.py"
python=.venv-physical-parcel/bin/python
for name in bubble paperboard kraft; do
    case_dir="$run_dir/$name"
    if ! "$python" -B "$recorder" --config "$case_dir/smoke.json" --output "$case_dir/smoke" \
        --record-penetration-peaks > "$case_dir/smoke.log" 2>&1; then
        printf '%s model assembly failed, see %s/smoke.log\n' "$name" "$case_dir"
        continue
    fi
    smoke="$case_dir/smoke/summary.json"
    jq -c '{outcome,sim_time_s,peaks,warnings}' "$smoke"
    if ! jq -e '.outcome=="smoke_only" and .sim_time_s>=.02 and .peaks.penetration_mm<=1 and .warnings=={}' \
        "$smoke" >/dev/null; then
        continue
    fi
    "$python" -B "$recorder" --config "$case_dir/config.json" --output "$case_dir/recording" \
        --record-penetration-peaks --record-capture-forces > "$case_dir/run.log" 2>&1
    trial="$case_dir/recording/trial"
    jq -c '{outcome,sim_time_s,peaks,warnings,normal_press}' "$trial/result.json"
    if jq -e 'any(.[]; .phase == "close")' "$trial/trace.json" >/dev/null; then
        "$python" -B test/bag_model/rss2027_yam_capture_audit.py "$trial" "$case_dir/capture"
    fi
done

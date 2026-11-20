#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
recorder="$run_dir/../rss2027-stage1-wrap-content-continuation-r1/record_prepared_top.py"
python=.venv-physical-parcel/bin/python
cases=(bubble paperboard kraft)
if (( $# )); then
    cases=("$@")
fi
for name in "${cases[@]}"; do
    case_dir="$run_dir/$name"
    checks=(smoke)
    if [ -f "$case_dir/settle.json" ]; then
        checks+=(settle)
    fi
    for mode in "${checks[@]}"; do
        if ! "$python" -B "$recorder" --config "$case_dir/$mode.json" --output "$case_dir/$mode" \
            --record-penetration-peaks > "$case_dir/$mode.log" 2>&1; then
            printf '%s model assembly failed, see %s/%s.log\n' "$name" "$case_dir" "$mode"
            continue 2
        fi
        smoke="$case_dir/$mode/summary.json"
        jq -c '{material:.config.bag_profile,outcome,sim_time_s,peaks,warnings}' "$smoke"
        if ! jq -e '.outcome=="smoke_only" and .sim_time_s>=.config.smoke_seconds and .peaks.penetration_mm<=1 and .warnings=={}' \
            "$smoke" >/dev/null; then
            continue 2
        fi
        if [ "${REQUIRE_CLEAR_APPROACH:-0}" = 1 ] && ! jq -e '.peaks.grip_N==0 and .peaks.down_N==0' "$smoke" >/dev/null; then
            printf '%s has finger load before approach during %s\n' "$name" "$mode"
            continue 2
        fi
    done
    "$python" -B "$recorder" --config "$case_dir/config.json" --output "$case_dir/recording" \
        --record-penetration-peaks --record-capture-forces > "$case_dir/run.log" 2>&1
    trial="$case_dir/recording/trial"
    jq -c '{outcome,sim_time_s,peaks,warnings,normal_press}' "$trial/result.json"
    if jq -e 'any(.[]; .phase == "close")' "$trial/trace.json" >/dev/null; then
        "$python" -B test/bag_model/rss2027_yam_capture_audit.py "$trial" "$case_dir/capture"
    fi
done

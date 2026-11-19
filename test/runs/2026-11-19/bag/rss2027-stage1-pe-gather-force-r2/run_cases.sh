#!/usr/bin/env bash
set -u
run_dir=$(dirname "$0")
python=.venv-physical-parcel/bin/python
fixture=test/bag_model/rss2027_micropatch.py
common=(--match-whole-bag-pe --slack .008 --gather-seconds .96 --pull-seconds 1.20)
run_case() {
    local name=$1
    shift
    "$python" -B "$fixture" "${common[@]}" --output "$run_dir/$name" "$@" > "$run_dir/$name.log" 2>&1
    local trial_exit=$?
    printf '%s %d\n' "$name" "$trial_exit"
    if [ "$trial_exit" -eq 0 ]; then
        "$python" -B test/bag_model/rss2027_gather_fit.py "$run_dir/$name" "$run_dir/$name/fit.json" >> "$run_dir/$name.log" 2>&1
    fi
}
run_case coarse
run_case half-step --dt .00005
run_case half-mesh --refine-contact both

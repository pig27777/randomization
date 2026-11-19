#!/usr/bin/env bash
set -u
run_dir=$(dirname "$0")
python=.venv-physical-parcel/bin/python
fixture=test/bag_model/rss2027_micropatch.py
common=(--match-whole-bag-pe --slack .008 --pull-seconds 1.2 --prefold --prefold-width .010 --grip-target 6.5)
for axis in x y
do
    "$python" -B "$fixture" "${common[@]}" --refine-contact "$axis" --output "$run_dir/$axis" > "$run_dir/$axis.log" 2>&1
    trial_exit=$?
    printf '%s %d\n' "$axis" "$trial_exit"
done

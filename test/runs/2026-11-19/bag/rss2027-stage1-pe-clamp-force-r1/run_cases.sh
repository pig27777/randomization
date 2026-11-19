#!/usr/bin/env bash
set -u
run_dir=$(dirname "$0")
python=.venv-physical-parcel/bin/python
fixture=test/bag_model/rss2027_micropatch.py
common=(--match-whole-bag-pe --slack .008 --pull-seconds 1.2 --prefold --prefold-width .010 --grip-target 6.5)

for label in preformed preformed-half-mesh
do
    mesh=()
    if [ "$label" = preformed-half-mesh ]; then
        mesh=(--refine-contact)
    fi
    "$python" -B "$fixture" "${common[@]}" "${mesh[@]}" --output "$run_dir/$label" > "$run_dir/$label.log" 2>&1
    trial_exit=$?
    printf '%s %d\n' "$label" "$trial_exit"
done

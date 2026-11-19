#!/usr/bin/env bash
set -u
run_dir=$(dirname "$0")
python=.venv-physical-parcel/bin/python
fixture=test/bag_model/rss2027_micropatch.py
common=(--match-whole-bag-pe --slack .008 --prefold --prefold-width .010 --pull-seconds 1.2 --refine-contact both)
for levels in 1 2
do
    "$python" -B "$fixture" "${common[@]}" --refine-levels "$levels" \
        --output "$run_dir/level-$levels" > "$run_dir/level-$levels.log" 2>&1
    trial_exit=$?
    printf 'level-%d %d\n' "$levels" "$trial_exit"
done

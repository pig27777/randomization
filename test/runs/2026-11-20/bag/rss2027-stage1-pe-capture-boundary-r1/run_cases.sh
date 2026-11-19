#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
python=.venv-physical-parcel/bin/python
fixture=test/bag_model/rss2027_micropatch.py
common=(--match-whole-bag-pe --gather-seconds .96 --pull-seconds .65 --refine-contact both)
cp -- "$fixture" "$run_dir/fixture_source.py"
"$python" -B "$fixture" "${common[@]}" --slack .004 --press-force .5 \
    --output "$run_dir/slack-4mm" > "$run_dir/slack-4mm.log" 2>&1
"$python" -B "$fixture" "${common[@]}" --slack .008 --press-force .8 \
    --output "$run_dir/pressure-0p8N" > "$run_dir/pressure-0p8N.log" 2>&1

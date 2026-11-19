#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
fixture=test/bag_model/rss2027_micropatch.py
cp -- "$fixture" "$run_dir/fixture_source.py"
.venv-physical-parcel/bin/python -B "$fixture" --match-whole-bag-pe --slack .008 \
    --press-force .5 --gather-seconds .914 --refine-contact both --refine-levels 2 \
    --hold-seconds .25 --grip-target 1.0 --pull-seconds .25 --closed-gap .00110 \
    --output "$run_dir/level-2"

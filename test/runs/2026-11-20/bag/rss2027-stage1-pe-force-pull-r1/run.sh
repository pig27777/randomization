#!/usr/bin/env bash
set -euo pipefail
run_dir=${1:-$(dirname "$0")}
grip_target=${2:-6.5}
fixture=test/bag_model/rss2027_micropatch.py
cp -- "$fixture" "$run_dir/fixture_source.py"
.venv-physical-parcel/bin/python -B "$fixture" --match-whole-bag-pe --slack .008 \
    --prefold --prefold-width .010 --refine-contact both --grip-target "$grip_target" --pull-force-target .30 \
    --pull-seconds 2.0 --output "$run_dir/level-1"

#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
.venv-physical-parcel/bin/python -B test/bag_model/rss2027_micropatch.py \
    --match-whole-bag-pe --slack .008 --pull-seconds 1.2 --prefold --prefold-width .010 \
    --refine-contact both --refine-levels 2 --output "$run_dir/preformed"

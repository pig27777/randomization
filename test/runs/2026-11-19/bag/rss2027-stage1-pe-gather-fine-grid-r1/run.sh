#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
python=.venv-physical-parcel/bin/python
"$python" -B test/bag_model/rss2027_micropatch.py --match-whole-bag-pe --slack .008 \
    --gather-seconds .96 --gather-only --refine-contact both --refine-levels 2 \
    --output "$run_dir/flat"
"$python" -B test/bag_model/rss2027_gather_fit.py "$run_dir/flat" "$run_dir/flat/fit.json"

#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
fixture=test/bag_model/rss2027_micropatch.py
case "${1:?use coarse-extended, fine-extended or half-step}" in
    coarse-extended) levels=1; dt=.0001; pull=.65 ;;
    fine-extended) levels=2; dt=.0001; pull=.65 ;;
    half-step) levels=2; dt=.00005; pull=.65 ;;
    *) exit 2 ;;
esac
cp -- "$fixture" "$run_dir/fixture-$1.py"
.venv-physical-parcel/bin/python -B "$fixture" --match-whole-bag-pe --slack .008 \
    --press-force .5 --gather-seconds .914 --refine-contact both --refine-levels "$levels" --dt "$dt" \
    --hold-seconds .25 --hold-downforce .05 --grip-target 1.0 --pull-seconds "$pull" \
    --closed-gap .00110 --coupon-offset-x .000625 --output "$run_dir/$1"

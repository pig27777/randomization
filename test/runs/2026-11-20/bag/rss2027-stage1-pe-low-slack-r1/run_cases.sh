#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
python=.venv-physical-parcel/bin/python
fixture=test/bag_model/rss2027_micropatch.py
common=(--match-whole-bag-pe --gather-seconds .96 --pull-seconds .65 --refine-contact both)

cp -- "$fixture" "$run_dir/fixture_source.py"
while read -r name slack pressure; do
    output="$run_dir/$name"
    if "$python" -B "$fixture" "${common[@]}" --slack "$slack" --press-force "$pressure" \
        --output "$output" > "$output.log" 2>&1; then
        "$python" -B test/bag_model/rss2027_gather_fit.py "$output" "$output/fit.json" >> "$output.log" 2>&1
    else
        printf 'Trial %s stopped. See %s.log\n' "$name" "$output"
    fi
done <<'CASES'
slack-2mm .002 .5
pressure-0p3N .008 .3
zero-slack 0 .5
CASES

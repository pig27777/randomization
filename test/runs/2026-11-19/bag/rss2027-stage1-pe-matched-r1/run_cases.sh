#!/usr/bin/env bash
set -u
run_dir=$(dirname "$0")
python=.venv-physical-parcel/bin/python
fixture=test/bag_model/rss2027_micropatch.py
common=(--match-whole-bag-pe --slack .008 --pull-seconds 1.2)
crease=(--prefold --prefold-width .010)

run_case() {
    local label=$1
    shift
    "$python" -B "$fixture" "${common[@]}" --output "$run_dir/$label" "$@" > "$run_dir/$label.log" 2>&1
    local exit_code=$?
    printf '%s %d\n' "$label" "$exit_code"
}

run_case flat-with-slack
run_case preformed "${crease[@]}"
run_case preformed-zero-friction "${crease[@]}" --friction 0
if "$python" -B - "$run_dir/preformed/summary.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
result = json.loads(path.read_text())
raise SystemExit(not (result["numerically_valid"] and result["capture_t_s"] is not None))
PY
then
    run_case preformed-half-mesh "${crease[@]}" --refine-contact
    run_case preformed-half-step "${crease[@]}" --dt .00005
fi

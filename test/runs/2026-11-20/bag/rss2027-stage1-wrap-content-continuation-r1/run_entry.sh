#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
case_parent=${1:-$run_dir/accepted-cases}
record=${2:-$run_dir/../rss2027-stage1-randomized-top-local-mesh-r1/refine_parcel.py}
python=.venv-physical-parcel/bin/python
for number in 074 078; do
    case_dir="$case_parent/sample-$number"
    for mode in smoke settle; do
        result="$case_dir/$mode"
        "$python" -B "$record" --config "$case_dir/$mode.json" --output "$result" \
            --record-penetration-peaks > "$case_dir/$mode.log" 2>&1
        if ! "$python" -B - "$result/summary.json" <<'PY'
import json
from pathlib import Path
import sys

r = json.loads(Path(sys.argv[1]).read_text())
print(r["outcome"], r["reason"], r["sim_time_s"], r["peaks"]["penetration_mm"])
raise SystemExit(not (r["outcome"] == "smoke_only" and r["dimensions"]["film_vertices"] == 142
                     and not r["warnings"] and r["peaks"]["penetration_mm"] <= 1.
                     and r["sim_time_s"] >= r["config"]["smoke_seconds"]))
PY
        then
            break
        fi
    done
done

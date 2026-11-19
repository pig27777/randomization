#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
load_ramp="$run_dir/../rss2027-stage1-kraft-load-ramp-r1/run.sh"
if (( $# == 0 )); then set -- bubble paperboard pe; fi
for material in "$@"; do
    case "$material" in
        bubble|paperboard|pe) ;;
        *) echo "Unknown material $material" >&2; exit 2 ;;
    esac
    case_dir="$run_dir/$material"
    flock /tmp/claude-gpu.lock bash "$load_ramp" "$case_dir" --load-rate 5 --withdraw-supports > "$case_dir/run.log" 2>&1
done

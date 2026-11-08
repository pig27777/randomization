#!/usr/bin/env bash
# Round 1 of test/proxy_eval/REPORT.md. Each object holds the shared GPU lock for its 50 samples.
# Usage: [LIMIT=n] bash run_round1.sh X1 X2 ...   (resumes: samples already in results/e2_<X>.jsonl are skipped)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home/ubuntu/Dev/Projects/real2sim/bubble_mailer/.venv/bin/python   # Newton 1.6.1 (bag_sim's Newton backend)
mkdir -p "$HERE/results/logs"
for o in "$@"; do
  flock /tmp/claude-gpu.lock "$PY" -B "$HERE/run_e2.py" run "$o" --limit "${LIMIT:-50}" >> "$HERE/results/logs/e2_$o.log" 2>&1
done

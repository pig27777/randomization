#!/bin/bash
# C1 (REPORT round 1): Newton kraft bag, full/loose box, R2 and R3 once each; serialized on the shared GPU lock.
cd "$(dirname "$0")/../.."
NP=/home/ubuntu/Dev/Projects/real2sim/bubble_mailer/.venv/bin/python
for fill in loose full; do
  for t in r3 r2; do
    flock /tmp/claude-gpu.lock $NP -B c1_newton.py $t $fill --out c1_newton.jsonl > outputs/round1/logs/c1_${t}_${fill}.log 2>&1
  done
done

#!/usr/bin/env bash
# Round 1 of the membrane benchmark (REPORT.md). Each formal run holds the shared GPU lock.
# Usage: bash run_round1.sh a | regress | b1 | b2   (A1 finger r1 was run first on its own)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
CT=$HERE/../contact_transfer
GG=$HERE/../gripper_grasp
PY=/home/ubuntu/Dev/Envs/newton_latest/bin/python    # Newton 1.7.0
PY161=/home/ubuntu/Dev/Projects/real2sim/bubble_mailer/.venv/bin/python
LOG=$CT/results/logs
mkdir -p "$LOG" "$GG/results/logs"
A="--ke 3000 --kd 3 --membrane-ke 300 --membrane-kd .3 --friction .5 --iterations 10 --seconds 8 --checkpoints 2"

bench() {  # tag case membrane dt
  flock /tmp/claude-gpu.lock "$PY" -B "$CT/benchmark.py" --cases "$2" --membrane "$3" --dt "$4" $A --tag "$1" \
    > "$LOG/$1_$2.log" 2>&1
}

case "$1" in
  a)
    for r in 2 3; do bench bag_a1_r$r membrane_finger bench .0001; done
    for c in membrane_plate membrane_finger_off; do for r in 1 2 3; do bench bag_a1_r$r $c bench .0001; done; done
    for c in membrane_plate membrane_finger membrane_finger_off; do bench bag_a2_r1 $c bench .00005; done
    for c in membrane_finger membrane_finger_off; do for r in 1 2 3; do bench bag_a3_r$r $c bubble .0001; done; done
    ;;
  regress)  # grasp.py is unchanged; reference grip_dt0p25_firm_off5 was Newton 1.6.1
    flock /tmp/claude-gpu.lock "$PY161" -B "$GG/grasp.py" --material firm --offset-mm 5 --dt .00025 --tag bag_regress_grip_dt0p25 \
      > "$GG/results/logs/bag_regress_grip_dt0p25.log" 2>&1
    ;;
  b1)
    for o in 0 5; do for r in ${REPS:-1 2 3}; do
      flock /tmp/claude-gpu.lock "$PY" -B "$HERE/bagged_grasp.py" --offset-mm $o --tag bag_b1_r$r \
        > "$GG/results/logs/bag_b1_r${r}_off$o.log" 2>&1
    done; done
    ;;
  b2)  # no-bag control: same pads, step, iterations and friction epsilon
    for o in 0 5; do for r in ${REPS:-1 2 3}; do
      flock /tmp/claude-gpu.lock "$PY" -B "$GG/grasp.py" --material firm --offset-mm $o --dt .00025 --iterations 10 \
        --pad-radius-mm 4 --friction-eps .01 --tag bag_b2_r$r > "$GG/results/logs/bag_b2_r${r}_off$o.log" 2>&1
    done; done
    ;;
  *) echo "usage: $0 a|regress|b1|b2"; exit 2 ;;
esac

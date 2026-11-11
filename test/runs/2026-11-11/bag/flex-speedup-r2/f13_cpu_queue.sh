#!/bin/bash
# F13 正式 CPU 测量：单线程单进程，每个配置单独排 GPU 锁（锁内机器不跑其他正式任务）。
cd /home/ubuntu/Dev/Projects/random
export OMP_NUM_THREADS=1
P=/home/ubuntu/Dev/Envs/mjwarp/bin/python
O=test/runs/2026-11-11/bag/flex-speedup-r2/f13
for c in bendable_only04 compressible_only04; do flock /tmp/claude-gpu.lock $P -B test/bag_model/flex_soft_cost.py --output $O --case $c --seconds 2; done
for c in hard08 hard04 bendable04 compressible04; do flock /tmp/claude-gpu.lock $P -B test/bag_model/flex_soft_cost.py --output $O --case $c; done

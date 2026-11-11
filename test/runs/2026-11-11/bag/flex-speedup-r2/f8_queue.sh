#!/bin/bash
# F8 正式运行：同一输入（centre 种子 1，第 3 轮 1024 环境穿透最大者）在 1/64/256/1024 环境下的逐环境穿透分布；
# 另跑 1 环境重复 3 次查确定性。每条命令单独排 GPU 锁。
cd /home/ubuntu/Dev/Projects/random
P=/home/ubuntu/Dev/Envs/mjwarp/bin/python
O=test/runs/2026-11-11/bag/flex-speedup-r2/f8
for r in 1 2 3; do flock /tmp/claude-gpu.lock $P -B test/bag_model/flex_round2.py --output $O --tag base_rep$r --nworld 1 --cases centre:1; done
for n in 64 256 1024; do flock /tmp/claude-gpu.lock $P -B test/bag_model/flex_round2.py --output $O --tag base --nworld $n --cases centre:1; done

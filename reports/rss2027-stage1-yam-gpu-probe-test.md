# RSS 2027 Stage 1 整袋 YAM GPU 入口检查

最近检查时间 2026-11-19 20:49（北京时间）。机器 ubun，RTX 5090。MuJoCo / MuJoCo Warp 4.0.1，Warp 1.19.0。

```sh
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B \
  test/runs/2026-11-19/bag/rss2027-stage1-yam-gpu-probe-r1/probe.py
git diff --check
```

- 完整 YAM linear_4310 归档模型匹配 200 节点、396 膜面、614 自由度和 7 个执行器。静置和合指位置帧分别以零速度起跑，CPU 与 GPU 逐步接收同一组共享 YAM 控制量，每个工况 10 步、步长 25 μs。两组完成且状态有限，GPU 无溢出。
- 首步膜弯曲力相对 L2 差为 0.0200% / 0.00565%；10 步后节点最大位置差为 1.61 / 0.946 μm。两个值分别为静置和合指。
- 合指首步 GPU 记录 4 / 6 个有效指尖—膜接触，CPU/GPU 左右指尖合力向量差 0.525 / 0.122 N，满足本轮首步容差。合指后续左指合力差最大 1.046 N，出现在第 7 步，右指最大 0.125 N。首步 CPU / GPU 全部接触候选数为 103 / 94。完整回合与反力一致性仍待验证。
- 单环境整卡占用采样起点 1148 MiB、运行后 2086 MiB，含其他进程与 GPU 上下文。入口耗时含编译、同步及主机读回，暂未取得闭环GPU回合吞吐。

[轮次计划](../test/runs/2026-11-19/bag/rss2027-stage1-yam-gpu-probe-r1/plan.md) · [逐步原始数据与源码指纹](../test/runs/2026-11-19/bag/rss2027-stage1-yam-gpu-probe-r1/results.json) · [运行日志](../test/runs/2026-11-19/bag/rss2027-stage1-yam-gpu-probe-r1/probe.log)

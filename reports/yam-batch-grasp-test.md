# YAM 8,000 路抓取测速检查

2026-11-20 19:18，北京时间，机器 `ubun`，RTX 5090，MuJoCo Warp 4.0.1。使用归档的 142 节点整袋、相同完整 YAM 轨迹入口和 CUDA Graph。正式命令持 `/tmp/claude-gpu.lock`。[试验报告](../test/bag_model/YAM_BATCH_GRASP_REPORT.md)

| 检查 | 结果 |
| --- | --- |
| `rss2027_yam_batch_grasp.py --worlds 8000 --step-us 10000 --probe` | 10 ms 装配时超过 5.823 ms 弯曲限制。 |
| `--worlds 1/8000 --step-us 5000 --probe` | 同一悬停短窗分别 56.91／18,125.57 环境步/s，算术复核为 318.48 倍总吞吐。两档穿透 2.00／6.56 mm，8,000 路有 EPA 溢出。 |
| `--worlds 8000 --step-us 5000` 完整入口 | 同一准备态复查，在第 5 步静置期超过 1 mm。参考的有效膜触点 1.183 mm，批量峰值 1.231 mm。 |
| `--worlds 8000 --step-us 200 --probe` | 18,535.91 环境步/s，3.707 仿真秒/墙钟秒。悬停短窗穿透 0.253 mm，零溢出和非有限环境。 |
| 独立抓取观测 | 两路实际 GPU 悬停位置，GPU 力和原参考读回双指与支撑差均在 0.0002 N 内，两路通过包覆检查。初次逐帧相等检查在 0.0202 s 时已不足以证明各环境的独立物理结果，因此改用逐环境观测。 |
| `--worlds 8000 --step-us 200 --contact-capacity 384 --constraint-capacity 1200 --sparse-capacity 24576` | 19:18 已启动完整动作。从各路包覆、接触、离台和悬停样本统计有效抓取回合。 |

本轮实现和数据通过 Python 解析、吞吐换算、测力回归以及 `git diff --check`。短窗表仅给出实际计算吞吐。完整抓取完成率和实际有效回合/小时以全程结果为准。[运行目录](../test/runs/2026-11-20/bag/yam-batch-full-grasp-r1/)

## 2026-11-20 异质闭合与悬停扩展扫描

最近验证于 2026-11-20 19:46（台北时间）。机器 `ubun`，RTX 5090、驱动 580.178.04、MuJoCo/MuJoCo Warp 4.0.1、Warp 1.19.0。

| 范围与方法 | 结果 |
| --- | --- |
| `flock /tmp/claude-gpu.lock bash <run>/prepare.sh` | 检出缺原准备态和轨迹，按归档配置重建。隔离源码哈希一致。4.77005 s 在线回合通过，峰值穿透 0.435 mm。 |
| `flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B <run>/scan.py` | 1–4,096 路两阶段、三个种子，54/54 窗口通过 64 步时钟、有限状态、零溢出和 1 mm 门槛。4,096 路悬停为 20,457.13 环境步/s，4.091 仿真秒/秒，显存峰值采样 11,516 MiB。 |
| 原 `rss2027_yam_batch_profile.py --worlds 1 --phase close` 入口 | 同一补录轨迹，原容量与无扰动的 64 步短窗通过，44.71 环境步/s，零溢出。 |
| JSON 独立复核、Python 语法、`bash -n` 和 `git diff --check` | 核对 18 个配置、54 个窗的时钟、吞吐换算、中位值、零溢出和世界间触点差异。通过。 |

扫描在用户给定的 4,096 路上限结束，仍有 17.4% 吞吐增长。回合当量以 4.77 s 归一化，完整并行闭环和训练成本仍待实测。[运行目录](../test/runs/2026-11-20/bag/rss2027-yam-heterogeneous-scaling-r1/) · [试验报告](../test/bag_model/YAM_BATCH_GRASP_REPORT.md#2026-11-20-异质环境闭合与悬停扩展扫描)

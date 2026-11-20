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

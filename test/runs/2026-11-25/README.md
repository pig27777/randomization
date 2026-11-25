# 2026-11-25 运行索引

## Isaac Lab

| 运行 | 指标与命令 | 结论 |
| --- | --- | --- |
| [Newton Yam_B 中精 PE 装袋](isaaclab/newton-yam-medium-bag-r1/) | `ubun`，RTX 5090，Isaac Lab v3.0.0，1,010 节点、0.5 ms、25 Hz，同轴样本 0.20／0.35 kg 和零反馈。仓库根目录执行 `bash test/runs/2026-11-25/isaaclab/newton-yam-medium-bag-r1/run.sh --ticks 62 --dt .0005 --output <新结果>`，额外两档加 `--mass .35` 或 `--feedback 0`。入口持 GPU 锁。 | 两质量完成 2.48 s，最大拉伸 ≤1.019；外壳—内芯旁路与零反馈首窗越限使纯薄膜关节辨识未通过。单路约 0.78 策略步/秒，712 MiB。[主题报告](../../isaaclab/REPORT.md) · [测试报告](../../../reports/isaaclab-newton-medium-bag-test.md) |

## Yam_B 纯仿真

| 运行 | 指标与命令 | 结论 |
| --- | --- | --- |
| [机器人刚体与 GPU](yam/pure-sim-r4/) | `ubun`，RTX 5090，Menagerie linear_4310，3.17.0，1 ms，25 Hz。持锁运行 `PYTHONPATH=experience .venv/bin/python -B test/yam/smoke.py <tracking/contact/grasp/gpu> <新目录>`。 | 空载、软硬接触、夹放、nominal 和 DR 批量通过，最高扫至 256 路。[主题报告](../../yam/REPORT.md) · [检查](../../../reports/yam-pure-sim-test.md) |
| [ABC 工作站](yam/pure-sim-r5/abc/) | 同机器、模型和环境，`smoke.py abc <新目录>`，使用原配置的桌面、双基座、顶视和右腕光学帧。 | 右侧 Yam_B 通过刚体抓放，左侧 Yam_A 停驻。[录像及反馈](yam/pure-sim-r5/media/) |
| [PE Flex 接入](yam/pure-sim-r6/) | MuJoCo/MJWarp 4.0.1，0.2 ms，三路同轨迹。持锁运行 `PYTHONPATH=.:experience .venv-physical-parcel/bin/python -B test/yam/bag_try.py <新目录> --scan-z-m .0347`。 | 中位触膜约 2.33 N 且无直接内芯接触，左右未触膜，E 未通过。前序结果见 [主题报告](../../yam/REPORT.md)。 |

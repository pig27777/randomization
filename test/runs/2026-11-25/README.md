# 2026-11-25 运行索引

## Isaac Lab

| 运行 | 指标与命令 | 结论 |
| --- | --- | --- |
| [Newton Yam_B 中精 PE 装袋](isaaclab/newton-yam-medium-bag-r1/) | `ubun`，RTX 5090，Isaac Lab v3.0.0，1,010 节点、0.5 ms、25 Hz，同轴样本 0.20／0.35 kg 和零反馈。仓库根目录执行 `bash test/runs/2026-11-25/isaaclab/newton-yam-medium-bag-r1/run.sh --ticks 62 --dt .0005 --output <新结果>`，额外两档加 `--mass .35` 或 `--feedback 0`。入口持 GPU 锁。 | 两质量完成 2.48 s，最大拉伸 ≤1.019；外壳—内芯旁路与零反馈首窗越限使纯薄膜关节辨识未通过。单路约 0.78 策略步/秒，712 MiB。[主题报告](../../isaaclab/REPORT.md) · [测试报告](../../../reports/isaaclab-newton-medium-bag-test.md) |

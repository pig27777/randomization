# 2026-11-25 运行索引

## Isaac Lab

| 运行 | 指标与命令 | 结论 |
| --- | --- | --- |
| [Newton Yam_B 中精 PE 装袋](isaaclab/newton-yam-medium-bag-r1/) | `ubun`，RTX 5090，Isaac Lab v3.0.0，1,010 节点、0.5 ms、25 Hz，同轴样本 0.20／0.35 kg 和零反馈。仓库根目录执行 `bash test/runs/2026-11-25/isaaclab/newton-yam-medium-bag-r1/run.sh --ticks 62 --dt .0005 --output <新结果>`，额外两档加 `--mass .35` 或 `--feedback 0`。入口持 GPU 锁。 | 两质量完成 2.48 s，最大拉伸 ≤1.019；外壳—内芯旁路与零反馈首窗越限使纯薄膜关节辨识未通过。单路约 0.78 策略步/秒，712 MiB。[主题报告](../../isaaclab/REPORT.md) · [测试报告](../../../reports/isaaclab-newton-medium-bag-test.md) |

## 袋体物理

| 运行 | 指标与命令 | 结论 |
| --- | --- | --- |
| [PE Flex / VBD 标准夹具基准](bag/pe-physics-benchmark-r1/) | `ubun`，RTX 5090，封边膜 142 节点。MuJoCo 4.0.1 / MJWarp 4.0.1，Newton 1.6.1 / Warp 1.19.0。仓库根目录执行 `$PE_PYTHON -B test/bag_model/pe_physics_protocol.py <新目录>`。VBD 加 `--backend vbd`，配置独立 Newton 依赖路径，使用 GPU 锁。 | 基础 pilot 与故障对照已运行。Flex 原基准静止与 U 折未通过，平面台面修复 U 折。VBD 同网格载荷下仍观察到内芯露角。正式 B07–B12 暂停。[主题报告](../../bag_model/REPORT.md#2026-11-25-pe-袋统一物理基准) · [测试报告](../../../reports/pe-physics-benchmark-test.md) |
| [PE 最小操作与 GPU 接触](bag/pe-mvp-r1/) | `ubun`，原 142 节点 Flex。仓库根目录运行 `$PE_PYTHON -B test/bag_model/bag_mvp.py <case> <新目录>`。GPU 入口 `bag_mvp_gpu.py` 使用同夹具和共享 GPU 锁。 | 四类 CPU 操作通过。GPU 主动接触通过，完整 GPU 夹持仍未通过。[测试报告与复现](../../../reports/pe-mvp-test.md) |

# 袋子模型测试

模型实现与参数在 [bag_sim/](../../bag_sim/README.md)，实验计划和逐轮结论统一维护于 [REPORT.md](REPORT.md)。本目录存放共享测试、演示、诊断与测速入口，按日期查轮次见 [runs/](../runs/README.md)。

## 入口分类

| 测试对象或用途 | 入口 |
| --- | --- |
| Newton 外袋材料、悬臂与缓冲 | `check_newton.py` |
| MuJoCo 外袋与原资产回归 | `check_unified.py` |
| 空袋、装内容物演示与录制渲染 | `demo_newton.py`、`demo_contents.py`、`render_demo.py` |
| 历史 R2/R3 与松装测速 | `c1_newton.py`、`loose_protocols.py`、`loose_speed.py`、`loose_mjwarp.py` |
| 薄膜聚拢网格对照 | [film_gather_grid.py](film_gather_grid.py) |
| flex 袋与软内容物的 CPU 单步成本 | [flex_soft_cost.py](flex_soft_cost.py) |
| 原生 3D flex 内容物对照 | [模型适配器](../flex3d/native_flex.py) |
| 松弛绳袋子探索演示与 MJWarp 绳长上限检查 | `slack_rope_demo.py`、`mjwarp_tendon_range_check.py` |
| 四部分软包探索演示（内容物+表面+松弛膜+夹爪）；`--set top` 顶面捏取（r1 情况），`--set side` 水平捏侧面（r2） | `four_part_pack_demo.py` |
| 统一YAM平台的侧面捏取适配器（v1.3本体＋松弛绳＋夹持规则），需要平台的可选接近方向，尚未运行 | `side_pinch_adapter.py`、`test_side_pinch_adapter.py` |
| 自动回归 | `test_*.py` |
| 既有双膜片纯接触原型，MuJoCo/MJWarp4.0.1 | `physical_parcel_checks.py`、`physical_parcel_benchmark.py`、`physical_parcel_yam.py`；结果见 [主题报告](PHYSICAL_PARCEL_REPORT.md) |

## 运行与结果

从仓库根目录使用报告指定的环境运行。下列命令分别检查双膜片模型的录像入口和 CPU 回归。

```sh
.venv-physical-parcel/bin/python -B test/bag_model/record_physical_parcel.py --help
.venv-physical-parcel/bin/python -B -m unittest discover -s test/bag_model -p test_physical_parcel.py -v
```

Newton、MuJoCo 3.14 对照与当前 MuJoCo 3.17 环境分别遵循 [袋子环境说明](../../bag_sim/README.md)。`--help` 和单元检查不等于物理验收。

新结果使用 `test/runs/YYYY-MM-DD/bag/<run-id>/`，在支持的入口显式传 `--output`、`--output-dir` 或 `--out`。机械臂模型对比使用 [experience 统一抓取平台](../../experience/README.md)。

## 历史批次

- F6/F7 的驱动为 [round3_flex.py](round3_flex.py)、[flex_batch.py](flex_batch.py) 和 [flex_diagnostics.py](flex_diagnostics.py)。[film_grasp.py](film_grasp.py) 提供原 F7 捏取流程。已退役的期限看守见 [迁移表](../MIGRATION.md)。
- B1–B4 的驱动为 [round3_effect.py](round3_effect.py) 和 [effect_batch.py](effect_batch.py)。

这些驱动使用原批次的固定期限和输出目录。原始配置和结果见 [第 3 轮报告](REPORT.md)。

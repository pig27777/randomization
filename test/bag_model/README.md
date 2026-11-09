# 袋子模型测试

模型实现与参数在 [bag_sim/](../../bag_sim/README.md)，实验计划和逐轮结论统一维护于 [REPORT.md](REPORT.md)。本目录存放共享测试、演示、诊断与测速入口，按日期查轮次见 [runs/](../runs/README.md)。

## 入口分类

| 测试对象或用途 | 入口 |
| --- | --- |
| Newton 外袋材料、悬臂与缓冲 | `check_newton.py` |
| MuJoCo 外袋与原资产回归 | `check_unified.py` |
| 空袋、装内容物演示与录制渲染 | `demo_newton.py`、`demo_contents.py`、`render_demo.py` |
| 历史 R2/R3 与松装测速 | `c1_newton.py`、`loose_protocols.py`、`loose_speed.py`、`loose_mjwarp.py` |
| Round 3 薄膜行为与批量检查 | `round3_flex.py`、`flex_batch.py`、`flex_diagnostics.py`、`film_grasp.py` |
| Round 3 效果袋与控制器对照 | `round3_effect.py`、`effect_batch.py`、`effect_control.py`、`benchmark_effect_control.py` |
| 袋子与软内容物联合求解测速 | `soft_joint_speed.py`、`soft_flex_speed.py`、`benchmark_soft_joint.py` |
| 原生 3D flex、几何、动力学与接触 | `probe_*`、`diagnose_*`、`validate_flex_dynamics.py`；具体配置见报告 |
| 自动回归 | `test_*.py` |

`deadline_watch.py` 是旧 F6/F7 批次的期限看守，只服务其历史 manifest，不能作为新实验的通用启动器。旧脚本里的期限、阈值和动作不因目录整理而改变。

## 运行与结果

从仓库根目录使用报告指定的环境运行。例如：

```sh
.venv/bin/python -B test/bag_model/loose_speed.py --help
.venv/bin/python -B -m unittest discover -s test/bag_model -p 'test_*.py' -v
```

Newton、MuJoCo 3.14 对照与当前 MuJoCo 3.17 环境分别遵循 [袋子环境说明](../../bag_sim/README.md)。`--help` 和单元检查不等于物理验收。

新结果使用 `test/runs/YYYY-MM-DD/bag/<run-id>/`，在支持的入口显式传 `--output`、`--output-dir` 或 `--out`；机械臂模型对比统一走 [flex3d 公共平台](../flex3d/README.md)。历史脚本若仍有固定输出路径，只作为原轮次复现入口，运行前须读报告并指定新结果路径；不要覆盖 `bag_sim/outputs/` 中已有证据。

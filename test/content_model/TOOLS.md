# 内容物模型测试入口

模型与版本目录见 [contents/](../../contents/README.md)。[README.md](README.md) 维护响应测法，[REPORT.md](REPORT.md) 维护实验计划与逐轮结论；本页只说明代码入口。按日期查记录见 [runs/](../runs/README.md)。

| 用途 | 入口 |
| --- | --- |
| 第一、二轮构建与响应对照 | `run_round1.py`、`run_round2.py` |
| 本构与模型验收、步长与交互查看 | `validate_model.py`、`stability_estimate.py`、`view_m2.py` |
| 拖动整体响应κ并查看真实CPU变形 | `parameter_ui.py`（界面为同名HTML，冻结v1.3） |
| v1 验收、分布与顶面抓分析 | `freeze_v1.py`、`response_distribution.py`、`analyze_top_grasp.py` |
| v1.2/v1.3 捏取验收 | `validate_pinch.py` |
| v1.4 捏取候选 CPU 对照 | `validate_pinch_v14.py` |
| v1.2 GPU 等价性 | `validate_acceleration.py`、`validate_warp.py` |
| v1.3 GPU 验收与续接 | `validate_warp_v13.py`、`validate_tasks_v13.py`、`validate_gpu_resume_v13.py` |
| GPU 任务诊断与离线审计 | `diagnose_r3_v13.py`、`analyze_r3_v13.py`、`diagnose_tasks_v13.py`、`audit_gpu_v13.py` |
| GPU 测速与历史探测 | `benchmark_gpu_v13.py`、`mjwarp_speed.py`、`mjwarp_validate.py` |
| Newton 分段模型检查 | `newton_check.py` |
| 单元测试 | `tests/test_*.py` |
| 测试工具的源码清单 | `source_manifest.py`，同时记录模型与测试驱动 |

在仓库根目录运行，使用报告指定环境：

```sh
.venv/bin/python -B test/content_model/validate_model.py --help
.venv/bin/python -B -m unittest discover -s test/content_model/tests -t test/content_model -v
```

新轮次结果放 `test/runs/YYYY-MM-DD/contents/<run-id>/`，在支持的入口显式传输出参数。原脚本用于历史轮次复现时，保留其固定输入、冻结源码选择与输出约定，先读对应报告；不要把旧脚本直接指向当前模型后引用其历史通过结论。新的机械臂抓取对比仍使用 [公共平台](../flex3d/README.md)。

κ预览从仓库根目录启动：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B test/content_model/parameter_ui.py --port 8765
```

在运行项目的机器打开 `http://127.0.0.1:8765`。调参重建并计算3秒，支持两类内容物、横向/厚度压缩和固定端悬垂。响应刚度和下垂范围来自冻结映射表，阻尼与载荷为预览控制。此处边界载荷与参考测法不同，结果用于探索，不能当作响应命中或抓取验收；关闭页面后2秒暂停步进。暂停/重新运行、旋转/缩放不改冻结源码。

原 `contents/segment/tests/` 和独立测试脚本已移到本目录。模型的采样、参数换算、测量和任务适配接口仍随模型维护，不从模型代码反向导入本目录。

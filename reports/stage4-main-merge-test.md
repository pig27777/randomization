# Fix 4 合流 main 检查

2026-11-22 20:17（北京时间），`ubun` RTX 5090。合流保留固定 25 mm 的 Policy 1 默认动作、Step 2 检查点与报告、C3/C6 录像和摆动分析。历史试验仍可显式传入外观高度使用校正相机接口。

| 检查 | 结果 |
| --- | --- |
| 冻结源文件 | 四个规定物理核心、有限指垫、YAM 电流滤波、DR 及配置与 `046384ab` 的 Git 差异为空。 |
| CPU Step 1 | `.venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_stage4_rl.py -v`，5 项通过，CUDA 专属项交由对应设备。 |
| 4,096 路固定角点 | 共享 GPU 锁下使用 `evaluate(model, "cuda", 4096, 88003, True, corners=True)`。每个角点的整个指标字典都等于正式报告原始数据，成功数为 0、512、512、512、512、12、0、512。 |
| 工作树与产物 | 无遗留合并标记，`git diff --cached --check` 通过。C6 曲线、日志、权重和录像全部纳入 main。 |

[CPU 日志](../test/runs/2026-11-22/policy/stage4-main-fix4-merge-r1/cpu.log) · [RTX 5090 逐角核对](../test/runs/2026-11-22/policy/stage4-main-fix4-merge-r1/parity.log) · [Step 2 报告](stage4-policy1-training-test.md) · [C6 摆动报告](stage4-c6-sway-test.md)

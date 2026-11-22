# Fix 4.1 与 ABC 场景合入 main

本轮计划。合并 `stage4/route-b-fix41@db5dbe48` 至 `main`。分别检查冻结物理源和 USD 的一致性，以及主分支校准视觉与 Fix 4.1 可配置视觉窗口的兼容性。运行 CPU 定向回归、CUDA Stage 3 与 ABC 双臂场景。所有 GPU 运行在 ubun RTX 5090 持共享锁。通过后提交并推送，失败先定位。

最近验证 2026-11-22 22:16（北京时间）。两条 RGB 路径均保留。默认投影的采样窗为 25 mm，可以传入逐环境行程。已有调用传入外观高度时继续使用标定算法和原有的视觉有效位。

| 检查 | 命令或方法 | 结果 |
| --- | --- | --- |
| 冻结源及资产 | 用 `git diff --exit-code` 对照 `046384ab`、`fix4.1-route-b-base` 和 `db5dbe48` | 两个冻结核心、Fix 4.1 物理实现、随机化参数及 ABC 场景资产均与对应源一致。 |
| CPU 回归 | 使用项目 Python 执行 `test_neck_damping.py`、`test_stage3*.py`、`test_stage4_rl.py` 和 `test_stage4_lift_probe.py` | 24 项通过，跳过 1 项 CUDA 专用 4,096 路宏步试验。 |
| Stage 3 CUDA | `STAGE3_TEST_DEVICE=cuda` 下执行 `test_stage3*.py` | 15/15 通过。 |
| ABC 场景 | 原生 Isaac Python 执行 `check_route_b_direct_scene.py --headless --dual-arm` | 4 路双臂通过。物理步长 2 ms，一次 25 Hz 步执行 20 子步，台面 0 m，物理相机 0。 |

PhysX 仍记录了已知的执行器 CUDA Graph 回退。本轮执行的是合并回归，Step 1 真实 YAM–包裹闭环与吞吐仍待验收。[CPU](../test/runs/2026-11-22/isaaclab/route-b-main-merge-r1/cpu.log) · [CUDA](../test/runs/2026-11-22/isaaclab/route-b-main-merge-r1/cuda.log) · [ABC](../test/runs/2026-11-22/isaaclab/route-b-main-merge-r1/abc.log)

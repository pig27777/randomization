# Stage 4 Step 1 测试报告

最近复验为 2026-11-21 20:51（北京时间）。测试在 `ubun` 进行，使用 Fix 4 冻结提交 `046384ab` 的物理、Stage 3 域随机化及传感器模块。环境为 float32、2 ms。CUDA 使用 RTX 5090、PyTorch 2.11.0+cu130 和 `/tmp/claude-gpu.lock`。微提拉按附件固定为 25 mm，上升 0.40 s，停留 0.15 s。为覆盖不同包高，标准 20 mm 触探之前用同一编码器和去皮电流做低刚度预定位。

| 范围 | 结果 |
| --- | --- |
| CPU 定向检查 | 5 项通过，4,096 路用例由 CUDA 执行。Actor 与 Critic 真值隔离，无效历史不影响动作。未激活阶段、同阶段未选连续头及离散锁定的梯度均精确为零。直接验证力保持率的两个截断端点。 |
| RTX 5090 批量宏步 | 两套环境各 4,096 路全宽域、各三次宏步调用。抽查点的观测、奖励、包裹位置与速度以及两臂接触张量均有限。固定混合指令下 Pick 和 Handover 分别完成 110 和 258 路。CUDA 整套 5 项及补强后的定向复检 2 项通过。 |
| C10 拉拔 | 真值目标 6 N、保压 130 ms。CPU 双臂估力同时超过 3.5 N 连续 7 帧（140 ms），保压真值均值为 6.15/6.14 N，瞬态峰值 7.06 N。CUDA 连续 8 帧（160 ms），保压真值均值 6.28/6.27 N，瞬态峰值 7.21 N。 |
| C10 交接与放置补充探针 | 6 路固定指令中 1 路达到 Arm B 独立稳悬并完成放置。把卸夹力、张爪和撤离依次执行后，该路的松爪高度由 13.6 mm 降到 0 mm，峰值桌面力由 16.88 N 降到 1.17 N。 |

隔离复验首轮发现共享工作树的微提拉模块尚未纳入 Fix 4，改用工作台标定常量。同期另有尺寸自适应微提拉与 RGB 接口试验，其改动已归档为独立补丁。本次门禁从 `046384ab` 取底座和传感器，只叠加 Stage 4 文件。以并行更新 `554f335d` 为底座另跑过 CPU 接口检查。期望观测为 Pick `15 / 28×12 / 86`、Handover `18 / 28×14 / 102`，力保持率截断为 `[-0.5, 2.0]`。

在 Stage 4 交付提交上运行以下命令复现冻版门禁。记录分别在 [CPU](../test/runs/2026-11-21/policy/stage4-step1-r1/final-cpu-r4.log)、[CUDA](../test/runs/2026-11-21/policy/stage4-step1-r1/final-cuda-r3.log)、[CUDA 定向复检](../test/runs/2026-11-21/policy/stage4-step1-r1/final-cuda-mask.log) 和 [C10 放置探针](../test/runs/2026-11-21/policy/stage4-step1-r1/c10-placement-probe-r9.log)。

```sh
stage4_repo=$PWD
stage4_snapshot=$(mktemp -d /tmp/parcel-stage4.XXXXXX)
git archive 046384ab src/low_order_parcel configs | tar -x -C "$stage4_snapshot"
git archive HEAD src/low_order_parcel/envs src/low_order_parcel/rl test/low_order_parcel/test_stage4_rl.py |
  tar -x -C "$stage4_snapshot"
cd "$stage4_snapshot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHONPATH="$PWD" "$stage4_repo/.venv-physical-parcel/bin/python" -B \
  -m unittest discover -s test/low_order_parcel -p test_stage4_rl.py -v
PYTHONPATH="$stage4_repo/.venv-physical-parcel/stage3-cu130:$PWD" STAGE4_TEST_DEVICE=cuda \
  flock /tmp/claude-gpu.lock "$stage4_repo/.venv-physical-parcel/bin/python" -B \
  -m unittest discover -s test/low_order_parcel -p test_stage4_rl.py -v
```

本轮覆盖仿真入口和规定的 Step 1 门禁。训练收敛、策略串联盲测和三种子基线留待附件的 Step 2–4。

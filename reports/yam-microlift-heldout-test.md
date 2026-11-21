# YAM 微提反馈留出实验检查

最近检查于 2026-11-21 20:07 北京时间，ubun。单臂 YAM、PE 外袋和四块内容物，冻结动力学 d6e519bc，2 ms 步长，MuJoCo 4.0.1、PyTorch 2.15.0+cpu。实验按预定标准未通过，矩阵执行和配对完整性检查通过。

| 检查 | 结果 |
| --- | --- |
| 入口试跑 | 独立衣物、210 g、0.04 s，成功装配并记录，不计入正式矩阵。 |
| 正式比较 | 持共享锁，6 个 CPU worker。9 组 × 3 个控制规则全部完成；留出后没有改模型、动作或决策规则。 |
| 信息隔离和复放 | 9/9 决策前逐步指令/关节前缀哈希、估力与夹口采样、包裹轨迹相同。反馈控制只接收两个载荷特征与夹口变化量。9/9 反馈组的完整边界数组及平台轨迹与所选固定分支完全一致。各组机器人资产及源码快照匹配，27/27 无 MuJoCo 警告。 |
| 成功和限制 | 三组 3 s 稳悬均 0/9，平台抓起松开依次 4/9、5/9、5/9，未发生过压或意外回台。决策时九例净空均为零。仅测试确定性模拟，无真机验证。 |
| 共享平台回归 | 关闭微提选项，通过原入口复跑 heavy-raised-fold。与无源性审计 final-yam 同工况的 trace.json、trajectory.npz 和 boundary.npz 三个文件均逐字节相等。 |
| 归档 | 27 个试次的 810 个文件经无损压缩并逐文件回读比对。包括每次的原协议结果与力、关节轨迹、逐步包裹指标、源码快照和机器人资产指纹。 |

正式重跑（使用空闲的新目录），从仓库根目录执行：

```sh
out=$(mktemp -d)
flock /tmp/claude-gpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-physical-parcel/bin/python -B test/policy/run_yam_microlift.py "$out/matrix" --workers 6
.venv-physical-parcel/bin/python -B test/policy/analyze_yam_microlift.py "$out/matrix"
```

复查已归档的结果：

```sh
run=test/runs/2026-11-21/policy/yam-microlift-heldout-r1
review=$(mktemp -d)
cp "$run"/{manifest,results}.json "$review/"
tar -xzf "$run/trials.tar.gz" -C "$review"
.venv-physical-parcel/bin/python -B test/policy/analyze_yam_microlift.py "$review" > "$review/analysis.json"
cmp "$run/analysis.json" "$review/analysis.json"
```

[实验报告](../test/policy/PROBE_REPORT.md#配对结果) · [逐回合结果](../test/runs/2026-11-21/policy/yam-microlift-heldout-r1/results.json) · [源码清单与任务矩阵](../test/runs/2026-11-21/policy/yam-microlift-heldout-r1/manifest.json)

## 2026-11-21 Fix 4 与按尺寸 Phase B

最近检查于 20:54 北京时间，ubun。原矩阵补记九个 2 ms 决策步的柔索张力；九次复放的控制/传感前缀哈希与整回合边界数组均与 r1 一致。物理源文件逐项匹配 Fix 4 冻结提交 046384ab。固定 25 mm 与按尺寸 90 mm 各 27/27 个正式回合，两批的 9/9 个策略组通过决策前指令和观测一致、反馈选择的全程轨迹与固定分支一致、资产指纹及警告检查。两批最长稳悬分别为 2.702 s 与 2.882 s，均未通过 3 s 任务标准。独立八个视觉对照全部完成并达到决策点；校正视觉排序 AUC 0.833，估力单独 0.500。上述是完整 YAM 确定性仿真，未验证真机图像或策略泛化。

单臂策略 CPU 入口及 25 mm 后视觉跟踪两项通过，既有 Stage 3 相机相关回归三项通过。RTX 5090、PyTorch 2.11.0+cu130 上持锁，以随机种子 726893 运行八环境 CUDA Graph 试夹/Phase B/Actor 决策，输出 8 × 32 × 18，数值有限，目标行程和视觉有效位通过。旧 Actor 的 15 通道权重能以新增通道零权重加载。实验初始化的隔离源码副本第一次启动缺少 Stage 1 包络模块，补齐同版本依赖并做 0.04 s 装配试跑后才启动正式矩阵。三批轨迹压缩归档后解包复算，三份 projection.json 和两份 analysis.json 均与归档结果逐字节一致。独立视觉对照命令的 0.04 s 装配检查通过。原始九例的 FK 读数与决策步均相差 0 ms，决策瞬时相对闭合基线上升 25.12–25.39 mm。

复现矩阵及独立对照（均使用新输出目录）：

```sh
work=$(mktemp -d)
py=.venv-physical-parcel/bin/python
uniform=(flock /tmp/claude-gpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1)
"${uniform[@]}" "$py" -B test/policy/run_yam_microlift.py "$work/fix4" --fix4 --workers 9
"${uniform[@]}" "$py" -B test/policy/run_yam_microlift.py "$work/sized" --fix4 --size-probe --workers 9
"${uniform[@]}" "$py" -B test/policy/run_yam_microlift.py "$work/vision" --projection-controls --workers 8
"$py" -B test/policy/analyze_yam_microlift.py "$work/fix4"
"$py" -B test/policy/attribute_yam_microlift.py "$work/fix4" "$work/fix4-attribution.json"
"$py" -B test/policy/check_probe_projection.py "$work/vision"
```

原 r1 可从其 trials.tar.gz 解包，并将同目录的 results.json 复制到解包根目录。执行 `attribute_yam_microlift.py 解包根目录 新归因文件 --replay-loads`，同样使用上述锁和线程设置。程序验证复放状态完全一致才补记张力。CPU 功能检查为 `python -B -m unittest discover -s test/policy -p test_stage4_lift_probe.py -v` 及 `python -B -m unittest discover -s test/low_order_parcel -p test_stage3_dual_arm.py -k rgb -v`。

[逐案归因](../test/runs/2026-11-21/policy/yam-fix4-25mm-r1/baseline-attribution.json) · [独立对照数据及检查日志](../test/runs/2026-11-21/policy/yam-microlift-rgb-validation-r1/) · [完整实验报告](../test/policy/PROBE_REPORT.md#原始失败归因与-fix-4-配对结果)

同一工作区正在开发的 PickToAirEnv 经 CPU、CUDA 各两行验证，顶抓和侧抓均到达执行后决策。复现命令为 `PYTHONPATH="$PWD" .venv-physical-parcel/bin/python -B test/runs/2026-11-21/policy/yam-microlift-rgb-validation-r1/check_pick_to_air.py cpu`。CUDA 复现使用 `flock /tmp/claude-gpu.lock env PYTHONPATH="$PWD/.venv-physical-parcel/stage3-cu130:$PWD"` 作为前缀，脚本参数改为 `cuda`。实发行程、两个策略通道与缺测标志断言均通过。并行会话交付固定行程宏入口基线 `2fcc183d` 后，本次更改已接入正式源码；归档中附有只含两个接入位置的[可复现差异](../test/runs/2026-11-21/policy/yam-microlift-rgb-validation-r1/pick-to-air-phase-b.patch)。

在宏入口基线交付后的最终工作树再次执行 CPU 两例，输出与原日志逐字节相同。随后执行 `python -B -m unittest discover -s test/low_order_parcel -p test_stage4_rl.py -k test_pick_macro_history_and_retry -v`，正常决策、重抓和重试历史回归 1/1 通过。[回归日志](../test/runs/2026-11-21/policy/yam-microlift-rgb-validation-r1/pick-retry-regression.log)

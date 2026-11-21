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

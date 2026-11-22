# 路线 B Part I · Fix 4.1 冻结验收

机器 `ubun`，RTX 5090。包裹封装使用 `046384ab` 的核心接触与四块载荷方程。兼容性验证使用项目原虚拟环境，GPU 回放与相图持 `/tmp/claude-gpu.lock`。标准 2 ms，场景与原录像回放为 float32。能量审计保留原历史工况的既有精度。 [计划](../test/low_order_parcel/STAGE34_REPORT.md) · [运行证据](../test/runs/2026-11-22/bag/route-b-fix41-r1/)

## 公共底座

横向阻尼先扣除指垫绕自身旋转引起的牵连速度，再投影至膜颈轴的横向平面。挂点收到阻尼力，指垫收到反力与纯力矩。摩擦和同时活动指的方向等效逆惯量限制单步冲量；冻结宏观腕部的平移惯量仍为 0.25 kg，姿态是指定边界。无空气阻尼。系数名义 12 N·s/m，Tier 1 取 10–14 N·s/m，特征长度 50 mm。Tier 1 新参数使用独立随机流。禁用时沿用冻结运算路径。视觉上升窗口接受标量或逐环境行程，默认 25 mm，保留既有观测布局与力保持率。 [阻尼实现](../src/low_order_parcel/neck_damping.py) · [视觉窗口](../src/low_order_parcel/virtual_rgb.py#L165) · [参数](../configs/stage3_dr_and_corners.yaml)

| 冻结门禁 | 结果 |
| --- | --- |
| 原 19 工况及新增 C6 | **20/20**。最差 **−0.049388 mJ**，限值 −0.1 mJ，越限 0。首次捕获拉伸和抗转储能均为 **0 J**。 |
| 新增 C6 单项 | 最差 −0.0000190 mJ，2,250 步双指持续捕获。横向阻尼累计耗散 0.40970 mJ，末 3 s 最大质心速度 0.136 mm/s。 |
| 旋转反例及零退化 | **2/2**。原反例在 2 ms 内由注能 +0.105 mJ 转为做功 **−0.735 mJ**。共同刚转零力，反力矩残差为零，摩擦和冲量截断通过。Macro 与 DualArm 的零系数逐元素匹配禁用路径。 |
| GPU 32×32 相图 | **3/3**，非有限格 0。硬书误捕获、误离台和相图 B 逆序均为 0。相图 C 低区 84 格零捕获，高区 54 格均双指捕获。 |
| Stage 2 | **7/7**，1 ms 和 2 ms 各执行 1,000,000 步极端输入。 |
| Stage 3 CPU / CUDA | **15/15 / 15/15**。含默认 25 mm 逐元素对照及批内自适应窗口使离台样本恢复视觉比值。 |
| 原物理包 | **20/20**。 |

## 原动作复放

用原检查点、固定角点种子和完整 4,096 CUDA 批取得冻结版第一次抓取与验证动作，候选复用同一动作向量。未训练旧策略。表中速度取实际触发成功的连续 3 s 窗口。

| 固定动作 | 结果 | 速度中位／P95／最大值，mm/s | 最小净空，mm |
| --- | --- | ---: | ---: |
| C1 第 0 行 | 连续 3.0 s，成功 | 2.369 / 9.040 / 19.799 | 190.281 |
| C6 第 5 行 | 连续 3.0 s，成功 | 2.269 / 15.996 / 19.969 | 210.612 |
| C7 第 6 行 | 连续 3.0 s，成功 | 3.424 / 15.819 / 19.892 | 15.900 |

C6 的质心横向峰峰值从录像复算的 29.448 mm 降至这次稳悬窗的 13.369 mm。系数 8 的候选能量通过，C7 最长 2.91 s，因此以 12 完成冻结门禁。 [初轮复放](../test/runs/2026-11-22/bag/route-b-fix41-r1/replay-c8.log) · [冻结候选复放](../test/runs/2026-11-22/bag/route-b-fix41-r1/replay-c12.log)

## 执行入口

```sh
export PYTHONPATH="$PWD/.venv-physical-parcel/stage3-cu130:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
yam=test/runs/2026-11-21/bag/parcel-passivity-fix-r1/final-yam
.venv-physical-parcel/bin/python -B test/low_order_parcel/audit_energy_passivity.py "$out/energy" --finite-pad --neck-coefficient 12 --yam-root "$yam"
.venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_neck_damping.py -v
.venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_stage2_proxy.py -v
.venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p 'test_stage3*.py' -v
flock /tmp/claude-gpu.lock env STAGE3_TEST_DEVICE=cuda .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p 'test_stage3*.py' -v
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B -m unittest discover -s test/bag_model -p test_physical_parcel.py -v
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B test/low_order_parcel/replay_fix41_sway.py --coefficient 12
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B test/low_order_parcel/audit_phase_diagrams.py "$out/phase-maps" --finite-pad --neck-coefficient 12
```

[独立新增 C6 能量账本](../test/runs/2026-11-22/bag/route-b-fix41-r1/energy-c12/c6-neck.csv) · [20 项总表](../test/runs/2026-11-22/bag/route-b-fix41-r1/energy-c12/energy-summary.json) · [三张相图](../test/runs/2026-11-22/bag/route-b-fix41-r1/phase-maps/)

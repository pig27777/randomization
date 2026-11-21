# 包裹物理真实性审计检查

最近检查为 2026-11-21 18:33（北京时间）。机器 `ubun`，基线 `96a896c1`。CPU 和 MuJoCo 使用本工作树的 `.venv-physical-parcel/bin/python -B`，MuJoCo 4.0.1。该解释器为 Python 3.14，当前安装的 PyTorch 为 CPU 版。CUDA 用同机另一工作树已有的 Python 3.12 解释器加载本工作树内从缓存组装的依赖目录，PyTorch 2.11.0+cu130、RTX 5090。正式运行持 `/tmp/claude-gpu.lock`。

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| 冻结 P6、PE 刚度、敏感性 | `-m src.low_order_parcel.audit_physics_realism`、`test/bag_model/audit_stage1_stiffness.py`、`test/low_order_parcel/audit_stage2_sensitivity.py` | 8,003/8,192 路悬停，343.974884 N。P6 逐路和波形与冻结归档 SHA-256 一致。YAM PE 刚度和敏感性汇总与归档一致。 |
| YAM 闭环及逐步能量 | `test/low_order_parcel/audit_yam.py`、`test/low_order_parcel/audit_energy_passivity.py` | Stage 2 完整侧抓和受控松夹通过。Stage 2.5 能量门禁失败。修正折边入口条件后，YAM 重包首次捕获仍出现 −9.569 mJ，冻结模型静止捕获出现 −6.826 mJ。 |
| GPU 相图 | `test/low_order_parcel/audit_phase_diagrams.py`，每图 32×32 | 三图通过。修复初筛 Phase B 错误的探针入参，四个交叉网格单元的全部帧输入与单环境轨迹相符。 |
| Stage 1 膜片 | `test/bag_model/audit_stage1_scaling.py` | 膜厚倍增后弯曲、拉伸刚度分别为 7.905 倍和 2.000 倍。宏观摩擦斜率为 0.446。绝对垂坠约为解析值 0.44 倍，低速蠕移单列。 |
| UMI 模板 | `test/low_order_parcel/check_real_envelope_containment.py`，空模板与合成输入 | 三种区间判定、覆盖率和缺失机制识别通过。真机测量待填。 |
| 原整包回归 | `flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B -m unittest discover -s test/bag_model -p test_physical_parcel.py -v` | 20/20 通过，8.730 s。 |

首轮 CUDA 启动因指定解释器为 CPU 版失败，随后以兼容解释器和缓存依赖成功复跑。YAM 顶抓初始 90° 姿态不可达，采用已验证可达的 270° 姿态。首个折边补充适配器在接触失效后仍施加入口推力，该异常样本退出模型验收。膜片首次时间步未满足原弯曲积分稳定性上界，按计算上界改用 4 µs。

[主审计报告](../docs/PHYSICS_REALISM_AUDIT_REPORT.md) · [运行记录和日志](../test/runs/2026-11-21/bag/physics-realism-audit-r2) · [回归日志](../test/runs/2026-11-21/bag/physics-realism-audit-r2/physical-parcel-final.log)

## 修复后全量复验

最近复验 2026-11-21 19:56（北京时间），机器 `ubun`，基线 `09a0901e`。CPU 和 MuJoCo 使用 `.venv-physical-parcel/bin/python -B`。同一解释器通过项目内的 `.venv-physical-parcel/stage3-cu130` 加载 PyTorch 2.11.0+cu130。GPU 运行使用 RTX 5090 和共享锁。

**CONDITIONAL-GO。** 六组规定门禁通过。17 个逐步能量工况无越限，最差 −0.050156 mJ。静止捕获储能由 6.826 mJ 降至 0 J，YAM 重包首次捕获净耗散由 −9.569 mJ 变为 +0.071 µJ。独立硬折角限位两次激活，最小净耗散 +0.106061 mJ。轻包侧推零捕获。衣物与枕头 YAM 顶压力带仍超时，重包闭合后未离台，UMI 实物包络待填。[门禁结论](../docs/PHYSICS_REALISM_AUDIT_REPORT.md#修复与复验结论)

| 检查 | 记录 |
| --- | --- |
| 逐步能量与独立捕获、阻尼、限位 | [17 工况和逐步 CSV](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/gate1-final/) · [三项定向单测](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/gate1-isolated-final.log) |
| 三张 GPU 相图 | [相图、CSV 与判定](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/gate2-phases/) |
| P6 8,192 路 | [指标](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/gate3-p6/p6.json) · [原 6383 修复后波形与参数](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/gate3-p6/) |
| 代表性 YAM | [闭环日志](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/final-yam.log) · [轻包 22.961 mm、5909 帧零捕获](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/gate4-light-travel.log) |
| Stage 2／2.5 | [Stage 2 七项](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/final-stage2-unit.log) · [1 ms](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/final-1ms/alignment_report.json) · [2 ms](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/final-2ms/alignment_report.json) · [刚体 8,192 路](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/final-stage2-cuda.json) · [P1–P4](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/final-p1-p4.log) · [P5–P6](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/final-p5-p6.log) |
| Stage 3A 与原整包 | [CPU 11/11](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/gate6-cpu.log) · [CUDA 11/11](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/gate6-cuda.log) · [整包 20/20](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/final-physical-parcel.log) |

复测中，旧能量输入在反馈探针把上一变速段外推为下一段，改用同一物理指令轨迹的相邻指垫端点。硬折角工况剩余越限发生在刚性顶压退出，势能割线原分母遗漏了越过零穿入后仍继续回撤的距离，保留完整带符号坐标后低于阈值。低入口另修一次，外侧名义进给在触壁前也由几何限制，去掉了提前捕获。Stage 3A 轻重压台架延长聚拢并调整压力跟踪，以实测维持原 0.3 N／3 N 压力目标与原容差。

```sh
export OMP_NUM_THREADS=1
run=/tmp/parcel-passivity-recheck
mkdir -p "$run"
.venv-physical-parcel/bin/python -B test/low_order_parcel/audit_yam.py "$run/yam" stage2-release low-side-push heavy-raised-fold book-top clothes-top pillow-top
.venv-physical-parcel/bin/python -B test/low_order_parcel/audit_energy_passivity.py "$run/energy" --yam-root "$run/yam"
.venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_contact_passivity.py -v
.venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_stage2_proxy.py -v
.venv-physical-parcel/bin/python -B -m src.low_order_parcel.verify_stage2_alignment --dt .001 --output "$run/1ms"
.venv-physical-parcel/bin/python -B -m src.low_order_parcel.verify_stage2_alignment --dt .002 --output "$run/2ms"
.venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_unified_payload.py -v -k p1 -k p2 -k p3 -k p4 -k test_top
.venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_stage3_dual_arm.py -v
.venv-physical-parcel/bin/python -B -m unittest discover -s test/bag_model -p test_physical_parcel.py -v
export PYTHONPATH="$PWD/.venv-physical-parcel/stage3-cu130:$PWD"
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B test/low_order_parcel/audit_phase_diagrams.py "$run/phases"
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B -m src.low_order_parcel.audit_physics_realism "$run/p6"
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B test/low_order_parcel/benchmark_stage2_cuda.py --worlds 8192 --output "$run/stage2-cuda.json"
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_unified_payload.py -v -k p5 -k p6
flock /tmp/claude-gpu.lock env STAGE3_TEST_DEVICE=cuda .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_stage3_dual_arm.py -v
```

## Fix 4 有限指垫抗转计划

本轮使用已经通过 Fix 1–3 的 `d6e519bc` 物理底座，历史 Stage 3B 对照为 `09a0901e`。正式运行在 `ubun` 的 RTX 5090 上持锁，步长 2 ms。各配置正式运行一遍，同一种子固定对比。

- Gate 1 保留 17 组能量回归，新增悬空抗转、轻夹打滑及零合力检查。所有逐步净耗散至少 −0.1 mJ，捕获转动储能 0 J。
- Gate 2 关闭 P6 平移反馈，同种子 1726481 分别用零半径和有限半径运行 8,192 路 × 2,385 步。报告可稳悬数量、峰值张力、墙钟和显存。1.5 kg 偏心硬书分别验证低夹打滑和高夹稳持。
- Gate 3 用种子 347、每组 1,024 个动作复跑 C1–C10 和 C7 21.6 s，沿用原 3 s 稳持及交接判据。Gate 4 复跑三张 32×32 相图。Gate 5 执行 Stage 2、2.5、3 系列、随机化与整包回归。
- 门禁全部通过才冻结物理底座。若失效，保留原验收阈值，记录失败范围与证据。

## Fix 4 复验与冻结

2026-11-21，ubun，RTX 5090，2 ms。五组规定门禁通过。Stage 4 仿真训练使用本轮冻结的低阶物理。保留前轮 YAM 软料顶压、重包离台和 UMI 实物包络限制。

| 门禁 | 结果 |
| --- | --- |
| 1 · 逐步能量 | 有限指垫 **19/19**，最小净耗散 −0.050156 mJ，低于 −0.1 mJ 的步数为 0。首次捕获抗转储能 0 J。四子块力偶合力最大分量 **0.030 µN**，力矩残差最大分量 3.73 nN·m。[总表](../test/runs/2026-11-21/bag/parcel-pad-torsion-r1/gate1-finite-pad/energy-summary.json) · [力偶核对](../test/runs/2026-11-21/bag/parcel-pad-torsion-r1/couple-balance.log) |
| 2 · 开环与打滑 | 同种子同匀速竖提，关闭 P6 平移反馈。零半径到 12 mm，稳悬 **84 → 5,481 / 8,192**，峰值张力 **20.702 → 22.019 N**，墙钟 **4.114 → 5.213 s**，进程显存 **4,738 → 4,740 MiB**。两组均零非有限、零几何越界、零超过 50 N。1.5 kg 偏心硬书 18 N/指时参考滑移 0.943 rad，末角 55.41°。90 N/指时零滑移，末角 9.73°，末 0.2 s 最大角速度 0.01453 rad/s。[零半径 P6](../test/runs/2026-11-21/bag/parcel-pad-torsion-r1/p6-open-zero/p6.json) · [12 mm P6](../test/runs/2026-11-21/bag/parcel-pad-torsion-r1/p6-open-pad-final/p6.json) · [专项 3/3](../test/runs/2026-11-21/bag/parcel-pad-torsion-r1/pad-torsion-final.log) |
| 3 · 角点 | C1–C10 每组 1,024 个动作均有通过。C3 462 个，C7 511 个，C9 1,024 个，C10 89 个。C7 21.6 s 压力跑 512 个通过，零撕裂，最长稳持 16.182 s。[角点与历史对照](stage3b-corners-test.md#fix-4-有限指垫复跑) |
| 4 · 相图 | 有限指垫下三张 32×32 GPU 相图均通过。A 硬书边界零误捕获、零离台。B 零逆序。C 低区 84 格零捕获，高区 54 格全双指捕获。[相图与判定](../test/runs/2026-11-21/bag/parcel-pad-torsion-r1/phases-finite-pad/) |
| 5 · 回归 | Stage 2 七项通过，1 ms／2 ms 各 6/6，刚体 CUDA 8,192/8,192，1.393 s。P1–P6 与挂点虚功通过，Stage 3 CPU 与 CUDA 各 14/14，随机化 2/2，原整包 20/20。零半径专项与原封装逐元素一致。[运行日志](../test/runs/2026-11-21/bag/parcel-pad-torsion-r1/) |

初版分开预测扭转冲量和原有载荷，高夹仍有缓慢倾斜。最终先由原有载荷预测自由角速度，再解惯量耦合的隐式力矩。参考姿态更新包含摩擦截断后的滑移，膜颈变短时的已有储能受上一时刻结余约束。YAM 能量沿用历史 2 ms 指垫边界，腕部旋转由存档关节轨迹插值得到。

    export OMP_NUM_THREADS=1
    run=/tmp/parcel-pad-torsion-check
    mkdir -p "$run"
    yam=test/runs/2026-11-21/bag/parcel-passivity-fix-r1/final-yam
    .venv-physical-parcel/bin/python -B test/low_order_parcel/audit_energy_passivity.py "$run/energy" --finite-pad --yam-root "$yam"
    .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_pad_torsion.py -v
    export PYTHONPATH="$PWD/.venv-physical-parcel/stage3-cu130:$PWD"
    flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B -m src.low_order_parcel.audit_physics_realism "$run/p6-zero" --open-loop
    flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B -m src.low_order_parcel.audit_physics_realism "$run/p6-pad" --open-loop --pad-radius .012
    flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B test/low_order_parcel/audit_phase_diagrams.py "$run/phases" --finite-pad

Stage 2、P1–P6 和整包沿用本报告上一节命令。Stage 3 CPU 和 CUDA 用 test_stage3*.py，另运行 test_domain.py。角点命令见[角点报告](stage3b-corners-test.md#fix-4-有限指垫复跑)。

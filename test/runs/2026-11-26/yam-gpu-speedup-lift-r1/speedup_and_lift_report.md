# Yam_B MJWarp 求解器提速与捏膜提拉复测

## 本轮计划

机器 `ubun`，RTX 5090 32607 MiB，MuJoCo / MJWarp 4.0.1。所有 GPU 任务持同一外层锁。以原 A2 物理场景为主对照，每档采 5 个 25 Hz 控制帧，记录所有子步的接触数、活跃约束、溢出、边长比和力。缩容量后，同 GPU 比较 200 / 400 / 500 µs 与对应迭代上限。选稳定档后以 256 路、5 帧复测。悬停筛选要求零溢出、零非有限状态和零硬停，膜边长比不超过 1.12。接触阶段用实际抓放和试探复核。

捏膜在原 A2 种子与邻近抓姿上比较闭合、提拉数据。保留 8 N 冻结、11 N 泄压及原物理硬门禁。候选通过后用随机内芯偏置复测，要求至少 3 条连续真实入篮。试探固定 physical-r4 起手，左位 1.6 N 子步卸载，要求三个位置撤回、获得 H_try，峰值低于 3.5 N 且侵入、露角、旁路均为零。两个物理门禁与提速都通过才启动 C1/C2/C3。

## 只读基线配置

| 场景 | dt | 子步/帧 | iterations | ls_iterations | solver | integrator | 单路接触预算 | 单路约束预算 | 单路 Jacobian nnz 预算 |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 历史 nominal 分项 | 200 µs | 200 | 300 | 50 | Newton | discrete | 1024 | 4096 | 32768 |
| 本轮原 A2 捏膜基线 | 200 µs | 200 | 360 | 75 | Newton | discrete | 1024 | 4096 | 32768 |

400 µs 与 500 µs 均通过现有显式折弯步长上界检查。历史 9,724 ms/帧是扣除已计时代码后的原生物理图余量，本轮额外拆出 `solver.solve` 实测。

## RTX 5090 悬停求解器实测

每档预热并捕获一次 CUDA Graph，随后计时 5 帧。墙时含 25 Hz 状态读回和 OBB 核查。求解时间由每个子步 `solver.solve` 前后的 CUDA event 相加。显存为逐帧采样的实验进程峰值。硬停后继续跑满诊断帧的配置无资格入选。

| 档位 | 路 | dt (µs) | Newton/CG 上限 | 线搜索上限 | 单路接触/约束 | ms/帧 | 求解 ms/帧 | env-steps/s | 显存 MiB | 峰值膜边长比 | 悬停筛选 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [A2 基线](benchmark-a2-base-64/profile.json) | 64 | 200.0 | 360 | 75 | 1024/4096 | 15,348.4 | 14,977.6 | 4.17 | 1665 | 1.01015 | **通过** |
| [仅缩容量](benchmark-compact-64/profile.json) | 64 | 200.0 | 360 | 75 | 112/352 | 15,700.5 | 15,351.8 | 4.08 | 1401 | 1.00968 | **通过** |
| [A](benchmark-a-64/profile.json) | 64 | 400.0 | 20 | 25 | 112/352 | 5,908.3 | 5,695.6 | 10.83 | 1301 | 1.48428 | 失败 |
| [B](benchmark-b-64/profile.json) | 64 | 500.0 | 12 | 12 | 112/352 | 2,973.9 | 2,813.3 | 21.52 | 1277 | 1.96155 | 失败 |
| [A 恢复收敛余量](benchmark-a-stable-64/profile.json) | 64 | 400.0 | 84 | 50 | 112/352 | 7,424.1 | 7,202.2 | 8.62 | 1301 | 1.00991 | **通过** |
| [B 恢复收敛余量](benchmark-b-stable-64/profile.json) | 64 | 500.0 | 84 | 50 | 112/352 | 5,841.0 | 5,646.0 | 10.96 | 1277 | 1.01269 | **通过** |
| [B 容差 1e-5](benchmark-b-tol-1e-5-64/profile.json) | 64 | 500.0 | 84 | 50 | 112/352 | 5,680.1 | 5,486.8 | 11.27 | 1277 | 1.01057 | **通过** |
| [B 容差 5e-5](benchmark-b-tol-5e-5-64/profile.json) | 64 | 500.0 | 84 | 50 | 112/352 | 5,593.9 | 5,418.5 | 11.44 | 1277 | 1.00834 | **通过** |
| [补测 CG](benchmark-cg-64/profile.json) | 64 | 500.0 | 100 | 50 | 112/352 | 1,104.7 | 609.3 | 57.93 | 1209 | 2.43579 | 失败 |
| [补测 666.7 µs](benchmark-newton-666-64/profile.json) | 64 | 666.7 | 96 | 60 | 112/352 | 4,325.7 | 4,155.1 | 14.80 | 1259 | 1.01302 | **通过** |
| [补测 800 µs](benchmark-newton-800-64/profile.json) | 64 | 800.0 | 96 | 60 | 112/352 | 3,522.8 | 3,370.2 | 18.17 | 1251 | 1.01748 | **通过** |
| [800 µs 放量](benchmark-newton-800-256/profile.json) | 256 | 800.0 | 96 | 60 | 112/352 | 4,529.6 | 4,236.3 | 56.52 | 1637 | 1.01644 | **通过** |

- 原 A2 的活跃接触峰值 **60/路**，约束峰值 **189/路**，实际 Newton 迭代峰值 **37**。求解占完整控制帧墙时 **97.58%**。仅缩容量节省 264 MiB，墙时无可测提速。旧轨迹的 25 Hz 原生几何回放见到 111 个接触与 363 条约束，物理验收使用 224/736 的独立裕量。
- A/B 在迭代上限前未收敛，分别超限至 1.48、1.96 的边长比。CG 同样触发未收敛、容量告警与几何失效。MJWarp 将默认 XML 的 `1e-8` 抬高为设备有效 `1e-6`。将设备容差改为 `5e-5` 仍只使 500 µs 档达到约 2.74×。
- 增测的 666.7 和 800 µs 均满足 250 Hz 执行器、125 Hz 命令、25 Hz 控制时钟的整除约束与现有显式折弯步长上限。800 µs 的 64 路同口径比原 A2 快 **4.36×**。同机历史 256 路诊断约 14.47 env-steps/s，当前 56.52 属于 5 帧悬停测量，历史数据含不同负载及不活跃环境。

## A2 原种子失效诊断

| 时刻 | 实际开度 (mm) | 历史记录 L/R (N) | TCP−内芯质心 dx/dy/dz (mm) | 原生几何回放 |
| --- | ---: | --- | --- | --- |
| 2.32 s 闭合末 | 0.093 | 5.732 / 4.710 | +3.75 / −63.50 / +2.46 | 双指均接触膜 |
| 2.36 s 提拉首帧 | 0.026 | 5.463 / 5.007 | +4.16 / −63.61 / +3.18 | 双指均接触膜 |
| 2.48 s 提拉 | −0.017 | 7.938 / 3.006 | +7.41 / −63.51 / +14.62 | 指垫互触，仍有指垫–膜接触 |
| 2.52 s 提拉 | −0.012 | 8.001 / 0.000 | +9.49 / −63.41 / +21.04 | 只剩左右指垫互触 |
| 3.56 s 高位 | −0.012 | 8.001 / 0.000 | +15.42 / −61.63 / +170.53 | 只剩左右指垫互触 |

原抓点在内芯近侧边以外约 28 mm，闭合到接近零开度。提拉前段膜脱离指垫，内芯峰值抬升 0.138 mm。MJWarp 4.0.1 复用接触槽时，刚体录入保留旧的 Flex 字段。膜力、软接触距离和材料覆写现统一依据端点几何有效性识别刚体–柔体。假正力不会触发膜力限幅或计入双指托举。保留 8 N / 11 N 子步限幅与全部硬门禁。

原轨迹时间依次为 reset 0.52、接近前 0.48、接近 0.32、闭合 1.00、提拉 1.20、搬运 2.04、放置 0.92、释放 1.36、撤离 0.92、篮内稳态 0.52 秒。压缩候选保留原闭合和提拉速度，采用高位直接搬运、提高空走速度及缩短释放开度和冗余等待。静态 IK 按相同刚体碰撞判据过滤，−3 mm 与 −4 mm 的竖直下移未通过该筛选，改用最深可行的 −2.5 mm。低角度候选的规划耗时约 6.5–6.6 秒，实际有效性以回合日志为准。[原始回放抽样](a2-archived-contact-replay.json)

几何回放还显示，闭合末实际接触法线在夹爪闭合轴上的绝对投影最高为 0.323，提拉到 2.48 s 时仅 0.003。那几帧记录的总法向力不能等同于指间夹紧力。

## 800 µs 接触验收

| 试探位置 | 卸载触发 N | 峰值 N | H_try 帧数 × 48 | 侵入/露角/旁路 | 撤回、v1 | 停因 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| [−18 mm](physical-screen-800/r4-left/summary.json) | 1.659 | 2.752 | 54×48 | 0/0/0 | 是 | 通过 |
| [0 mm](physical-screen-800/r4-center/summary.json) | 1.803 | 12.645 | 15×48 | 0/0/0 | 否 | contact_force_limit |
| [+18 mm](physical-screen-800/r4-right/summary.json) | 1.974 | 11.608 | 15×48 | 0/0/0 | 否 | contact_force_limit |

| 捏膜候选 | 实际终止阶段 | 模拟秒 | 膜力峰值 N | 峰值边长比 | 硬停距离 mm | 接触对 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [film-a2-reference](physical-screen-800/film-a2-reference/summary.json) | close | 1.36 | 19.34 | 1.05648 | -1.202 | flex ↔ flex |
| [film-cross-24](physical-screen-800/film-cross-24/summary.json) | approach | 2.36 | 23.91 | 1.04459 | -2.045 | left_pad ↔ flex |
| [film-diagonal-18](physical-screen-800/film-diagonal-18/summary.json) | approach | 1.24 | 24.22 | 1.04895 | -1.211 | payload_geom_0 ↔ flex |
| [film-diagonal-right](physical-screen-800/film-diagonal-right/summary.json) | approach | 1.24 | 19.94 | 1.08519 | -2.004 | left_pad ↔ flex |
| [film-low-12](physical-screen-800/film-low-12/summary.json) | approach | 1.24 | 7.33 | 1.04529 | -1.058 | flex ↔ flex |
| [film-low-20](physical-screen-800/film-low-20/summary.json) | approach | 1.24 | 22.37 | 1.03102 | -2.053 | right_pad ↔ flex |
| [film-low-24](physical-screen-800/film-low-24/summary.json) | approach | 1.24 | 25.34 | 1.04164 | -2.097 | right_pad ↔ flex |
| [film-tilt-20](physical-screen-800/film-tilt-20/summary.json) | approach | 1.88 | 20.11 | 1.01193 | -2.035 | left_pad ↔ flex |

800 µs 接触门禁结果为试探 1/3、入篮 0/8。捏膜硬停来自垫–膜、膜自接触或内芯–膜阈值。改在 500 µs / 84 / 50 复验，保留同一子步控制，捏膜降低下降速度并增加封边和靠近内芯的上缘候选。

## 500 µs 捏膜与 nominal 666.7 µs 试探复测

500 µs / 84 / 50 的封边筛选并行跑 11 路，用时 471.0 秒，运行 88 个控制帧。两条严格物理有效的捏膜候选完成提拉阶段，随后按 `not_lifted` 提前结束。另六条在接近或闭合触发穿透护栏。16 条捏膜诊断共 0 条入篮。

| 500 µs 捏膜候选 | 终止阶段 | 模拟秒 | 墙时秒 | 双指稳持中位 N | 峰值膜力 N | 边长比 | 峰值抬升 mm | 停因 |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| [film-diagonal-ridge](physical-seam-500/film-diagonal-ridge/summary.json) | approach | 1.48 | 203.5 | — | 18.96 | 1.08303 | — | excessive_contact_penetration |
| [film-near-low](physical-seam-500/film-near-low/summary.json) | close | 1.92 | 261.9 | 6.098 / 4.477 | 10.43 | 1.08540 | — | excessive_contact_penetration |
| [film-near-ridge](physical-seam-500/film-near-ridge/summary.json) | close | 1.88 | 256.6 | 13.172 / 7.417 | 17.20 | 1.05412 | — | excessive_contact_penetration |
| [film-reference](physical-seam-500/film-reference/summary.json) | approach | 1.44 | 198.6 | — | 5.26 | 1.04322 | — | excessive_contact_penetration |
| [film-seam-300](physical-seam-500/film-seam-300/summary.json) | close | 1.76 | 240.3 | 3.162 / 3.058 | 11.19 | 1.04003 | — | excessive_contact_penetration |
| [film-seam-303-diagonal](physical-seam-500/film-seam-303-diagonal/summary.json) | lift | 3.52 | 471.0 | 0.012 / 0.000 | 1.13 | 1.01093 | 0.153 | not_lifted |
| [film-seam-306](physical-seam-500/film-seam-306/summary.json) | lift | 3.44 | 461.0 | 0.242 / 0.000 | 1.66 | 1.01366 | 0.010 | not_lifted |
| [film-seam-cross](physical-seam-500/film-seam-cross/summary.json) | close | 2.52 | 340.4 | 1.226 / 0.000 | 23.30 | 1.03555 | — | excessive_contact_penetration |

| 配置 | 位置 | 子步触发 N | 试探峰值 N | H_try 帧数 | 侵入/露角/旁路 | 撤回、v1 | 结果 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| A2 500 µs | [left](physical-seam-500/r4-left/summary.json) | 1.623 | 1.825 | 53 | 0/0/0 | 是 | 通过 |
| A2 500 µs | [center](physical-seam-500/r4-center/summary.json) | 2.403 | 15.227 | 15 | 0/0/0 | 否 | contact_force_limit |
| A2 500 µs | [right](physical-seam-500/r4-right/summary.json) | 3.363 | 13.713 | 15 | 0/0/0 | 否 | contact_force_limit |
| nominal 666.7 µs | [left](physical-r4-nominal-666/left/summary.json) | — | 0.000 | 18 | 0/1/0 | 否 | film_core_geometry |
| nominal 666.7 µs | [center](physical-r4-nominal-666/center/summary.json) | 2.022 | 2.022 | 54 | 0/0/0 | 是 | 通过 |
| nominal 666.7 µs | [right](physical-r4-nominal-666/right/summary.json) | 1.846 | 2.516 | 54 | 0/0/0 | 是 | 通过 |
| nominal 666.7 µs | [center-slow](physical-r4-nominal-666/center-slow/summary.json) | — | 0.000 | — | 0/1/0 | 否 | film_core_geometry |
| nominal 666.7 µs | [right-slow](physical-r4-nominal-666/right-slow/summary.json) | — | 0.722 | 31 | 0/1/0 | 否 | film_core_geometry |

封边 −306 mm 及 −303 mm 斜向两例在提拉期间的双指承力帧均为 0，峰值内芯抬升分别为 0.010 mm 和 0.153 mm。静态侧入筛选在原工作胞无碰撞 IK 下找到的候选，未在开度扫描中同时具备两侧可用接触。压缩路径已通过静态规划，实际未完成物理有效的全程入篮，4.5–5.5 s 目标与释放后净空没有得到实测认证。

加速档分别复现了左、中、右的安全撤回和 H_try。A2 500 µs 是 1/3，原 nominal 场景 666.7 µs 是 2/3。同一求解器与场景尚未同时达到 3/3。捏膜 0/16，成功视频、C1、C2、C3 未生成。

## 复现

仓库根目录下运行。GPU 命令通过同一外层锁和指定解释器。

```bash
OUT=test/runs/2026-11-26/yam-gpu-speedup-lift-r1
RUN=test/runs/2026-11-26/yam-gpu-flex-rl-scaled-r1/run_gpu.sh
bash "$RUN" test/policy_v3/yam_gpu_solver_sweep.py "$OUT/replay-800-256" \
  --worlds 256 --timestep .0008 --iterations 96 --ls-iterations 60 \
  --tolerance 1e-6 --contacts 112 --constraints 352
bash "$RUN" test/policy_v3/yam_gpu_acceptance.py "$OUT/replay-seam-500" \
  --batched --solver-profile a2 --case-file "$OUT/physical-seam-cases.json" \
  --timestep .0005 --iterations 84 --ls-iterations 50 --tolerance 1e-6 --contacts 224 --constraints 736
bash "$RUN" test/policy_v3/yam_gpu_acceptance.py "$OUT/replay-r4-666" \
  --batched --solver-profile nominal --case-file "$OUT/physical-r4-nominal-cases.json" \
  --timestep 0.0006666666666666666 --iterations 96 --ls-iterations 60 \
  --tolerance 1e-6 --contacts 224 --constraints 736
```

[冻结路径核验](frozen-check.log) · [Policy V3 12 项](policy-v3-regression.log) · [子步和接触归类回归](yam-substep-regression.log) · [测试报告](../../../../reports/yam-gpu-speedup-lift-test.md)

# Stage 3A 双臂 Step 1 测试报告

机器 `ubun`，RTX 5090。冻结基线 `96a896c1`，Stage 3A 使用 2 ms、float32。最近验证 2026-11-21 18:34（北京时间）。

| 检查 | 结果 |
| --- | --- |
| Stage 3A CPU | 11/11 通过，PyTorch 2.15.0+cpu。 |
| Stage 3A CUDA | 11/11 通过，PyTorch 2.11.0+cu130。 |
| Stage 2 冻结代理 | 7/7 通过，包含 1 ms、2 ms 各百万步。 |
| Stage 2 对齐门禁 | 1 ms、2 ms 各 6/6 通过。 |
| Stage 2.5 P1–P4 与挂点虚功 | 7/7 通过。 |
| Stage 2.5 CUDA P5–P6 | 2/2 通过。P5 为 32 路各 100,000 步。P6 为 8,192 路完整回合，3.668 s 内 8,003 路达到原稳悬标准。 |
| Stage 2 CUDA | 8,192/8,192 路达到原稳悬标准，完整回合批次用时 1.079 s。 |
| 原整包回归 | 20/20 通过。 |

## 复现

在仓库根目录使用指定解释器。对齐命令的输出参数用本轮目录复现时需另选空目录。

```sh
export OMP_NUM_THREADS=1
.venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_stage3_dual_arm.py -v
.venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_stage2_proxy.py -v
.venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_unified_payload.py -v -k p1 -k p2 -k p3 -k p4 -k test_top
.venv-physical-parcel/bin/python -B -m unittest discover -s test/bag_model -p test_physical_parcel.py -v
.venv-physical-parcel/bin/python -B -m src.low_order_parcel.verify_stage2_alignment --dt .001 --output /tmp/stage3a-alignment-1ms
.venv-physical-parcel/bin/python -B -m src.low_order_parcel.verify_stage2_alignment --dt .002 --output /tmp/stage3a-alignment-2ms
git diff --exit-code 96a896c1 -- src/low_order_parcel/lumped_contact_proxy.py src/low_order_parcel/unified_payload.py src/low_order_parcel/macro_parcel_sim.py
git diff --check
```

CUDA 通过同一解释器加载项目内独立的 `stage3-cu130` 依赖目录。完整依赖安装遇到其他任务持有的下载缓存锁，改用 Python 3.14 的 `torch==2.11.0+cu130` wheel，并链接本机 Isaac Sim 环境中匹配的 `nvidia/` 运行库。所有 CUDA 回归持共享 GPU 锁。

```sh
export PYTHONPATH="$PWD/.venv-physical-parcel/stage3-cu130:$PWD"
flock /tmp/claude-gpu.lock env STAGE3_TEST_DEVICE=cuda .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_stage3_dual_arm.py -v
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_unified_payload.py -v -k p5 -k p6
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B test/low_order_parcel/benchmark_stage2_cuda.py --worlds 8192 --output /tmp/stage3a-stage2-cuda.json
```

[Stage 3A CPU](../test/runs/2026-11-21/policy/stage3a-dual-arm-r1/stage3a-cpu.log) · [Stage 3A CUDA](../test/runs/2026-11-21/policy/stage3a-dual-arm-r1/stage3a-cuda.log) · [Stage 2](../test/runs/2026-11-21/policy/stage3a-dual-arm-r1/stage2.log) · [1 ms 对齐](../test/runs/2026-11-21/policy/stage3a-dual-arm-r1/alignment-1ms.log) · [2 ms 对齐](../test/runs/2026-11-21/policy/stage3a-dual-arm-r1/alignment-2ms.log) · [Stage 2.5](../test/runs/2026-11-21/policy/stage3a-dual-arm-r1/stage25.log) · [P5–P6](../test/runs/2026-11-21/policy/stage3a-dual-arm-r1/stage25-cuda.log) · [Stage 2 CUDA](../test/runs/2026-11-21/policy/stage3a-dual-arm-r1/stage2-cuda.json) · [整包](../test/runs/2026-11-21/policy/stage3a-dual-arm-r1/physical-parcel.log)

## 关键数据与复测

- `F_half=0` 时，单活动臂的接触状态、指张力和合力与原代理逐元素相等。两臂同时聚拢和滑移时，剩余与当步抽离余料合计均在 0.150 mm 预算内。
- 同袋衣物在 0.3 N 和 3.0 N 指令下，聚拢期实测均值为 0.291 N 和 2.897 N。最大压陷 0.743 mm 和 5.424 mm。轻压零捕获，重压双指捕获、桌面支撑为零、离台 37.279 mm。
- 20 N 门限下 40.00 N 试算张力清零，60 N 门限下 343.97 N 试算张力清零。未越限指当步继续传力。空中侧推平移 22.512 mm，名义进给 11.700 mm 对应末帧净进给零。
- 65 g 拉拔的双臂真值峰值均为 5.126 N。在 0.75 增益、2 N 摩擦项、4 Hz 低通、0.6 N 噪声和 50 ms 延迟下，CUDA 两臂估力峰值 3.761 N 和 4.017 N，连续 76 ms 同时高于 3.5 N。CPU 为 80 ms。离台最小 160.0 mm，试验结束未撕裂且两臂双指保持捕获。
- 首轮轻重压台架受瞬时压力波动影响，改用低通下压伺服并在聚拢窗取均值。拉拔台架首轮指垫穿入厚度包络，修正为初始预载的包裹轮廓外拉拔。腕相机首轮夹具的光轴写错，按右、下、前列向量修正，复测通过。
- 空中侧推初次在 CPU 与 CUDA 分别测到 3.011 N 和 29.778 N。侧壁命中与横向夹压同时施力，导致刚体包裹形变对数值误差敏感。双臂封装将入射侧壁接触与横向夹压互斥后，两端均为 3.011 N 峰值、22.512 mm 位移、净进给零。新增横向形变上限检查通过。[修复前逐步记录](../test/runs/2026-11-21/policy/stage3a-dual-arm-r1/air-side-diagnostic.log)
- Step 1 台架从已抓持状态验证 65 g 拉拔。相机标定误差与估力为仿真注入。完整交接、真机迁移与双臂吞吐需要后续阶段的独立验证。

## 无源物理修复后复跑

2026-11-21 19:56（北京时间），`ubun`。CPU 与 RTX 5090 均 11/11 通过。轻重压台架保持原目标与容差，CUDA 实测 0.280／2.883 N，重压双指捕获并离台 21.540 mm。65 g CUDA 拉拔双臂真实峰值均为 5.126 N，双侧退化读数同时超过 3.5 N 达 76 ms。逐指撕裂与余料共有量检查通过。双臂侧推改用无反弹接触，CUDA 测得 0.817 N 峰值、10.383 mm 位移，名义 11.700 mm 对应末帧净进给 1.617 mm。[CPU 日志](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/gate6-cpu.log) · [CUDA 日志](../test/runs/2026-11-21/bag/parcel-passivity-fix-r1/gate6-cuda.log) · [六组门禁与命令](physics-realism-audit-test.md#修复后全量复验)

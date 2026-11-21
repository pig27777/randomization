# Stage 3B Step 2 检查报告

2026-11-21 19:03（北京时间），`ubun`，RTX 5090，PyTorch 2.11.0+cu130，float32，步长 2 ms。物理台架基于 `85cb6f0e`。角点配置 SHA-256 为 `99d42da06997aabcd80bf73d26f880407b6b6afcfa3427129802777fc1ed55ee`。

**通过。** 16,384 路全宽采样的 CUDA 整袋刚度覆盖 4.000–4,997.376 N/m，硬包占 34.6%，侧向遮挡占 30.1%。CPU 与 CUDA 的 Stage 3 固定角点、分布、局部复位、课程和 Stage 3A 回归各通过 14 项，原域随机化回归 2 项通过。冻结代理、模板和单臂台架文件与基线一致。

## 固定角点

种子 347，每角 1,024 个 Sobol 动作，统一运行 50.545 s 墙钟。抓取通过要求双指抓牢、离台大于 1 mm、桌面支撑为零、质心速度低于 20 mm/s 连续 3 s。交接通过要求接取臂独立稳悬至少 0.5 s，腕相机预观察有效且切入近场后无效。通过动作全段无掉台和撕裂。表中连续时间对应保存的可行动作，十条证据的首次捕获指张力均为零。

| 角点 | 通过数 | 证据行 | 路线 | 连续稳持 s |
| --- | ---: | ---: | --- | ---: |
| C1 超滑气泡膜 | 512 | 903 | 侧抓 | 18.108 |
| C2 90% 填充硬书 | 496 | 903 | 侧抓 | 18.210 |
| C3 1.5 kg 偏心与低右指摩擦 | 3 | 639 | 侧抓 | 9.762 |
| C4 低聚拢硬纸板袋 | 456 | 811 | 侧抓 | 18.292 |
| C5 最坏硬件退化 | 737 | 425 | 顶抓 | 18.692 |
| C6 65 g 轻包与低桌面摩擦 | 510 | 40 | 侧壁上方切入 | 18.248 |
| C7 30% 填充松散重物 | 1 | 631 | 偏置挂点侧抓 | 8.156 |
| C8 两侧遮挡紧绷厚被 | 579 | 903 | 顶抓 | 18.718 |
| C9 0.59 rad 垂折交接 | 124 | 61 | 交接 | 6.848 |
| C10 65 g 主动拉拔交接 | 87 | 241 | 交接 | 3.394 |

- C1 用 5 N/s 切向增载在 **0.560 N** 脱手。此前零桌面支撑，滑移持续 110 ms。
- C7 的挂点由动作选择并在整个回合固定，仍使用同一质量分布、填充、内部滑移和撕裂参数。16.5 s 固定动作验真中，证据行全段无掉台或撕裂，末帧离台 160 mm，峰值指张力 6.437 N。同一动作在 21.6 s 压力跑中先稳悬 9.636 s，随后撕裂掉台。3 s 验收阈值沿用原计划。[有限回放逐步记录](../test/runs/2026-11-21/policy/stage3b-corners-r1/pick-c7-16s.trace.json) · [延长压力跑](../test/runs/2026-11-21/policy/stage3b-corners-r1/pick-347-site.json)
- C9 实测初始折角 0.590 rad、初始摆角 15°。18 cm 腕部盲区在回放中实际生效。C10 双侧真实对拉峰值约 4.494 N，在 2 N 库仑项、50 ms 延迟与噪声下估力同时超过 3.5 N 达 **40 ms**。
- Step 2 检验低阶指垫仿真中的存在性。开环动作使用初始工作站几何。柔顺腕响应所受机械载荷。侧向遮挡由可用进给空间和动作限制表达。未验证机械臂关节可达、真机识别或完整策略成功率。独立物理审计的训练准入仍为 NO-GO。[物理审计](../docs/PHYSICS_REALISM_AUDIT_REPORT.md#a4-训练准入)

## 复测记录

- 初轮 CUDA Graph 首次调用 cuBLAS 失败，增加同流预热。C1 复用原脱手判据时补齐滑移与腕力日志。
- 侧抓先将松弛的褶皱引向包裹中心、再抬升，使 C1、C2、C4、C6 获得稳定解。C3 的慢抬升、偏置和被动腕部柔顺抑制了偏心运动。C7 的居中动作在无掉台候选中仍持续横摆，加入偏置抓取点后得到上表证据。C9 初轮折角实际只有 0.292 rad，修正为规定折角后与 C10 一起复验被动腕部柔顺、盲区和交接。
- CUDA 局部复位初轮把未指定序号的 `cuda` 与张量的 `cuda:0` 直接比较，改为按回合张量的实际设备校验，通过 GPU 回归。[初轮 CUDA 单测](../test/runs/2026-11-21/policy/stage3b-corners-r1/stage3-gpu.log)

## 命令和数据

同一项目解释器使用 Step 1 的本机隔离 CUDA 依赖目录。正式运行持共享 GPU 锁。更换输出路径可复跑。

```sh
export OMP_NUM_THREADS=1
.venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p 'test_stage3*.py' -v
.venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_domain.py -v
export PYTHONPATH="$PWD/.venv-physical-parcel/stage3-cu130"
flock /tmp/claude-gpu.lock env STAGE3_TEST_DEVICE=cuda .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p 'test_stage3*.py' -v
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B test/low_order_parcel/stage3_dual_arm_corners.py --seed 347 --candidates 1024 --output /tmp/stage3b-corners.json
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B test/low_order_parcel/stage3_dual_arm_corners.py --seed 347 --candidates 1024 --corners 7 --loose-seconds 21.6 --trace-row 631 --output /tmp/stage3b-c7-stress.json
git diff --exit-code 85cb6f0e -- src/low_order_parcel/lumped_contact_proxy.py src/low_order_parcel/unified_payload.py src/low_order_parcel/macro_parcel_sim.py
git diff --check
```

[全部候选与证据行](../test/runs/2026-11-21/policy/stage3b-corners-r1/c1-c10-347.json) · [角点日志](../test/runs/2026-11-21/policy/stage3b-corners-r1/c1-c10-347.log) · [CPU 单测](../test/runs/2026-11-21/policy/stage3b-corners-r1/stage3-cpu-verified.log) · [CUDA 单测](../test/runs/2026-11-21/policy/stage3b-corners-r1/stage3-gpu-final.log) · [旧版回归](../test/runs/2026-11-21/policy/stage3b-corners-r1/legacy-domain.log)

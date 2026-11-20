# RSS 2027 Stage 2 测试报告

最近验证为 2026-11-20 19:16（北京时间），机器 `ubun`，CPU i9-14900KF，项目环境 `.venv-physical-parcel`。本轮新增依赖为 `torch==2.15.0+cpu` 和 `matplotlib==3.11.2`。

| 检查 | 结果 |
| --- | --- |
| 包络和配置生成 | Milestone 1 的 3 项均通过。Stage 2 配置与生成器输出一致，所用冻结档案及哈希一致。 |
| Gate B 双指代理 | 7 项通过，20.3 s。1 和 2 ms 各 100,000 步，批大小 16，零非有限量、零负余料，峰值张力低于输入推得的上界。 |
| Gate A 刚体与代理对齐 | 两档步长各 6 类全过。预检中 PE 加载脱手前触台。增加探针离台净空后，各正式加载和松夹试验直至脱手的支撑力为 0 N。 |
| Gate C 既有整包回归 | 20 项通过，8.7 s。测试用网格 `crossing_triangle` 产生一条约束警告。 |

```sh
.venv-physical-parcel/bin/python -B src/low_order_parcel/build_stage1_envelopes.py
OPENBLAS_NUM_THREADS=1 .venv-physical-parcel/bin/python -B policy/rss2027/test_stage1_envelopes.py -v
flock /tmp/claude-gpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_stage2_proxy.py -v
flock /tmp/claude-gpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-physical-parcel/bin/python -B src/low_order_parcel/verify_stage2_alignment.py
flock /tmp/claude-gpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-physical-parcel/bin/python -B src/low_order_parcel/verify_stage2_alignment.py --dt .001 --output test/runs/stage2_freeze/dt-1ms
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B -m unittest discover -s test/bag_model -p test_physical_parcel.py
```

重跑对齐时用 `--output` 指定新的空目录。脱手绝对值的主要偏差在气泡膜，PE 在 11.25 mm 进给档的持续双侧承载偏长。耗时为同机器 CPU 单环境实际探针回合。

[冻结报告和偏差明细](../docs/STAGE2_FREEZE_REPORT.md) · [2 ms JSON](../test/runs/stage2_freeze/alignment_report.json) · [1 ms JSON](../test/runs/stage2_freeze/dt-1ms/alignment_report.json)

## Milestone 2 补充验收

2026-11-20 19:29，北京时间，`ubun`。A、B、C 的通过记录见上。D 真正对每个基础步长运行一百万次状态递推，两档各 16 路。E 使用已有的 `egoallo` CUDA 环境，PyTorch 2.7.1+cu128，验证过 RTX 5090 的 sm_120。首次隔离环境下载超时，随后查到已有兼容环境并停止下载。一次动作生成器拆分漏带动作切换时间，修复后重新执行两档六类对齐，均通过。

```sh
flock /tmp/claude-gpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_stage2_proxy.py -k million -v
flock /tmp/claude-gpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/ubuntu/miniconda3/envs/egoallo/bin/python -B test/low_order_parcel/benchmark_stage2_cuda.py --output test/runs/stage2_freeze/milestone2/cuda-throughput.json --worlds 512 2048 4096 8192
```

压力测试两档均通过，共 191.22 s。GPU 四档均完成 2,385 步／4.77 s，零非有限环境，全部达到 0.20 s 悬停门槛。8,192 路整批 1.119 s，17,462,714 环境步/s，PyTorch 预留峰值 2,002 MiB。使用同一探针与 CPU 四材质批次复核，最短末段悬停均为 3.002 s。以上为代理的完整抓取和悬停探针，训练和回合重置另有开销。[压力测试日志](../test/runs/stage2_freeze/milestone2/stress.log) · [完整 GPU 结果](../test/runs/stage2_freeze/milestone2/cuda-throughput.json)

## 双指代理的 MuJoCo 回放

2026-11-20 19:38，北京时间，`ubun`。选用冻结 PE 参数、3 mm 偏心和 2 ms 步长，生成一次悬空加载回合。检查录制位置和姿态可用于回放，并确认原生窗口能显示单侧承载帧。

```sh
run_dir=test/runs/2026-11-20/bag/rss2027-stage2-proxy-ui-r1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-physical-parcel/bin/python -B test/low_order_parcel/view_stage2_proxy.py "$run_dir" --check
MUJOCO_GL=glfw .venv-physical-parcel/bin/python -B test/low_order_parcel/view_stage2_proxy.py "$run_dir"
```

检查通过。录制包含 2,950 帧，右侧 4.014 s 脱手，左侧 4.108 s 脱手。首次打开时，MuJoCo Python 绑定拒绝可视化柔索的空参数。显式传入数组后，原生窗口成功显示，并验证了回放及返回单侧承载帧的快捷键。截图为 1706 × 960。在 4.060 s，左侧张力 7.19 N，右侧 0 N。UI 使用简化包裹几何显示代理的位姿、承载和滑移。
[原生窗口截图](../test/runs/2026-11-20/bag/rss2027-stage2-proxy-ui-r1/mujoco-proxy-ui.png)

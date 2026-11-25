# PE 袋最小可用性测试

最近测试为 2026-11-25 19:39（北京时间）。在 `ubun` 上复用仓库 142 节点封边 Flex 袋和 PE 工程参数。CPU 夹持使用 50 μs，完整通过样本使用 14.4 × 16.4 mm 指垫和 0.20 kg、88 × 68 × 24 mm 内芯。两种夹持各验证 1 s 净空和松爪后的自然下落。模型参数、夹具配置和原始轨迹随回合保存。[推荐配置](../test/runs/2026-11-25/bag/pe-mvp-r1/recommended_config.json)

| 测试 | 结果 | 实测 |
| --- | --- | --- |
| 包覆 | PASS | 0.20／0.35 kg 均随支撑面自然移动约 65 mm，无角点露出 |
| Active Try | PASS | 三组使用相同慢降和 190 mm 扫描。左、中、右对应区段的平均力为 0.758／0.759／0.760 N，各自的对应段最强。连续有效信号为 0.535／0.515／0.420 s |
| 薄膜夹持 | PASS | 封边附近布置。稳持 1 s 期间内芯最低净空 13.3 mm，双侧接触 100%，张开后落下 |
| 隔袋夹持 | PASS | 利用台面边缘获得内芯下方空间。稳持 1 s 期间最低净空 42.7 mm，双侧有效接触 99.0%，松爪后最低点下落 44.2 mm |
| MJWarp 1 路主动接触 | PASS | 200 μs，中心段 0.756 N，包覆通过，零溢出 |
| MJWarp 1 路完整隔袋 | FAIL | 位置重放起抬 44.0 mm，稳持中滑落。设备夹口力闭环复测也滑落，并出现内芯露角和指垫几何侵入 |
| MJWarp 批量主动接触 | PASS | 1／16／64／256 路均有效，范围为 0.12 s 真实移动接触窗 |
| 进入 Yam_B 完整操作 | FAIL | CPU 夹持通过，设备端夹持和力范围仍未通过 |

CPU 的 7 条选定回合逐帧检查，膜面均未切入内缩 2 mm 的名义内芯，两条成功夹持也无指垫切入同一内缩体。夹具—内芯直接碰撞对已屏蔽，两侧载荷均从膜接触读取。两条 CPU 夹持通过轨迹的膜边长比分别不超过 1.037 和 1.035。隔袋回合双侧法向力都大于 0.25 N 的比例为 67.2%，本值保留作力度描述。验收依据为正反力、净空、保持、自然释放及几何检查。[CPU 回合汇总](../test/runs/2026-11-25/bag/pe-mvp-r1/cpu_acceptance.json) · [抓取反力](../test/runs/2026-11-25/bag/pe-mvp-r1/grasp-forces.png)

## 四类视频与曲线

[包覆](../test/runs/2026-11-25/bag/pe-mvp-r1/contain.mp4) · [主动 Try](../test/runs/2026-11-25/bag/pe-mvp-r1/try.mp4) · [薄膜夹持](../test/runs/2026-11-25/bag/pe-mvp-r1/film.mp4) · [隔袋夹持](../test/runs/2026-11-25/bag/pe-mvp-r1/through.mp4) · [Try 反力曲线](../test/runs/2026-11-25/bag/pe-mvp-r1/active-try.png)

## GPU 速度与显存

在通过的单路移动接触状态上初始化，统一运行 600 子步（0.12 s 仿真），200 μs。计时包含设备同步，诊断读回单列。显存为该进程的采样峰值。

| 环境数 | 有效环境步/秒 | 物理墙钟时间（s） | 显存（MiB） | 有效环境 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 37.7 | 15.92 | 978 | 1/1 |
| 16 | 403.6 | 23.79 | 1106 | 16/16 |
| 64 | 1,415.2 | 27.13 | 1396 | 64/64 |
| 256 | 4,304.3 | 35.69 | 2676 | 256/256 |

单路完整主动 Try 为 39.2 步/秒和 978 MiB。完整隔袋位置重放与实际反力闭环复测分别为 44.3 和 56.0 步/秒，两个回合的有效夹持吞吐均为 0。首个单路短窗在写表时遇到 NumPy 布尔类型序列化错误，原轨迹和日志保留。修正后重跑，同表使用重跑记录。[GPU 性能 CSV](../test/runs/2026-11-25/bag/pe-mvp-r1/gpu_performance.csv) · [GPU 夹持原始记录](../test/runs/2026-11-25/bag/pe-mvp-r1/gpu-through-feedback-1/summary.json)

## 执行与检查

使用 MuJoCo/MJWarp 4.0.1、Warp 1.19.0 与 RTX 5090。原有 `test_physical_parcel.py` 的 20 项回归通过。新脚本通过 `py_compile`，四支视频用 FFmpeg 核验。CPU 中位 Try 单独用 200 μs 复测，中心段平均反力为 0.772 N，包覆通过。50 μs 单路 GPU 回合实际运行 11 分钟后主动结束，记录到 978 MiB 进程显存，缺少完整窗口吞吐数据。200 μs 的设备结果见上表。[GPU 50 μs 日志](../test/runs/2026-11-25/bag/pe-mvp-r1/gpu-try-1.log)

在仓库根目录运行以下入口，输出目录使用新的路径。

```sh
PE_PYTHON=/home/ubuntu/workspace/gzl/randomization/.venv-physical-parcel/bin/python

"$PE_PYTHON" -B test/bag_model/bag_mvp.py contain-350 /tmp/pe-contain-350
"$PE_PYTHON" -B test/bag_model/bag_mvp.py try-center /tmp/pe-try-center
"$PE_PYTHON" -B test/bag_model/bag_mvp.py film /tmp/pe-film --variant near-seam
"$PE_PYTHON" -B test/bag_model/bag_mvp.py through /tmp/pe-through --variant exposed-underface --edge .015
flock /tmp/claude-gpu.lock "$PE_PYTHON" -B test/bag_model/bag_mvp_gpu.py through /tmp/pe-gpu-through \
  --dt .0002 --cpu /tmp/pe-through
```

**VBD 备选**跑完相同悬臂轨迹的 2.52 s，1 s 稳持最低净空 35.7 mm，几何检查通过。但稳持两指反力中位数分别约 105.5 和 104.2 N，峰值 323.1 N，远高于本轮 8 N 指垫目标。当前只确认这一接触路径可承载，力度未达到可直接接入的条件。[VBD 备选结果](../test/runs/2026-11-25/bag/pe-mvp-r1/vbd-through-1/summary.json)

两种通过的夹持都依赖可达的下侧空间。普通平面上、远离封边的薄膜抬拉和首次侧面隔袋方案均未通过。Yam_B 的近台面可达性、视觉条件与接触反馈仍需在机器人的真实运动范围中复核。

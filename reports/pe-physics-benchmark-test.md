# PE 袋统一物理基准检查

最近测试为 2026-11-25 18:33（北京时间），主机 `ubun`，Intel Core i9-14900KF 与 RTX 5090。采用 MuJoCo 4.0.1、MJWarp 4.0.1、Warp 1.19.0、Newton 1.6.1。运行于从 `72b558fe` 创建的独立工作树，原主工作树已有其他修改。

## 执行与结果

| 检查范围 | 命令或方法 | 结果 |
| --- | --- | --- |
| 仓库现有整包回归 | `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 $PE_PYTHON -B -m unittest discover -s test/bag_model -p test_physical_parcel.py` | 20 项通过 |
| 源码静态检查 | `python -m py_compile test/bag_model/pe_physics*.py bag_sim/physical_parcel.py` 与 `git diff --check` | 通过 |
| CPU Flex B01–B05 | 单种子，参考 25 μs，标准夹具及两质量，保存每 2 ms 的反力、位姿、膜点和几何诊断 | 静置包覆、拉伸通过。本轮空袋静止、原盒台面 U 折、原速边界提拉触发门槛。两项修正复测通过 |
| VBD CUDA 对照 | 同几何，50 μs，`flock /tmp/claude-gpu.lock`。重复记录运动学反力读回修正，并用未拟合的 Y 轴验证面内响应 | 空袋速度通过。两质量内芯相交，卸载局部伸长越限。Y 稳态标定误差降低约 77.5% |
| B06 诊断 | 同一轨迹的首轮三点及对准内芯的单点慢速复测 | 首轮上限与几何门槛存在失效。Flex 慢程 4.005 N 数值有效，实际膜—探头和膜—内芯都产力 |
| MJWarp CUDA 最小功能 | `pe_physics_capabilities.py`，原模型另加 `--parcel-only`，GPU 持锁 | 1/2/3D、所测弹性与边约束、接触、摩擦及两路图捕获均有设备实测。自接触有效行与 CPU 有差异 |
| 交付校验 | 汇总 CSV/JSON 行数、配对输入逐数组相等、FFprobe、故障门控启动 | 27 行完全对应，9 个配对的参考几何、初态、面和质量逐项一致。两支 H.264 各 61 帧。正式探针入口按预期拒绝使用未通过的基础证据 |
| 正式 B07–B12、统计重复 | 根据原物理阶段门槛审查 | 未进入正式执行 |

初次 VBD 入口排除了几何归档流未复位、配置默认项未展开、过重的体素预处理和 Warp 内核类型错误。旧日志保留在 [日期目录](../test/runs/2026-11-25/bag/)。早期 VBD 刚体邻接读取遗漏运动学接触，已修复并复测。B06 首轮用停车位误作位移零点，汇总已按命令起点纠正。越限力和旧数据保留。

## 复现入口

在仓库根目录运行。用新的输出路径保存不同轮次。

```sh
PE_PYTHON=/home/ubuntu/workspace/gzl/randomization/.venv-physical-parcel/bin/python
PE_DEPENDENCIES=.venv-physical-parcel/benchmark-deps

uv pip install --python "$PE_PYTHON" --target "$PE_DEPENDENCIES" --no-deps newton==1.6.1

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 "$PE_PYTHON" -B \
  test/bag_model/pe_physics_protocol.py /tmp/pe-flex-foundation

PYTHONPATH="$PE_DEPENDENCIES" OPENBLAS_NUM_THREADS=1 flock /tmp/claude-gpu.lock "$PE_PYTHON" -B \
  test/bag_model/pe_physics_protocol.py /tmp/pe-vbd-foundation --backend vbd

"$PE_PYTHON" -B test/bag_model/pe_physics_protocol.py /tmp/pe-flex-fold-plane \
  --cases B04-U --plane-table

"$PE_PYTHON" -B test/bag_model/pe_physics_protocol.py /tmp/pe-flex-lift-slow \
  --cases B05-D --motion-time-scale 1.8

"$PE_PYTHON" -B test/bag_model/pe_physics_protocol.py /tmp/pe-flex-press-gentle \
  --phase probe-diagnostic --cases B06-S2-P1 --gentle-probe
```

合并规则及每一来源记录见 [manifest.json](../test/runs/2026-11-25/bag/pe-physics-benchmark-r1/manifest.json)。原始 NPZ 包含采样状态，设备接触不是逐子步完整时序。两秒量级的 pilot 和两路 Graph 功能检查无法支持有效吞吐量估算。后续接触及扫网格需跨过同一套物理门槛。

[物理有效性](../test/runs/2026-11-25/bag/pe-physics-benchmark-r1/physics_validity_report.md) · [后端决策](../test/runs/2026-11-25/bag/pe-physics-benchmark-r1/solver_comparison_report.md) · [正式测试原始数据](../test/runs/2026-11-25/bag/pe-physics-benchmark-r1/raw_trajectories.npz)

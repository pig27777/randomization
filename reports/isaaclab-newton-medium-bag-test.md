# Isaac Lab Newton 中精 PE 整袋验证检查

2026-11-25（台北时间），机器 `ubun`。下表为最初的指定路径预检，获准改用本机隔离环境后的实测见下一节。

| 检查 | 方法与结果 |
| --- | --- |
| 指定启动器与源码路径 | 用 `ls -ld` 和 `test -e` 核对 `/home/ubuntu/workspace/czx/system0/sim`、`python.sh` 和 `IsaacLab/source`，均不存在。 |
| GPU | `nvidia-smi` 确认 RTX 5090、32,607 MiB、驱动 580.178.04。未启动仿真或 CUDA 基准。 |
| 本机候选安装 | 只读检查 `/home/ubuntu/Dev/Envs/isaaclab`。Isaac Lab tag `v3.0.0`，commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`，Isaac Sim 发行包为 6.2.0.0。候选安装尚未用于本轮运行。 |
| 耦合入口 | 静态阅读 `isaaclab_contrib/coupling/coupler_cfg.py` 和 `coupler.py`。有原生反力反馈配置和零反馈对照入口。该版本不支持通过通用 Newton 接触传感器读取耦合后的接触，因此独立诊断须取各求解器缓冲。 |
| 机器人与网格 | 读取 `configs/route_b_yam_scene.json` 和资产文件。Yam_B 封装资产存在，指垫面 14.4 × 16.4 mm。CPU NumPy 去重三角形边并计算 `bag_sim/geometry/pe.npz` 的边长及参考面积。549 顶点、1,064 三角形，边长中位/最大为 25.00/35.36 mm，参考面积 0.3325 m²。 |
| 差异检查 | `git diff --check` 通过。 |

预检时两项任务门槛均待动力学实测。[试验计划](../test/isaaclab/REPORT.md#2026-11-25--yam_b-与中精-pe-整袋的-newton-双向耦合)

## 本机路线实测

最近检查于 2026-11-25 17:25（台北时间）。RTX 5090、32,607 MiB、驱动 580.178.04。本机 Isaac Sim 6.2.0.0、Isaac Lab v3.0.0，Newton 1.6.1、Warp 1.17.0、MuJoCo Warp 3.12.0。GPU 脚本串行持 `/tmp/claude-gpu.lock`，OMP/OpenBLAS 各一线程。

| 检查 | 执行与结果 |
| --- | --- |
| Isaac Sim headless | `OMNI_KIT_ACCEPT_EULA=yes flock /tmp/claude-gpu.lock ./isaaclab/run.sh isaaclab/check_isaacsim.py --output <run>/isaacsim-smoke.json`。11.27 s 启动，stage 与十次更新通过。 |
| 中精入口 | 持锁运行 `benchmark_newton_yam_bag.py --headless --ticks 2`。原碰撞网格缺距离场，保留 USD 碰撞网格并增加 2.5 mm 距离场后可运行。1,010 顶点、2,016 面，最大边长 14.142 mm。 |
| 接触归属和准备态 | 移除 VBD 侧重复的刚体对后缓冲溢出消失。低置试件 74 mm 后 80 ms 拉伸峰值 1.00866。只允许粒子碰撞的旧承台未被代理选中，更换为两端可见的固定盒。 |
| 两质量及零反馈 | 运行目录的 `run.sh --ticks 62 --dt .0005 --output ...`，分别用默认 0.20 kg、`--mass .35` 和 `--feedback 0`。正反馈 124 个记录帧完整，峰值 1.01639／1.01890；零反馈首窗 1.06725。逐步汇总证明内芯与臂体直接加载。 |
| 原生约束形状核验 | 同入口各 5 步，默认和 `--lower-m .082`。两次均从约束力而非候选数确认外壳—内芯旁路，低置后仍实际加载。 |
| 吞吐与显存 | 正反馈单路 62 步除以各自的 79.38／79.52 s，均约 0.78 策略步/墙钟秒，含诊断读回。NVIDIA 进程显存 0.5 s 间隔抽样，共 345 条，峰值 712 MiB。 |
| 数据和静态复核 | 严格解析 JSON，检查两质量动作逐帧相同、62／62／1 时间对齐、拉伸／旁路统计和显存采样。Python AST、`bash -n`、`git diff --check` 均通过。 |

受控质量对照尚不能证明无旁路的关节辨识。偏心／摩擦内芯辨识、网格分辨率及并发吞吐扫档依物理门槛停在单路。原生距离场构建失败、Warp 诊断编译失败和修正后的成功数据均保留。[原始记录目录](../test/runs/2026-11-25/isaaclab/newton-yam-medium-bag-r1/) · [主题报告](../test/isaaclab/REPORT.md)

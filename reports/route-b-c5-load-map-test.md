# Plan B C5 分指顶压载荷测试

2026-11-23，北京时间。主机 `ubun`，RTX 5090，Isaac Sim 6.2 / Isaac Lab 3.0。只读查因在 19:28 完成。正式 GPU 运行使用 `flock /tmp/claude-gpu.lock`，无渲染。

## 只读查因

同动作复放 0.914 mm 的历史 C5 和 `2ef4a394` 冻结参考中种子 347 的第 425 条。从 80 到 2 mm 的 159 个双指闭合子步里，左／右实测分指顶压超过 0.25 N 的子步为 158／25。实测顶压超过 0.25 N 而代理法向力低于 0.25 N 的子步为 56／24。2 mm 以上积累 4.507／0.914 mm，最后 2 mm 增加 0.092／0 mm。冻参考在 87.673 mm 指距捕获，每指实测 Fz 约 5.583 N，脚本直接施加的 Fn 为 14.799 N/指。

逐 2 ms 数据记录分指 Fz、软层压陷、代理 Fn、闭合速度、聚拢增量与累计值。[PhysX 历史轨迹](../test/runs/2026-11-23/isaaclab/route-b-c5-load-map-r1/c5-live-closure.csv) · [冻结参考](../test/runs/2026-11-23/isaaclab/route-b-c5-load-map-r1/c5-frozen-closure.csv)

## 定向修改和回归

在 `DualArmParcelSim` 耦合层为 PhysX 环境开启可选分指映射。当软包存在真实压陷、双轨持续内闭且该指实际顶压为正时，代理法向容量取旧容量与该指 Fz 的较大值。硬书、空载、侧抓、张开和止挡状态不激活该映射。顶压已经通过原腕部/横向载荷接口回传。额外滑动摩擦项未引入。

| 同动作 A/B | 修复前 | 修复后 |
| --- | ---: | ---: |
| C5 早闭双侧聚拢 | 0.914 mm | 2.414 mm |
| C5 晚闭双侧聚拢 | 0.308 mm | 4.466 mm |
| C5 晚闭双指捕获 | 0/300 帧 | 2/300 帧 |
| C5 晚闭同时捕获与完全卸载离台 | 0 帧 | 0 帧 |
| C5 晚闭 2 ms 顶压峰值 / 25 Hz 闭合帧处于 8–12 N | 21.605 N / 0 帧 | 11.848 N / 0 帧 |
| C4，顶压峰值 12.356 N | 0/300 捕获，0 mm 离台 | 0/300 捕获，0 mm 离台 |

空载 0/4/30/35 N 的代理力与捕获均精确为零。双通道功 −0.191999963 mJ，静态 6 N/指去皮电流 +5.981641 N。CPU 分指检查覆盖单指有载、双指有载、硬书、张开、止挡、空载及侧抓，确认同一无捕获初态下力、力矩和包裹位移逐元素不变。两个冻结核心文件及 `yam_load_port.py` 的路径级 `git diff --exit-code` 通过。

原生环境运行 `check_route_b_closed_loop.py --headless --empty-clamp` 和 `check_route_b_wrench_power.py --headless`。三条固定对照的命令使用 `compare.py --headless --selected matched-candidates.json --output ...`，通过 `ROUTE_B_TOP_MAP=0/1` 切换。C5 完整门禁未通过。[A/B、空载及端口汇总](../test/runs/2026-11-23/isaaclab/route-b-c5-load-map-r1/summary.json)

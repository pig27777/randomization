# 2026-11-20 运行索引

袋与内容物使用同一 MuJoCo 4.0.1 整袋模型及共享 YAM 平台。CPU 与 RTX 5090 运行均在 ubun。主题结果和验收门槛见 [Stage 1 实验报告](../../bag_model/RSS2027_STAGE1_REPORT.md) 与 [任务检查](../../../reports/rss2027-stage1-test.md)。

| 运行 | 对象与问题 | 命令 | 状态 |
| --- | --- | --- | --- |
| [PE 收拢条件](bag/rss2027-stage1-pe-gather-conditions-r1/) | 膜片局部 2.5 mm 网格，改变余料或法向力 | 持锁执行本轮 `run_cases.sh` | 两组均有合格压力窗，斜率 0.713、0.709 m/m。 |
| [同态接触比对](bag/rss2027-stage1-payload-contact-parity-r1/) · [原生法向速度复核](bag/rss2027-stage1-payload-contact-parity-r2/) | 完整 YAM 侧抓的 4 个存档姿态，比较 CPU 和 Warp 角点约束 | 持锁运行 `rss2027_payload_contact_parity.py`，第二轮附 `--film-vertical-velocity .01` | 闭合末帧与停止帧膜面 191 法向差 89.18°、87.73°，Warp 原始读数与约束投影吻合。 |
| [0.18 m/s CPU](bag/rss2027-stage1-pickup-baseline-cpu-r4/) · [同配置 GPU](bag/rss2027-stage1-pickup-baseline-warp-r4/) | 142 节点整袋侧抓，刚性内容物，200 μs | 持锁执行各自 `run.sh` | CPU 通过 4.7704 s，GPU 在 2.7804 s 露角停止，墙钟 477.94 s。 |
| [整袋夹口材料](bag/rss2027-stage1-whole-bag-aperture-r1/) | 原空袋顶抓与同参 PE 侧抓的夹口材料面积 | `rss2027_yam_capture_audit.py <trial> <output>`，各回放一次 | 空袋 105 帧均为零；侧抓起抬 3.98 mm，悬停 8.74–10.19 mm。 |
| [整袋材料对照](bag/rss2027-stage1-whole-bag-materials-r1/) | 同一 142 节点封口袋和刚性内容物，4 类工程预设 | 持锁执行 `run.sh smoke`，入口通过后执行 `run.sh <material>` | PE、气泡膜通过入口。PE 成功夹入后露角，气泡膜未建立正拉力。牛皮纸和纸板入口穿透超 1 mm。 |
| [偏置 15 mm CPU](bag/rss2027-stage1-pickup-baseline-cpu-r5/) · [同配置 GPU](bag/rss2027-stage1-pickup-baseline-warp-r5/) | 原慢速算力工况移动内容物横向位置 | 持锁执行各自 `run.sh` | CPU 通过 4.7804 s。同配置 GPU 在 2.8805 s 露出同一角点，相交膜面由 191 变为 204。 |
| [纸材 0.5 ms 接触复测](bag/rss2027-stage1-whole-bag-materials-r2/) | 同牛皮纸与纸板，减小接触时间常数 | 持锁执行材料首轮 `run.sh smoke <本轮目录>` 及 `run.sh <material> <本轮目录>` | 入口峰值 0.855/0.748 mm，均通过。完整回合已启动。 |
| [约 30% 填充率 CPU](bag/rss2027-stage1-pickup-baseline-cpu-r6/) · [同配置 GPU](bag/rss2027-stage1-pickup-baseline-warp-r6/) | 已存紧凑包装尺寸和偏置，复用 YAM 短动作 | 持锁执行各自 `run.sh` | CPU 通过 4.7626 s，墙钟 91.35 s。条件 GPU 排队。 |
| [PE 膜量与载荷区间](bag/rss2027-stage1-pe-aperture-width-r1/) | 2.5/1.25 mm PE 预褶曲线的整指宽膜量、起滑前刚度和超载 | `measure.py <牵引数据目录> <本轮 audit.json>` | 捕获膜量差约 43%，候选刚度差 56.7%，连续双侧超载均仅 9 ms。 |
| [低余料和压力](bag/rss2027-stage1-pe-low-slack-r1/) · [补做拉拔](bag/rss2027-stage1-pe-capture-boundary-r1/) | 同参 PE 平坦侧余料的第三档压力、余料及空夹对照 | 持锁执行各轮 `run_cases.sh` | 0/2 mm 余料及 0.3 N 组三组均无正拉力。补做拉拔排队。 |

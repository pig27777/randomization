# 2026-11-20 运行索引

袋与内容物使用同一 MuJoCo 4.0.1 整袋模型及共享 YAM 平台。CPU 与 RTX 5090 运行均在 ubun。主题结果和验收门槛见 [Stage 1 实验报告](../../bag_model/RSS2027_STAGE1_REPORT.md) 与 [任务检查](../../../reports/rss2027-stage1-test.md)。

| 运行 | 对象与问题 | 命令 | 状态 |
| --- | --- | --- | --- |
| [PE 收拢条件](bag/rss2027-stage1-pe-gather-conditions-r1/) | 膜片局部 2.5 mm 网格，改变余料或法向力 | 持锁执行本轮 `run_cases.sh` | 两组均有合格压力窗，斜率 0.713、0.709 m/m。 |
| [同态接触比对](bag/rss2027-stage1-payload-contact-parity-r1/) · [原生法向速度复核](bag/rss2027-stage1-payload-contact-parity-r2/) | 完整 YAM 侧抓的 4 个存档姿态，比较 CPU 和 Warp 角点约束 | 持锁运行 `rss2027_payload_contact_parity.py`，第二轮附 `--film-vertical-velocity .01` | 闭合末帧与停止帧膜面 191 法向差 89.18°、87.73°，Warp 原始读数与约束投影吻合。 |
| [0.18 m/s CPU](bag/rss2027-stage1-pickup-baseline-cpu-r4/) · [同配置 GPU](bag/rss2027-stage1-pickup-baseline-warp-r4/) | 142 节点整袋侧抓，刚性内容物，200 μs | 持锁执行各自 `run.sh` | CPU 通过 4.7704 s，GPU 在 2.7804 s 露角停止，墙钟 477.94 s。 |
| [整袋夹口材料](bag/rss2027-stage1-whole-bag-aperture-r1/) | 原空袋顶抓与同参 PE 侧抓的夹口材料面积 | `rss2027_yam_capture_audit.py <trial> <output>`，各回放一次 | 空袋 105 帧均为零；侧抓起抬 3.98 mm，悬停 8.74–10.19 mm。 |
| [整袋材料对照](bag/rss2027-stage1-whole-bag-materials-r1/) | 同一 142 节点封口袋和刚性内容物，4 类工程预设 | 持锁执行 `run.sh smoke`，入口通过后执行 `run.sh <material>` | PE、气泡膜通过入口，牛皮纸和纸板穿透超 1 mm。PE 成功夹入，3.0202 s 提拉期间露角。气泡膜完整动作排队。 |
| [偏置 15 mm CPU](bag/rss2027-stage1-pickup-baseline-cpu-r5/) · [条件 GPU](bag/rss2027-stage1-pickup-baseline-warp-r5/) | 原慢速算力工况移动内容物横向位置 | 持锁执行各自 `run.sh` | CPU 通过 4.7804 s，悬停最小闭合轴反力 2.53 N。同配置 GPU 已启动。 |
| [纸材早期接触对照](bag/rss2027-stage1-whole-bag-materials-r2/) | 同牛皮纸与纸板的 0.5 ms 接触时间常数入口 | 持锁执行材料首轮 `run.sh smoke <本轮目录>` | 排队。 |

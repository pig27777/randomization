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
| [纸材 0.5 ms 接触复测](bag/rss2027-stage1-whole-bag-materials-r2/) · [单独收紧纸板膜接触](bag/rss2027-stage1-whole-bag-materials-r3/) | 牛皮纸与纸板接触时间常数，以及单独减小纸板膜参数 | 持锁执行材料首轮 `run.sh smoke <本轮目录>` 及 `run.sh <material> <本轮目录>` | 两种纸材离台但膜—膜穿透超限。纸板膜 0.25 ms 完整对照峰值降至 1.242 mm，仍超限。 |
| [约 30% 填充率 CPU](bag/rss2027-stage1-pickup-baseline-cpu-r6/) · [同配置 GPU](bag/rss2027-stage1-pickup-baseline-warp-r6/) | 已存紧凑包装尺寸和偏置，复用 YAM 短动作 | 持锁执行各自 `run.sh` | CPU 通过 4.7626 s，墙钟 91.35 s。GPU 3.1805 s 露出角点 0，墙钟 560.24 s。 |
| [PE 膜量与载荷区间](bag/rss2027-stage1-pe-aperture-width-r1/) | 2.5/1.25 mm PE 预褶曲线的整指宽膜量、起滑前刚度和超载 | `measure.py <牵引数据目录> <本轮 audit.json>` | 捕获膜量差约 43%，候选刚度差 56.7%，连续双侧超载均仅 9 ms。 |
| [低余料和压力](bag/rss2027-stage1-pe-low-slack-r1/) · [补做拉拔](bag/rss2027-stage1-pe-capture-boundary-r1/) | 同参 PE 平坦侧余料的第三档压力、余料及空夹对照 | 持锁执行各轮 `run_cases.sh` | 0/2 mm 余料及 0.3 N 组三组均无正拉力，4 mm／0.5 N 和 8 mm／0.8 N 均有持续承载和脱开。 |
| [临界捕获细网格](bag/rss2027-stage1-pe-capture-fine-r1/) · [慢速拉拔](bag/rss2027-stage1-pe-slow-pull-r1/) | 4 mm 余料的 1.25 mm 网格入口、既有预褶的半速拉拔 | 持锁执行各轮 `run.sh` | 慢速候选力 0.063 N，脱开位移约 13.6 mm，正超载 9 ms。4 mm 细网格无夹入。 |
| [8 mm 细网格捕获](bag/rss2027-stage1-pe-nominal-capture-fine-r1/) · [保留 1.10 mm 开口](bag/rss2027-stage1-pe-residual-gap-r1/) | PE 平坦余料，1.25 mm 网格，0.5 N，比较闭合过窄的影响 | 持锁执行各轮 `run.sh` | 原 0.40 mm 开口在提拉前膜量降至零，前 0.15 s 无正拉力。1.10 mm 排队。 |
| [三种内容物布局的夹口材料](bag/rss2027-stage1-content-aperture-r1/) | 居中、横移 15 mm、紧凑约 30% 的 CPU 整袋夹口复核 | `rss2027_yam_capture_audit.py <trial> <output>`，各回放一次 | 起抬等效膜长 8.066/8.363/9.394 mm，紧凑布局悬停仍缺同帧双侧内触点。 |
| [同态凸碰撞容差](bag/rss2027-stage1-ccd-parity-r1/) | 原 CPU 和 Warp 接触法向差异的容差敏感性 | 持锁执行本轮 `run_cases.sh`，再用 `check_normal_depth.py` | 0.1/1 μm 容差下，关键法向仍相差约 89°，设备原生法向与深度的投影不一致。 |
| [隔离 Warp 法向修复](bag/rss2027-stage1-warp-film-normal-r1/) | 保留产生深度的 GJK/EPA 法向，原同态对照与完整在线居中侧抓 | 用 `create_overlay.py` 生成隔离版本，持锁执行 `run_parity.sh` 和条件 `run_trial.sh`，再执行 `analyze_trial.py` | 整袋完成 4.76965 s，墙钟 539.12 s，显存峰值 966 MiB，穿透 0.469 mm。[褶皱录像](bag/rss2027-stage1-warp-film-normal-r1/yam-warp-fold.mp4)。 |
| [约 30% 布局的隔离版](bag/rss2027-stage1-warp-film-normal-30pct-r1/) | 同修正的另一内容物布局对照 | 持锁执行本轮 `run.sh` | 4.75966 s 完整通过，墙钟 579.86 s，穿透 0.813 mm，停止态保持包覆。 |
| [纸板上缘进给](bag/rss2027-stage1-whole-bag-materials-r4/) | 只将纸板接近目标升高 16 mm | 持锁执行材料首轮 `run.sh smoke <本轮目录>` 及 `run.sh paperboard <本轮目录>` | 完成 6.0969 s，悬停离台，穿透峰值 0.685 mm，最小单指力 6.98 N。 |
| [牛皮纸与气泡膜上缘进给](bag/rss2027-stage1-whole-bag-upper-r1/) | 原各自基准参数下，对照同一 16 mm 高度变化 | 持锁执行材料首轮 `run.sh smoke <本轮目录>` 和各自完整动作 | 排队。 |
| [PE 受控拉力](bag/rss2027-stage1-pe-force-pull-r1/) · [1 N 夹力对照](bag/rss2027-stage1-pe-force-pull-r2/) | 同参 PE 0.30 N 拉力反馈，对比 6.5 N 与 1.0 N 夹力 | 持锁执行首轮 `run.sh`。后者附 <本轮目录> 和 `1.0` | 6.5 N 有 0.1229 s 拉力达标窗，同期无整体滑移。1 N 组排队。 |
| [闭环场景的批量设备短窗](bag/rss2027-stage1-yam-batch-reference-r1/) | 通过轨迹的三种姿态，1/16/64 世界 | 持锁执行本轮 `run.sh` | 排队。 |

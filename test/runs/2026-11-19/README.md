# 2026-11-19 运行索引

| 运行 | 对象与问题 | 机器和命令 | 状态与证据 |
| --- | --- | --- | --- |
| [rss2027-stage1-r1](bag/rss2027-stage1-r1/) | PE 60 mm 试片、随机整袋几何、142 节点封口袋和 200 节点 YAM 入口。检查原生接触、聚拢、预褶承载，并回放整袋折痕 | ubun。正式动力学以 `flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B` 启动，用 `test/bag_model/rub_contact_e1.py`、`rss2027_micropatch.py`、`rss2027_geometry.py`。录像复用 `render_rss2027_fold_video.py` | E1、随机几何与预褶对照通过，四材料包络仍待验证。见[主题报告](../../bag_model/RSS2027_STAGE1_REPORT.md)和[测试报告](../../../reports/rss2027-stage1-test.md)。 |
| [rss2027-stage1-yam-gpu-probe-r1](bag/rss2027-stage1-yam-gpu-probe-r1/) | 完整 YAM 与 200 节点双膜片包裹，检查静置和合指的 GPU 短步进 | ubun，RTX 5090。 `flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B test/runs/2026-11-19/bag/rss2027-stage1-yam-gpu-probe-r1/probe.py` | 入口通过，接触力存在差异。见[测试报告](../../../reports/rss2027-stage1-yam-gpu-probe-test.md)和[逐步数据](bag/rss2027-stage1-yam-gpu-probe-r1/results.json)。 |
| [rss2027-stage1-yam-contact-parity-r2](bag/rss2027-stage1-yam-contact-parity-r2/) | 同一 200 节点 YAM 合指状态，按部件和触点对齐 CPU/GPU 接触 | ubun，RTX 5090。 `flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B test/runs/2026-11-19/bag/rss2027-stage1-yam-contact-parity-r2/compare.py` | 定位两处左指浅接触解释首步竖向差的约 92%。[原始结果](bag/rss2027-stage1-yam-contact-parity-r2/results.json)。 |
| [rss2027-stage1-pe-press-r1](bag/rss2027-stage1-pe-press-r1/) | 同一 PE 膜片的实体接触，两档指垫接触时间常数 | ubun。 `flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B test/bag_model/rss2027_micropatch.py --dt .00005 --pad-contact-time <.001 或 .0001> --output <本轮工况目录>` | 下压力目标未达，峰值 4.86 / 12.13 mN。[定量对照](bag/rss2027-stage1-pe-press-r1/press-comparison.json)。 |
| [rss2027-stage1-pe-normal-r1](bag/rss2027-stage1-pe-normal-r1/) | 原 PE 局部模型法向力读数与恒位移承载复核 | ubun。持锁运行记录目录中的 `normal_probe.py`、`normal_dwell.py` | 两条反力读数吻合；完成整袋方法切换前的诊断。 |
| [rss2027-stage1-empty-yam-r1](bag/rss2027-stage1-empty-yam-r1/) | 空袋 YAM 初始静置与薄壳自接触诊断 | ubun。持锁运行 `record_physical_parcel.py`、记录目录中的 `settle_probe.py` 和 `collision_probe.py` | 默认静置超限；原生 CCD 1 nm 复核 100/200 μs 静置通过。[主题报告](../../bag_model/RSS2027_STAGE1_REPORT.md)。 |
| [rss2027-stage1-empty-yam-r2](bag/rss2027-stage1-empty-yam-r2/) | 仓库 142 节点 PE 空袋，测试力控及合指提拉 | ubun。 `flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B test/bag_model/record_physical_parcel.py --config test/runs/2026-11-19/bag/rss2027-stage1-empty-yam-r2/config.json --output <新目录>` | 法向力门槛通过，提拉失联。见[完整尝试](bag/rss2027-stage1-empty-yam-r2/recording/trial/result.json)。 |
| [rss2027-stage1-empty-yam-r3](bag/rss2027-stage1-empty-yam-r3/) | 同一空袋与法向载荷目标，验证提前减速 | ubun。 `flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B test/bag_model/record_physical_parcel.py --config test/runs/2026-11-19/bag/rss2027-stage1-empty-yam-r3/config.json --output <新目录>` | 法向力通过，峰值 0.646 N。见[复测结果](bag/rss2027-stage1-empty-yam-r3/recording/trial/result.json)和[测试报告](../../../reports/rss2027-stage1-test.md)。 |
| [rss2027-stage1-empty-yam-r4](bag/rss2027-stage1-empty-yam-r4/) | 限速版完整尝试与夹入/滑出判别 | ubun CPU。持锁用 `record_physical_parcel.py --config <本轮 config.json> --output <新目录>`，再离线运行 `rss2027_yam_capture_audit.py` | 下压力通过，夹入失败。[触点与原始力对齐](bag/rss2027-stage1-empty-yam-r4/capture/capture-summary.json)。 |
| [rss2027-stage1-yam-warp-entry-r1](bag/rss2027-stage1-yam-warp-entry-r1/) · [在线下压](bag/rss2027-stage1-yam-warp-press-r1/) | 完整 YAM 与同一 PE 空袋，在线 GPU 反力反馈 | ubun，RTX 5090。持锁用 `record_physical_parcel.py --engine warp --config <各轮 config.json> --output <新目录>` | 20 ms 入口通过；在线下压到 9.859 s 时达到墙钟预算，压力带时长未达，采样显存最高 964 MiB。[加载结果](bag/rss2027-stage1-yam-warp-press-r1/recording/summary.json)。 |
| [rss2027-stage1-pe-matched-r1](bag/rss2027-stage1-pe-matched-r1/) | 同整袋材料的 PE 平坦余料、预褶、零摩擦及半网格/半步长 | ubun CPU。持锁执行本轮 `run_cases.sh` | 两组有夹持与脱开，零摩擦无正拉力；夹口网格减半后拉力峰值差约 216%。[主题报告](../../bag_model/RSS2027_STAGE1_REPORT.md)。 |
| [rss2027-stage1-empty-yam-close-hold-r1](bag/rss2027-stage1-empty-yam-close-hold-r1/) | PE 空袋，闭合限位保持 0.30 s，重跑完整抓取尝试 | ubun CPU。持锁用 `record_physical_parcel.py --config <本轮 config.json> --output <新目录>`，后处理同 R4 | 起抬夹口 0.581 mm，仍无夹入。[回放摘要](bag/rss2027-stage1-empty-yam-close-hold-r1/capture/capture-summary.json)。 |
| [rss2027-stage1-pe-matched-r2](bag/rss2027-stage1-pe-matched-r2/) | 同参 PE，三角面积质量分摊后五项对照复测 | ubun CPU。持锁执行本轮 `run_cases.sh` | 夹口加密的拉力峰值差降至 8.5%，起滑力候选仍差 32.7%。[主题报告](../../bag_model/RSS2027_STAGE1_REPORT.md)。 |
| [rss2027-stage1-content-pair-r1](bag/rss2027-stage1-content-pair-r1/) | 同填充率同袋下的两组不同内容物尺寸与位置，完整 YAM 顶抓 | ubun CPU。持锁执行本轮 `run_cases.sh` | 压力达标时间 10.088/8.465 s，均未夹入。[报告](../../bag_model/RSS2027_STAGE1_REPORT.md#2026-11-19-同填充率的整袋动力学对照)。 |
| [rss2027-stage1-pe-clamp-force-r1](bag/rss2027-stage1-pe-clamp-force-r1/) | 同参 PE，粗细网格的预褶施加 6.5 N 夹持力 | ubun CPU。持锁执行本轮 `run_cases.sh` | 夹持力均达标，夹口材料长度相差 3.43 mm，起滑候选未收敛。 |
| [rss2027-stage1-side-grasp-r1](bag/rss2027-stage1-side-grasp-r1/) | 侧抓整袋正对照，复核 1 nm CCD 容差下的原生夹持 | ubun CPU。持锁执行本轮 `run.sh` | 运行中。[计划](../../bag_model/RSS2027_STAGE1_REPORT.md#2026-11-19-既有整袋侧抓复测)。 |
| [rss2027-stage1-pe-mesh-axes-r1](bag/rss2027-stage1-pe-mesh-axes-r1/) | 同 6.5 N 预褶，分别沿闭合轴或横向加密 | ubun CPU。持锁执行本轮 `run_cases.sh` | 排队中。[计划](../../bag_model/RSS2027_STAGE1_REPORT.md#单轴加密诊断计划)。 |

## folded-clothes-6mm-r1：近满面薄层包裹

- 对象/版本：physical_parcel，142节点/280面，袋350×250 mm、三段内容物330×230×6 mm；ubun/CPU MuJoCo4.0.1。初始8 mm失败记录保留在folded-clothes-flat-r1。
- 问题/标准：近似叠衣外形并录制真实抓取；准备/几何按原1 mm预算，抓起≥50 mm且支撑<0.05 N，完整结果按共享协议。每抓法一次。
- 状态：17项回归、0.5 s准备和视频检查通过；顶抓失败；侧抓提起175.97 mm但落位失败且穿透1.223 mm超标，物理验收未过。
- [配置/命令/原始数据](bag/folded-clothes-6mm-r1/) · [主题报告](../../bag_model/PHYSICAL_PARCEL_REPORT.md) · [任务检查](../../../reports/folded-clothes-grasp-test.md) · [连续视频](bag/folded-clothes-6mm-r1/folded-clothes-grasp.mp4)。

## flat-bag-surface-rub-r1：完整袋面受控搓动

- 对象：上述6 mm薄层包裹、原142节点网格，ubun/CPU MuJoCo4.0.1，用户要求不占GPU。
- 配置/标准：每指0.5/2 N，0/5/10 mm、μ=.8/0；7组有效对照各一次，实际压力±20%、原1 mm几何/数值预算，计划与加载修复见主题报告。
- 结果：接触/反力/载荷通过，无明显新褶皱；节点运动约0.14–0.22 mm，最大局部凸起0.178 mm。压力标定失败保留，细网格/真实材料结论未验证。
- [原始数据/配置/命令](bag/flat-bag-surface-rub-r1/) · [主题报告](../../bag_model/PHYSICAL_PARCEL_REPORT.md) · [任务检查](../../../reports/surface-rub-test.md) · [2 N视频](bag/flat-bag-surface-rub-r1/surface-rub-2N.mp4)。

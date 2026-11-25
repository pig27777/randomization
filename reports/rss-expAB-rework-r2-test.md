# RSS Exp A/B r2 任务检查

主机 ubun，北京时间 2026-11-26。软件为指定虚拟环境的 Python 3.14.7、MuJoCo 4.0.1 与 CPU PyTorch。实验配置、门禁和数据见 [本轮实验报告](../test/runs/2026-11-26/rss-expAB-rework-r2/rework_report.md)。

| 范围 | 方法 | 结果 |
| --- | --- | --- |
| PE Flex 回归 | `.venv-physical-parcel/bin/python -B -m unittest discover -s test/bag_model -p test_physical_parcel.py -v` | 20/20 通过。[日志](../test/runs/2026-11-26/rss-expAB-rework-r2/parcel-regression.log) |
| Exp A | `rss_exp_ab_rework.py --part A` 读取 r1 轨迹与冻结 TCN 快照 | 135/135 原轨迹有效，轨迹与指标源哈希一致。45 组分在五折。四档、两解码器、两个任务及 348 个配对检验均已读回。 |
| Exp B 对角线复现 | 首轮三个对角格各重跑 1 次，5 ms 审计 | 均复现先失载并随后越过膜伸长门禁。[逐帧诊断](../test/runs/2026-11-26/rss-expAB-rework-r2/diagnostic-r1/) |
| Exp B 正式 | 22 个 CPU 进程、50 µs CPU Flex，项目锁。分析脚本比对原始 216 记录 | 独立种子 108+108，所有 27 格在两矩阵均有分母。三个对角格全数成功。全局 34/216 无效，未通过 5% 门禁。350 g 条件未触发。[正式日志](../test/runs/2026-11-26/rss-expAB-rework-r2/formal-workflow.log) |
| 图件 | PDF 目视、`pdffonts`、`pdfimages`、PNG DPI 读回 | 两 PDF 均单页，所有字体为嵌入 TrueType，均无位图。PNG 为 2194 × 1459 和 2194 × 905，均 300 DPI。标签、右栏下标和数值列完整。 |
| 冻结路径 | `bash test/runs/2026-11-26/rss-expAB-rework-r2/check-frozen.sh` | 四个路径级 `git diff --exit-code` 检查退出 0。启动时跟踪及未跟踪的 410 个文件哈希一致。[核验日志](../test/runs/2026-11-26/rss-expAB-rework-r2/frozen-check.log) |
| 数据与程序读回 | 解析两份结果 JSON、逐回合比较、Python 源码编译及 `git diff --check` | 通过，[交付检查](../test/runs/2026-11-26/rss-expAB-rework-r2/delivery-check.log)。条件估计保留对应的全结果界。 |

初次正式批因全局门禁已不可能达到而中止，保留 25 个已写回合。随后用独立种子完整执行正式矩阵。序列化失败已修复，低速与改退爪诊断仍复现物理失效，记录均留在实验目录。正式试验使用原 2 mm 内缩盒、直接接触与 1.05 边长比判据。两策略的独立评测差分别为 33.3 和 22.2 个百分点，下界选择的 95% 区间仍含 0。

最近核查为 2026-11-26 04:56 CST。最后用原种子逐帧复放每个过压对角格里原生峰值最高的回合。峰值 29.83、29.82 和 18.16 N，分别发生在下放、下放及松爪。对应稳持峰值为 8.74、8.57 和 8.32 N。[分段复放日志](../test/runs/2026-11-26/rss-expAB-rework-r2/diagnostic-peak-reactions/run.log)

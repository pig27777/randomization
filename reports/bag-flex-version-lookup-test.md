# 袋子与 flex 薄膜历史版本查找检查

最近检查时间为 2026-11-15 16:02（Asia/Taipei），机器为 ubun。

## 范围与方法

按用户要求查找薄膜袋装小球、牛皮纸袋装硬盒夹取、灰色 PE 袋装物或下压三类历史版本，同时记录现存 YAM 袋装书本和衣物录像。

使用 `rg --files --hidden --no-ignore` 检索本仓库和相邻 `real2sim` 资产目录，核对 README、主题报告、模型入口及录像来源。用 Python `Path.is_file()` 和 `stat()` 核对交付的 13 个文件，用 `hashlib.sha256()` 对照 flex 小球录像的既有检查记录。

## 查找结果

| 版本 | 模型与记录 | 现存视频或动画 |
| --- | --- | --- |
| 2026-11-09，2D flex 薄膜袋与自由颗粒 | [入口](../test/flex_granular/benchmark.py)、[实验报告](../test/flex_granular/REPORT.md)、[保存的 XML](../test/flex_granular/results/20261109_ubun_gpu5s/gpu64_5s.xml) | [5 秒 GPU 录像](../video/flex_granular_gpu5s.mp4)，64 球、98 膜顶点，半透明和线框并排 |
| 2026-11-05 至 11-07，牛皮纸松装 flex 袋 | [模型](../bag_sim/mj_loose.py)、[抓取入口](../test/bag_model/round3_flex.py)、[结果](../bag_sim/outputs/round3_flex/)、[主题报告](../test/bag_model/REPORT.md) | 本机检索未发现该版独立夹取 MP4。另有 [Newton 牛皮纸袋装物静置动画](../bag_sim/outputs/kraft_contents.gif)，来源见 [袋子说明](../bag_sim/README.md) |
| 2026-10-22，四类软包与柔性装填归档 | [包说明](/home/ubuntu/Dev/Projects/real2sim/soft-package-assets-20261022/START_HERE.txt)、[加载入口](/home/ubuntu/Dev/Projects/real2sim/soft-package-assets-20261022/load_bundle.py)、[七组示例索引](/home/ubuntu/Dev/Projects/real2sim/soft-package-assets-20261022/examples/flexible-fill/index.json) | [PE 外观](/home/ubuntu/Dev/Projects/real2sim/soft-package-assets-20261022/examples/flexible-fill/pe-large-soft/normal.mp4)、[PE 透明内视](/home/ubuntu/Dev/Projects/real2sim/soft-package-assets-20261022/examples/flexible-fill/pe-large-soft/transparent.mp4)、[牛皮纸袋下压](/home/ubuntu/Dev/Projects/real2sim/soft-package-assets-20261022/examples/flexible-fill/kraft-small-soft/normal.mp4)、[文件封抓取](/home/ubuntu/Dev/Projects/real2sim/soft-package-assets-20261022/examples/flexible-fill/paperboard-grasp-soft/normal.mp4) |
| 2026-11-09，同外袋书本与衣物的 YAM 抓取 | [场景模型](../policy/parcel_heap.py)、[实验报告](../test/policy/HIDDEN_CONTENTS_REPORT.md)、[全部录像说明](../video/README.md) | [书本与衣物并排](../video/book_clothes_comparison.mp4)、[书本](../video/book.mp4)、[衣物](../video/clothes.mp4) |

牛皮纸 flex 选定配置为 35 mm 网格、200 节点、5 mm 碰撞半径，刚性内容物步长 0.8 ms。报告记录该版本未通过完整行为验收。YAM 书本与衣物录像保留原实验的步长敏感性结论。上述归档均沿用原物理结论。

## 检查结果与限制

- 交付核对的 13 个模型、说明或媒体文件均存在且非空。
- flex 小球 MP4 为 1,987,222 字节，SHA-256 与 [既有检查记录](../test/flex_granular/results/20261109_ubun_gpu5s/video_checks.json) 一致。保存记录为 126 帧、文件时长 5.04 秒、仿真时间 0 至 5 秒。
- 本次尝试读取媒体元数据时，当前环境缺少 `ffprobe`，命令启动失败。未重新解码录像，也未运行物理仿真。
- 部分历史目录名与 manifest 路径已随项目整理发生变化，以上以本机实际存在的文件和对应说明为准。

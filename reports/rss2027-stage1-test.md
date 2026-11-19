# RSS 2027 Stage 1 首轮任务检查

最近检查时间 2026-11-19 21:38（北京时间）。机器 ubun，MuJoCo 4.0.1。完整实验配置与证据见[阶段报告](../test/bag_model/RSS2027_STAGE1_REPORT.md)。

| 范围 | 检查方法 | 结果 |
| --- | --- | --- |
| 单指原生接触 | 持共享锁运行 `rub_contact_e1.py`，碰撞开关各一次 | 旧刚体反力读数失败。改为从约束雅可比恢复反力后，正负对照均通过，最大反力差 $6.94\times10^{-18}$ N。 |
| 随机几何 | 持锁运行 `rss2027_geometry.py`，种子 20261119 | 128 个样本通过填充率、变化范围和余量非负检查。 |
| PE 微切片 | 持锁运行 `rss2027_micropatch.py`，比对余料、支撑、步长和摩擦 | 强支撑后穿透小于 7 μm，平膜未聚拢。预褶两档步长均出现正拉力，零指垫摩擦对照的正拉力为零。 |
| 整袋及 GPU | 统一 YAM 入口 2 ms，RTX 5090 静置与合指各 10 步 | CPU 装配与 GPU 短步进完成，GPU 左指接触力差最高 1.046 N。见[GPU 报告](rss2027-stage1-yam-gpu-probe-test.md)。 |
| MuJoCo 查看器 | `view_rss2027_fold.py --check` 核对预褶和仓库整袋轨迹，启动原生窗口后切换到侧抓褶皱近景 | 预褶首、中、末帧与归档节点在 1 pm 容差内一致。整袋顶抓 252 帧、侧抓 500 帧时间匹配。侧抓窗口以 `--start-time 7.42 --fold-edge 94 95` 打开。 |
| 褶皱视频 | `render_rss2027_fold_video.py --edge 94 95 --start .32 --end 8.20 --highlight-time 7.42`，FFmpeg 全量解码，ffprobe及两帧目检 | 1280×512，198 帧，25 fps，7.92 s。修订视角后，可同时看见整袋和夹口附近的膜面弯折。 |
| 既有袋模型单元检查 | `.venv-physical-parcel/bin/python -B test/bag_model/test_physical_parcel.py` | 16 项通过，测试夹具产生预期的 flex 诊断提示。 |
| 文件检查 | Python 源码语法检查、CSV 行尾规范化和 `git diff --cached --check` | 5 个 Python 文件通过解析，12 份 CSV 统一为 LF 行尾，暂存区格式检查通过。 |
| PE 下压力复测 | 持共享锁用 `rss2027_micropatch.py --dt .00005 --pad-contact-time .001` 与 `--pad-contact-time .0001` | 两组均完成，峰值实际下压力 4.86 / 12.13 mN，未达到 0.4–0.6 N 验收带。 |
| GPU 接触差异 | 同状态逐点比对 CPU/GPU 合指首步 | 两个左指浅触点解释首步左指竖向反力差的约 92%。其后的轨迹分岔仍待验证。[同状态检查报告](rss2027-stage1-yam-contact-parity-test.md) |

还需验证平膜起皱、材料曲线、脱手终点和闭环 GPU 基线。关键运行文件经仓库归档，其余探索轨迹和冗余快照保留在本机运行目录。

## 空袋整袋修复检查

本轮在 ubun、MuJoCo 4.0.1 上使用仓库 YAM、142 节点封口双膜片和 PE 预设，按用户要求改为整袋试验。正式动力学持 `/tmp/claude-gpu.lock`。

| 范围 | 检查方法 | 结果 |
| --- | --- | --- |
| 原试片反力 | `normal_probe.py` 和 `normal_dwell.py` | 静态触点力与广义反力吻合，原接触设置在恒位移足够时能传递超过 0.5 N。 |
| 空袋与力门槛 | 检查编译模型、指尖几何和连续 10 ms 门槛；运行 20 ms 入口；`.venv-physical-parcel/bin/python -B test/bag_model/test_physical_parcel.py` | 袋膜 142 节点、零内容物、零袋膜附着，保留 YAM 关节耦合。单元测试 17 项通过。 |
| 空袋静置 | `settle_probe.py` 与 `collision_probe.py` | 默认容差三档步长约 1 mm 自接触尖峰。原生凸碰撞收紧到 1 nm 后，两档步长的 0.30 s 静置均通过，峰值 0.150 mm。 |
| 空袋完整抓取尝试 | `record_physical_parcel.py` 读取 `rss2027-stage1-empty-yam-r2/config.json` | 2.2242 s 下压力门槛达标，直接碰桌反力零。总历时 4.3194 s，起抬后双指失联超过 50 ms，抓取失败。首次接触峰值 14.43 N。 |
| 空袋接近限速复测 | 同一入口读取 `rss2027-stage1-empty-yam-r3/config.json` | 3.4584 s 法向力验收通过，接近峰值降至 0.646 N、原生穿透峰值 0.150 mm、零告警、直接碰桌反力零。试验达法向门槛即停止。 |

本轮归档了完整尝试和独立加载复测，限速版的闭合与提拉尚待验证。后续记录整袋的连续参数样本、捕获与滑移临界点及完整 GPU 回合。

# RSS 2027 Stage 1 首轮任务检查

最近检查时间 2026-11-19 21:00（北京时间）。机器 ubun，MuJoCo 4.0.1。完整实验配置与证据见[阶段报告](../test/bag_model/RSS2027_STAGE1_REPORT.md)。

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

# YAM 平台归并检查

## 本轮计划（2026-11-11）

问题：公共代码迁入experience 目录 后，能否脱离外部机器人路径运行，并保持旧入口和既定行为。

配置与次数：仅 YAM 自身源包全文件 SHA256 校验一次；独立刚体适配器短入口一次；旧 native-flex/segmented 入口各一次；既有 I-37 测试一组；旧版与迁移版的机器人初始化和短控制轨迹对比一组。只做 CPU 装配/兼容验证，不运行正式抓取矩阵或测速。

通过标准：资产文件和来源完整，独立入口不激活内容物包；新旧机器人参数和控制状态一致；旧入口可用；实际公共源码包含在新结果快照中。通过后记录检查并提交；失败则修复本次迁移后复查相关项，不修改物理阈值。

## 结果

通过。机器 `ubun`，已有 `.venv`（MuJoCo 3.17.0），2026-11-11。按用户最后确认的范围，只纳入统一实验平台与 YAM 自身资产；推送到现有 randomization，experience 是普通目录。

| 检查 | 方法与结果 |
| --- | --- |
| 新旧行为 | `.venv/bin/python -B experience/tests/check_migration.py`：以 `7cf1d67` 为迁移前基线，两种侧抓方向及公共刚体场景，各 25 步；XML、惯量、限位、接触参数一致，qpos/qvel/ctrl/actuator_force 最大绝对差均为 **0** |
| YAM 资产 | 与共享原包逐字节比较：**34/34** 文件相同；MANIFEST 的 **33/33** 项 SHA256 一致。只复制 i2rt-yam-menagerie，未纳入其他资源包 |
| 独立入口 | 在 experience 中运行 `python -B -m yam_platform.angles --smoke-seconds .02 --output outputs/consolidation/standalone`：`smoke_only`；公共模块不导入冻结内容物 |
| 旧模型入口 | 原 angles.py 的 native-flex 20 ms、segmented 8 ms：均 `smoke_only`；模型源码仍由原模型适配器维护 |
| 既有受力测试 | `python -B test/flex3d/test_platform.py`：**2/2** 通过（连接拉离/压入符号、水平载荷） |
| 并行入口 | 原 run_suite.py，两个 4 ms 刚体用例、spawn 两进程：**2/2 smoke_only**；行为耗时未作为吞吐数据 |
| 视频与网页回放 | 原 angles.py 保存 60 ms native-flex；原 render_video.py 输出 H.264、1280×480、1 帧的入口样片，确认 llvmpipe CPU 渲染；原 export_tetra.py 成功输出 tetra.html |
| 快照与兼容 | 三次模型运行的源码快照逐项 SHA256 一致；包含真实 yam_robot/yam_geometry 等源码；独立子进程加载归档平台及其 YAM 资产成功 |
| 共享库检查 | `python -B workspace/asset/simulation-assets/manage.py check`：通过，32 个资产、1292 个清单文件；这是整个原共享库的检查计数，不是本次导入数量 |
| 静态检查 | 公共模块与修改的适配代码 AST 解析通过；自维护文件 `git diff --check` 通过 |

本次保留动作、阈值、默认模型、冻结材料代码和历史结果。未跑正式抓取矩阵、GPU 物理试验或 28 进程测速，不据此给出模型抓取可靠性结论。沿用已有环境，未安装或升级依赖。

父工作区已有其他未提交修改，使用隔离工作树准备提交，不暂存或覆盖其他会话的工作。

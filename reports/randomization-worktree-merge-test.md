# Randomization 三工作树合并检查

2026-11-25，北京时间，机器 `ubun`。合并基线为 `main` 的 `72b558fe`，纳入 PE 袋物理分支 `4907deb5` 和 Yam_B 纯仿真及 ABC 工作站分支 `54dc788c`。公共进展日志和 11 月 25 日运行索引有两处文档冲突，按时间顺序保留双方条目。

| 检查 | 结果 |
| --- | --- |
| Git 结构 | `git diff --cached --check` 通过，无未解析冲突。两个分支的代码与运行制品使用原提交内容。 |
| 统一 YAM 平台 | `experience/tests/check_migration.py` 通过。三个用例各 25 步，状态最大差均为 0，33 个资产清单文件及兼容导入通过。 |
| 烟测入口与装配 | `test/yam/smoke.py --help`、`test/bag_model/bag_mvp.py --help`、`test/bag_model/pe_physics_protocol.py --help` 均正常。用合并后的接口构建并初始化 Yam_B 与 PE Flex，读到 142 个膜顶点和 7 个执行器。 |
| 本地 Newton 依赖 | 复用了袋体工作树已有的 Newton 1.6.1 依赖，主目录的 `.venv-physical-parcel/benchmark-deps` 导入通过。 |
| 证据索引 | Yam_B 报告、袋体最小操作报告及共享日期索引的相对链接均可解析。 |
| 主目录保护 | 更新前核对了 6 份未提交文档与 12,334 个未跟踪文件。合并新增路径与现有文件无重叠。预演中的唯一文档冲突是本地 11 月 23 日记录与合并后 11 月 25 日记录的接续。 |

本轮是 Git 合并和兼容性检查。物理结果沿用各自的原始运行和验收状态。

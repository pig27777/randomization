# Git 历史与架构审计检查

最近检查于 2026-11-19 21:42（北京时间），机器 `ubun`。基线为主线 `b9ddb6a7` 和远端实验分支 `fbb4a0e4`。源码从 Git 提取至临时目录，适用候选只在该副本中试删。

| 范围 | 方法 | 结果 |
| --- | --- | --- |
| Git 历史与 PR | `git fetch origin`、`git ls-remote`、`git log`、`git rev-list`、GitHub Pulls API（`state=all`）及公开 PR 列表 | 主线 54 个提交，两个分支合计 55 个唯一提交。主线唯一合并为初始 README 汇合。开放和关闭 PR 均为 0。两线从 `ac0a963a` 分出，各有 5、1 个独有提交。 |
| 源码与入口 | `git ls-tree`、`git log --follow`、`git blame`、双分支 `git grep -I -F deadline_watch`、Python `ast.parse` | 245 个维护源码和测试通过语法检查。看守脚本名在两个基线均只见于 `test/MIGRATION.md` 和 `test/bag_model/README.md`。两个版本索引合计 7 个候选文件路径在两线均未提交。 |
| 固定发布与平台迁移 | 校验 v1.3 发布包和清单内成员；`.venv/bin/python -B experience/tests/check_migration.py --baseline 7cf1d679` | 发布包 SHA256 匹配，32/32 个成员匹配。YAM 资产清单 33/33 通过，独立及归档导入通过，三组各 25 步的状态最大绝对差为 0。 |
| 临时试删与回归 | 删除 `deadline_watch.py`，移除 `run` 内 `sign` 和 `niter_sum` 的赋值；用 `dis` 检查用法，重跑下列入口 | 两个变量均只有 `STORE_FAST`，试删后可编译。基线和试删副本各通过 27 项 CPU 回归与 3 项 CLI 检查。 |

回归从基线提取 `experience/`、`bag_sim/`、测试入口及 v1.3 发布包后执行。MuJoCo 3.17.0 下运行 `python -B -m unittest discover -s test/flex3d -p test_platform.py`（2 项）和 `-s test/bag_model -p test_side_pinch_adapter.py`（9 项）。MuJoCo 4.0.1 下运行 `-s test/bag_model -p test_physical_parcel.py`（16 项）。CLI 检查为 `round3_flex.py`、`round3_effect.py` 和 `flex_round2.py` 的 `--help`。

F6/F7 的历史 manifest 记录完成于 2026-11-07，期限为 2026-11-08。实际使用的 F6/F7 与 B1–B4 期限读取仍位于各批次驱动；共享函数的调用关系见 [跨线记录](../log/ideas.md)。

检查覆盖 Git 提交态和仓库内可见调用方。整袋 GPU 回合、吞吐和物理验收沿用原实验报告，本轮进行了离线检查及 CPU 回归。Stage 1 报告另记录 17 项单元检查，本轮的 16 项整包回归对应上述 Git 基线。

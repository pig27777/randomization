# 代码与测试目录整理检查报告

最近检查：2026-11-09 17:07:11 +08:00；机器：`ubun`。

结论：模型目录与独立测试入口已分开，测试同时提供对象导航和日期索引。最终持共享锁的 88 项回归全部通过，30 个 CLI 帮助入口通过；旧 Newton / MuJoCo 3.14 环境受本机条件限制，未作动力学重验。

## 整理范围

- 61 个独立检查、验收、分析、演示、测速和单元测试文件归入 `test/bag_model/`、`test/content_model/`、`test/contact_transfer/`；[逐文件映射](../test/MIGRATION.md)。
- `bag_sim/`、`contents/` 维护建模实现、参数、几何与共享运行接口；[袋子版本/方案](../bag_sim/README.md)、[内容物版本](../contents/README.md)。共享场景构建和后端激活与测试执行解耦。
- 两个旧模型目录连同原结果共 62 个文件归入 `contents/legacy/`，不删历史失败或原始数据。
- `test/runs/` 按 `YYYY-MM-DD/<对象>/<run-id>/` 管理新轮次；8 个日期索引链接 145 条已有报告记录，区分计划、结果、审阅与补充记录，不推测缺失的执行日期。
- 根 `.ignore` 只控制默认代码搜索，保留 Git 跟踪；`rg --no-ignore-dot` 可包含快照与历史证据。

## 最终检查

| 方法或命令 | 结果 |
| --- | --- |
| `flock /tmp/claude-gpu.lock .venv/bin/python -B -m unittest discover -s test/content_model/tests -t test/content_model -v` | 64/64 通过，包含冻结归档与迁移边界的 3 项新增回归 |
| 同一共享锁内执行 `.venv/bin/python -B -m unittest discover -s test/bag_model -p 'test_*.py' -v` | 22/22 通过 |
| 同一共享锁内执行 `.venv/bin/python -B -m unittest discover -s test/flex3d -p test_platform.py -v` | 2/2 通过 |
| 30 个迁移入口分别运行 `--help` | 30/30 通过；命令与退出码见 [CLI 记录](../test/runs/2026-11-09/repository/code-cleanup/cli-checks.json) |
| AST 语法、建模目录对测试驱动的导入检查 | 默认搜索可见的 198 个 Python 文件语法通过，建模目录反向导入独立测试驱动为 0 |
| 移动前后 SHA256 与内容清单 | 62 个旧模型归档文件逐字节一致，423 个冻结/结果目录中的源码指纹不变 |
| 抽取前后 AST 对照 | `FilmPinch`、接触力计算、F7 执行流程及 5 个场景构建函数共 8 项逻辑保持一致 |
| 文档相对链接、标题锚点及 `git diff --check` | 21 份文档、349 条相对链接与标题锚点通过，旧命令通过迁移表定位 |
| 清理经过 Git 忽略和未跟踪检查的工作区字节码 | 4 个缓存目录，共 431,394 字节；冻结/结果目录及完整旧模型归档中的内容保留 |

完整回归日志见 [unit-checks.log](../test/runs/2026-11-09/repository/code-cleanup/unit-checks.log)，命令、机器与结果见 [unit-checks.json](../test/runs/2026-11-09/repository/code-cleanup/unit-checks.json)。

## 失败与修复记录

- 首次 CLI 探查 28/30 通过，两个旧 F6/F7 入口在解析 `--help` 前检查已过期的运行期限。调整为先解析参数、实际执行时仍检查原期限；最终 30/30 通过，期限和验收阈值不变。
- 首次内容物检查 60/61 通过，离线冻结审计把测试脚本原路径不存在视为冻结模型损坏。现同时校验原归档及其每项源码、当前模型源码，并单独记录已迁移驱动的当前指纹；不放宽本构或任务验收条件。新增归档损坏、模型损坏/丢失、驱动迁移回归后，持锁重跑为 64/64。

## 限制与未提交工作

本轮验证目录迁移、导入、回归与来源完整性，不新增物理验收或吞吐结论。旧 Newton 与 MuJoCo 3.14 解释器路径在本机不存在，未安装或升级环境；相关入口完成语法和路径检查，未宣称其动力学运行通过。

其他会话已有修改和未跟踪文件保留，涉及迁移的随原文件移到新位置；本轮提交只包含目录整理所需的变更，不代交其他会话的实验实现或结果。冻结包、历史科学记录及并行实验输出不重写。

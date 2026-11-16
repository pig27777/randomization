# Rub-to-Pinch Phase 0 / E1 检查

最近检查：2026-11-16T22:51:34.926538+08:00，机器ubun，源码起点ac0a963a。

**结论：Phase 0审计完成；E1仅入口检查通过，正式压入/保持/退出测试受共享锁阻塞，未验证。**

| 范围 | 方法/命令 | 结果 |
| --- | --- | --- |
| 本地更新 | git pull --ff-only，按用户要求以origin/main覆盖旧改动；HEAD/origin/main哈希核对 | 更新至ac0a963a；详见[同步检查](repository-update-test.md) |
| 环境 | uv venv .venv-physical-parcel；uv pip install --python .venv-physical-parcel/bin/python -r envs/physical-parcel-requirements.txt | MuJoCo/MJWarp4.0.1、Warp1.19.0；仅项目独立环境 |
| 源码语法 | ast.parse 两个rub_contact_*.py | 通过 |
| 存档审计 | .venv-physical-parcel/bin/python -B test/bag_model/rub_contact_audit.py --output test/runs/2026-11-16/bag/rub-contact-e1-r1/audit-surface | 644帧运动学/几何复算，无积分；四个主要源码哈希与原记录一致 |
| E1短入口 | .venv-physical-parcel/bin/python -B -u test/bag_model/rub_contact_e1.py --smoke --output test/runs/2026-11-16/bag/rub-contact-e1-r1/cpu-entry | 通过；10步准备+10步悬空，零指尖候选、零warning、neq=0；未持GPU锁的CPU入口检查，不作物理验收 |
| 差异格式 | git diff --check | 通过 |
| 正式E1 | [准备好的命令](../test/runs/2026-11-16/bag/rub-contact-e1-r1/planned-commands.json) | 未执行；共享锁被另一正式套件长期持有，未中断其他进程。CPU并行例外尚未获用户回复 |

审计确定：最新失败轨迹并非始终零指尖力；旧observe漏读Flex，但当前控制使用另一条正确识别Flex的路径。实际指尖核心在close/lift有19/175个采样帧相交，负开度超行程且左右指碰撞被过滤。独立Rub/Hold未建立；以上不构成唯一抓取失败根因的证明。

原始记录与限制见[主题报告](../test/bag_model/PHYSICAL_PARCEL_REPORT.md#rub-to-pinch-phase-0--e12026-11-16ubun)、[审计摘要](../test/runs/2026-11-16/bag/rub-contact-e1-r1/audit-surface/summary.json)、[逐帧几何](../test/runs/2026-11-16/bag/rub-contact-e1-r1/audit-surface/geometry.json)、[入口原始日志](../test/runs/2026-11-16/bag/rub-contact-e1-r1/cpu-entry.log)。仅qpos的历史记录不含每帧qvel/ctrl/raw contact，故不伪造原力重算。

E1正式接触候选/约束/正反力记录尚无数据；E2/E3、网格/材料扫描、CPU/GPU动态对照、完整搓动抓取与物理回归均未运行。没有修改生产模型、材料、拓扑、摩擦、共享动作或成功标准。

补充：暂存差异检查首次发现复制的上游AUTHORS带额外末尾空行；移除这份非必需的冗余副本，保留源码原始版权头及完整Apache2 LICENSE，随后git diff --cached --check通过。

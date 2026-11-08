# log

记录进度和 idea。两份交接文档仍是各条线的权威说明，这里只记“发生了什么”和“想试什么”。

| 文件 | 内容 | 写法 |
|---|---|---|
| `progress.md` | 进度流水 | 按日期倒序追加；每条写做了什么、结果（带数字）、结论 |
| `ideas.md` | 想法池 | 一条一个；标状态：`待试` / `在试` / `成立` / `否定` / `搁置`，否定也保留并写原因 |
| `decisions.md` | 决策记录 | 和用户讨论后定下的方向、待确认的建议、待决事项；按日期倒序，标“已定”/“待确认” |

## 两条工作线

| 线 | 交接文档 | 平台 | 一句话现状 |
|---|---|---|---|
| 内容物 | `../contents/HANDOFF.md` | 四面体平台未定：Newton VBD 接触修复中；旧粒子模型在 MuJoCo 3.17 / MJWarp | 粒子键模型揉团、枕头受压塌；改四面体前先过接触验收，本构和装袋暂停 |
| 外层袋子 | `../bag_sim/README.md` | Newton 1.6.1 VBD（`bubble_mailer/.venv`）+ MuJoCo 3.14 对照 | 四种袋子同一套方程，第一阶段完成；塑性/记忆/折痕未做 |

两条线都**没有任何实物标定**，所有参数是设定值。

`papers.md`：文献清单，按用途分组。

实验报告放在 `../test/<主题>/`，这里只记进度和想法（见 `../AGENTS.md` 第 8 节）。接触诊断报告已移至 [REPORT_diagnosis.md](../test/contact_transfer/REPORT_diagnosis.md)。

2026-11-03 修复对照见 [REPORT_repair.md](../test/contact_transfer/REPORT_repair.md)；内容物第 2 项压密本构设计见 [compaction_model.md](../contents/newton/compaction_model.md)。平压基准的局部通过不代表完整平台已验收。

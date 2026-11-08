# 文档同步检查报告

最近检查：2026-11-03（Asia/Taipei）。

结论：静态检查通过。本任务只改文档和文件位置，没有修改仿真代码或运行仿真。

## 范围

- `contents/HANDOFF.md`：更新过时结论（Newton 四面体“不可用”、环境用途、文件地图、下一步、踩坑）。
- `bag_sim/README.md`：更正 `rigid_contact_k_start` 的说明。
- `AGENTS.md`：新增第 8 节“本项目协作约定”。
- `log/contact_report.md` 移至 `test/contact_transfer/REPORT_diagnosis.md`，同步更新 `log/README.md`、`log/progress.md`、`contents/HANDOFF.md`、`reports/log-update-test.md`、`test/contact_transfer/README.md` 中的引用。

## 方法与结果

- `k_start` 说明对照本机 Newton 1.6.1 源码：`solver_vbd.py:330`（默认 beta=0）、`:975`（负值报错）、`:995`（beta=0 时不用该值）。一致。
- 上述 Markdown 中的相对链接和 `../` 路径用 Python 逐一检查存在性：无失效链接。
- 全目录搜索 `contact_report`：只剩 `log/progress.md` 中的历史引用，已注明新位置。

## 限制

- 目录不是 Git 仓库，无法用 diff 核对；采用读回检查。
- 另一会话在本轮期间仍在修改 `log/`；若它之后重新写入 `log/contact_report.md`，需要按 AGENTS.md 第 8 节合并到新位置。

## 2026-11-03 追加：隔膜传力基准定义

- 新增 `test/bagged_contact/README.md`（基准定义）和 `REPORT.md`（第 1 轮计划，未运行）；`contents/HANDOFF.md` §7 第 1 项加入口。
- 参数核对来源：`test/contact_transfer/benchmark.py`（软块、膜、工具设置）、`test/gripper_grasp/grasp.py`（夹取流程）、`bag_sim/profiles/bubble.json` 和 `bag_sim/newton_bag.py`（气泡膜参数、碰撞半径 1.75 mm）。
- 相对链接与 `../` 路径存在性检查通过。未编写实现代码，未运行仿真。

## 2026-11-03 追加：κ 规格文档

- 新增 `contents/package_params.md`（设计，未实现）；`contents/HANDOFF.md` §1 加论文与规格引用，§4 文件地图加入口。
- 覆盖检查：`bag_sim/bag_model.py` 中 `DEFAULTS` 的 58 个参数全部在规格中有归属（κ 或平台配置）。
- 规格中引用的 `../`、`newton/` 路径存在性检查通过。未编写实现代码，未运行仿真。

## 2026-11-03 追加：内容物建模改为响应参数

- 重写 `contents/segment_model.md`；`contents/package_params.md` 的 κ_content 改为整体响应参数；新增 `test/content_model/README.md`（响应定义）；重排 `test/content_model/REPORT.md` 第 1 轮计划；更新 `contents/HANDOFF.md` 第 2、4、7 节和 `log/decisions.md`。
- 相对链接与路径存在性检查：除尚未创建的代码目录 `contents/segment/` 外均存在（该目录按计划由内容物线创建）。未编写实现代码，未运行仿真。

## 2026-11-03 追加：袋子线第 1 轮计划

- 新增 `test/bag_model/REPORT.md`（第 1 轮计划，未运行）；`contents/HANDOFF.md` 第 4、7 节加入口。
- 依据：`bag_sim/bag_model.py`（`DEFAULTS`、版本检查 433 行、`internal` 属性 585/609 行）、`bag_sim/README.md`（回归、速度、静置结果）。链接存在性检查通过。未改代码，未运行仿真。

## 2026-11-03 追加：第一版不单独建模袋子

- 依据用户确认（满装、松装都有；只用平行夹爪），更新 `contents/segment_model.md`、`contents/package_params.md`、`contents/proposal.md`、`contents/HANDOFF.md`、`log/decisions.md`，重写 `test/bag_model/REPORT.md` 第 1 轮计划（原计划保留为“已暂停”）。
- 修改前确认袋子线尚未开始第 1 轮（`bag_sim/outputs/round1/` 不存在、`bag_sim/` 无新改动）；内容物线已在 `contents/segment/` 开工，故新增外皮参数标为“第 1 轮之后加入”。
- 链接与路径检查通过；开题报告章节连续、表格列数一致、无本地链接。未改代码，未运行仿真。

## 2026-11-04 追加：RSS 2027 论文计划

- 新增 `paper/rss2027_plan.md`；更新 `log/decisions.md`、`contents/HANDOFF.md`、`contents/segment_model.md`、`contents/proposal.md`（松装必须覆盖）。
- RSS 2027 的日期、页数、匿名和引用要求来自官网 roboticsconference.org 及其 CFP 页（2026-11-04 读取）；计划中引用的论文仅列出本会话已检索到的条目，其余标为待检索。
- 链接与表格检查通过。未改代码，未运行仿真。

## 2026-11-04 追加：策略线与代理测试集计划

- 新增 `test/policy/README.md`（任务与 `evaluate` 接口定义）、`test/policy/REPORT.md`（策略第 1 轮计划）、`test/proxy_eval/REPORT.md`（代理真实测试集第 1 轮计划）；更新 `paper/rss2027_plan.md`（仿真跑通即上真机）、`AGENTS.md` 第 8 节（登记策略线、代理测试集归接触线）、`contents/HANDOFF.md` 文件地图。
- 链接检查：除尚未创建的代码目录 `policy/`、`proxy_eval/` 外均存在（按计划由各线创建）。未改代码，未运行仿真。

## 2026-11-04 追加：内容物第 1 轮审阅

- `test/content_model/REPORT.md` 追加“第 1 轮审阅意见”和“第 2 轮计划”；`test/content_model/README.md` 按审阅修改 4 项测法并加修改记录；`log/decisions.md` 加审阅结论。
- 审阅依据：报告第 1 轮结果全文、`contents/segment/mjwarp_speed.py`（吞吐按 0.25 ms 物理步计数）、`mj_adapter.py`（tendon 桁架实现）。未改代码，未运行仿真。

## 2026-11-04 追加：收窄战线（用户同意 7 条决定）

- 冻结任务协议 v1：重写 `test/policy/README.md`（以 `policy/env.py` 常量为准：力控维持、流程、判定先后、类别阈值、仿真失败判法与报告方式、开发集与测试集分开）。
- 更新：`test/policy/REPORT.md`（P5 取消、审阅意见与第 2 轮计划 Q1–Q4）、`test/proxy_eval/REPORT.md`（停止）、`test/content_model/REPORT.md`（冻结 v1 的要求）、`test/content_model/README.md`（“被弹飞”改为诊断触发）、`paper/rss2027_plan.md`（RJ 对照、去掉代理测试集、时间表；更正 arXiv 2303.04136 误标为 DROPO）、`log/decisions.md`、`log/ideas.md`（I-13–I-17、I-19、I-21、I-22 状态）、`contents/HANDOFF.md` 第 7 节。
- 链接检查通过。未改代码，未运行仿真。

## 2026-11-05 追加：袋子线优先级调整

- `test/bag_model/REPORT.md` 第 1 轮计划加“优先级调整”：C2 优先并接入分段内容物和冻结任务协议，C1 压缩为定性参考，判断标准 3 相应放宽（已告知用户），C3 只测 C2。未改代码，未运行仿真。

## 2026-11-05 追加：登记论文线

- `AGENTS.md` 第 8 节加入论文线（`paper/` 下的稿件、参考文献和图）；`paper/rss2027_plan.md` 第 6 节注明分工。未改代码，未运行仿真。

## 2026-11-05 追加：内容物形状 v1.1

- 更新 `contents/segment_model.md`（外形 v1.1 一节）、`contents/package_params.md`（`shape_p`、`shape_q`）、`test/policy/README.md`（观测加两个形状特征，修改记录）、`test/policy/REPORT.md`（训练推迟到 v1.1 后）、`test/content_model/REPORT.md`（v1.1 计划 V1–V6）、`paper/rss2027_plan.md`（参数与时间表）、`log/decisions.md`。
- 链接检查通过。未改代码，未运行仿真。

## 2026-11-05 追加：内容物冻结补充要求

- `test/content_model/REPORT.md` 在 v1.1 计划前加“冻结 v1 补充要求”（Newton 求解器重测、可行性扫描、I-27 接口）；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-05 追加：袋子第 2 轮与合拢时限

- `test/bag_model/REPORT.md` 加第 2 轮计划（flex 提速 F1–F4）；`test/policy/README.md` 合拢时限 4 s → 8 s 并记修改记录；`test/policy/REPORT.md` 加同步说明；`log/decisions.md` 更新。未改代码，未运行仿真。

## 2026-11-05 追加：以 YAM 为标准、顶面抓

- `test/policy/README.md` 改为协议 v2（YAM 平台、顶面抓、任务范围、指尖尺寸、修改记录）；`test/content_model/REPORT.md` 加 v1.2 计划；`contents/segment_model.md` 加“表层（v1.2）”；`paper/rss2027_plan.md` 任务与时间表；`test/bag_model/REPORT.md` 加 F5；`log/decisions.md` 加记录。
- YAM 夹爪行程 95 mm 来自经销商页面（roboticscenter.ai），夹持力未查到；已在文档中标为待实测。链接检查通过。未改代码，未运行仿真。

## 2026-11-05 追加：相机配置与观测

- `test/policy/README.md` 观测改为俯视 + 手腕侧视 RGB 轮廓特征（修改记录）；`test/content_model/REPORT.md` 的 V6 相应修改；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-05 追加：Isaac Lab 可行性计划

- 新增 `test/isaaclab/REPORT.md`（L1–L6）；`AGENTS.md` 第 8 节登记 Isaac Lab 线；`log/decisions.md` 加记录。本机核对：RTX 5090，驱动 580.178.04，磁盘剩余约 1.6 TB；`/home/ubuntu/Dev/Envs/` 下暂无 isaaclab 环境。未安装任何软件，未运行仿真。

## 2026-11-05 追加：GPU 训练为主路径

- `test/isaaclab/REPORT.md` 原则与结论规则更新、加第 2 轮预定计划 G1–G3；`test/policy/REPORT.md`、`paper/rss2027_plan.md`（训练方式与时间表）、`log/decisions.md` 同步。未改代码，未运行仿真。

## 2026-11-05 追加：内容物 v1.1 审阅

- `test/content_model/REPORT.md` 加“v1.1 审阅与下一步”（v1.2 优先、v1.1 恢复时的标准修改、论文数据需求）；`log/ideas.md` I-31 状态；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-06 追加：v1.2 修订计划

- `test/content_model/REPORT.md` 加“v1.2 审阅与修订计划”（T1′、R4、截止日）；`contents/segment_model.md` 表层修订说明；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-06 追加：v1.2 改为顶面捏取效果模型

- 按用户要求作废表层方案：`test/content_model/REPORT.md` 加新验收 E1–E6；`contents/segment_model.md` 以“顶面捏取（v1.2，效果模型）”替换表层一节；`contents/package_params.md` 加 `pinch_coeff`；`test/content_model/README.md` 加 R4；`log/decisions.md` 更正。未改代码，未运行仿真。

## 2026-11-06 追加：v1.2 改为顶面捏取效果模型

- 按用户要求作废表层方案：`test/content_model/REPORT.md` 加新验收 E1–E6；`contents/segment_model.md` 以“顶面捏取（v1.2，效果模型）”替换表层一节；`contents/package_params.md` 加 `pinch_coeff`；`test/content_model/README.md` 加 R4；`log/decisions.md` 更正。未改代码，未运行仿真。

## 2026-11-06 追加：顶面捏取的夹持力实现

- 回答内容物线的问题，选“等效夹紧弹簧”：`test/content_model/REPORT.md` 效果模型一节加补充说明；`contents/segment_model.md` 同步一句。未改代码，未运行仿真。

## 2026-11-06 追加：Isaac Lab 中期审阅

- `test/isaaclab/REPORT.md` 加“第 1 轮中期审阅与后续”（L5/L6 优先、L2′ 树结构原生模型、L3′ 按 κ 重新标定、L4′ 分桶）；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-06 追加：E4 审阅与协议 v2.1

- `test/policy/README.md` 改为 v2.1（提起 0.25 m、夹取失败判据、下压力限制，修改记录）；`test/content_model/REPORT.md` 加 E4 审阅；`test/policy/REPORT.md`、`log/decisions.md` 同步。未改代码，未运行仿真。

## 2026-11-06 追加：E5 审阅与协议 v2.2

- `test/policy/README.md` 改为 v2.2（下压力限制提前、夹持力读数定义，修改记录）；`test/content_model/REPORT.md` 加 E5 审阅（新增焊接约束跟随偏差 ≤ 5 mm 要求）；`test/policy/REPORT.md`、`log/decisions.md` 同步。未改代码，未运行仿真。

## 2026-11-06 追加：v2.2 E4 审阅

- `test/content_model/REPORT.md` 加“v2.2 E4 审阅”（单调性标准修订、w = 60 mm 诊断要求）；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-06 追加：v1.2 冻结后的安排

- `paper/rss2027_plan.md` 修订 R1 定义；`test/policy/REPORT.md` 加第 2 轮正式计划；`test/content_model/REPORT.md` 加 CPU 等价提速计划 S1–S3；`test/isaaclab/REPORT.md` 对照版本改为 v1.2；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-06 追加：Warp/MJWarp 提速计划

- `test/content_model/REPORT.md` 提速计划改用 Warp 并新增 S4–S6；`log/decisions.md` 加记录。核对：mjwarp 环境 Warp 1.19、MuJoCo 3.17；MJWarp 源码 `derivative.py` 含 `_qderiv_tendon_damping`，`types.py` 中 `eq_active` 为 (nworld, neq)。未改代码，未运行仿真。

## 2026-11-06 追加：S1/S2 审阅

- `test/content_model/REPORT.md` 加“S1/S2 审阅”（S2 标准改两层、取消 CPU 3 倍、直接 S4–S6）；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-06 追加：S2 响应层审阅

- `test/content_model/REPORT.md` 加“S2 响应层审阅”（离散测量量容差规则）；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-06 追加：GPU 首项审阅

- `test/content_model/REPORT.md` 加“GPU 首项审阅”（硬物下限不比较、改任务层检查）；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-06 追加：训练全部在 GPU 上

- `test/policy/README.md` 加批量评估接口；`test/policy/REPORT.md` 取消 CPU 训练、新增 Q5 批量学习器；`test/bag_model/REPORT.md` 加第 3 轮“松装袋效果模型”（B1–B4）；`paper/rss2027_plan.md` 时间表；`test/isaaclab/REPORT.md` 加组合要求；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-06 追加：GPU 对照标准统一

- `test/content_model/REPORT.md` 加“GPU 对照标准的统一修订”（五层标准、只三层停下）；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-06 追加：v1.3 计划

- `test/content_model/REPORT.md` 加“v1.3 计划：捏取改为点连接”（P1–P6）；`contents/segment_model.md` 加一句；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-06 追加：v1.3 计划

- `test/content_model/REPORT.md` 加“v1.3 计划：捏取改为点连接”（P1–P6）；`contents/segment_model.md` 加一句；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-06 追加：v1.3 加 P7

- `test/content_model/REPORT.md` v1.3 计划加 P7（顶面抓位置扫描）；`log/decisions.md` 更新。未改代码，未运行仿真。

## 2026-11-06 追加：日期更正

- 本会话此前把 2026-11-06 当天写的若干记录误标为 2026-11-07（系统时间为 2026-11-06）。已在 11 个 Markdown 文件中统一改回 2026-11-06，包括其他会话引用这些标题的锚点链接，确保链接仍然有效。

## 2026-11-06 追加：v1.3 P2 审阅

- `test/content_model/REPORT.md` 加“v1.3 P2 审阅”（解开条件改为 5 ms 低通）；`log/decisions.md` 加记录。未改代码，未运行仿真。

## 2026-11-06 追加：优先级与 L5 补充

- `test/isaaclab/REPORT.md` 加“L5 / L6 补充”（填充程度缩放、加速手段对照、v1.3 捏取）；`test/bag_model/REPORT.md` 第 3 轮补充；`log/decisions.md` 记当前优先级。未改代码，未运行仿真。

## 2026-11-06 20:20 · v1.3 冻结审阅
- 改动：`test/content_model/REPORT.md` 追加“v1.3 冻结审阅”（I-36、合拢失败率归入 I-26、GPU G1–G5）；`log/decisions.md` 加“续八”。
- 检查：核对 `acceptance.json` 七阶段 passed=true、`v1.3.json` 存在；I 编号此前最大 I-35，新号 I-36 不冲突。HANDOFF 已由内容物线更新为 v1.3，本次未改。
- HANDOFF 下一步第 0 项追加审阅要点（一处替换，断言唯一）。

## 2026-11-06 20:40 · 袋子 F6/F7 计划
- 改动：`test/bag_model/REPORT.md` 追加“第 3 轮并行项：flex 提速与袋膜捏取”；`log/decisions.md` 加“续九”。静态检查：标题唯一、引用的 F3 基准数值（76 mm、26 mm）与第 2 轮选定配置一致。

## 2026-11-06 21:00 · v1.3 GPU G2 审阅
- 改动：内容物报告追加“v1.3 GPU G2 审阅”（判法修订待用户同意、硬物 R3 诊断、G5 时间拆分、流程）；decisions 加“续十”。核对：引用的 4 次 R1y 数值、R3 18.92/7.42 N、冒烟 15.534 N 与报告“G1、G2 正式结果”一致。
- 21:20 内容物报告追加“G5 优化补充”（求解器容差、float64、按环境串行、无图捕获）；依据为对 mjwarp_v13.py/warp_v13.py/mj_adapter.py 的只读检查，未运行。

## 2026-11-08 · 袋子第 3 轮审阅、Isaac Lab L5 入口审阅
- 改动：袋子报告追加“第 3 轮审阅”；Isaac Lab 报告追加“L5 入口审阅”；decisions 加 2026-11-08 条（建议待决定）。核对：B2 记录 press_peak_N=20.008、pkg_weight_N=3.218 来自 bag_sim/outputs/round3_flex/effect/b2.jsonl；check_load 用模长见 contents/segment/pinch.py:195；接触缓冲数值引自 Isaac Lab 报告。

## 2026-11-08 · 硬物 R3 诊断审阅
- 改动：内容物报告追加“硬物 R3 诊断审阅”（I-38）；decisions 加续条，并把 11/08 平台/松装条标为用户已下发采用、判法改为已定。核对：Closer 只被 protocols.py 内部及验证/标定脚本引用，top_grasp.py、policy_protocol.py 不引用（rg 结果）；增益公式见 protocols.py:131。

# 内容物v1.2顶面捏取效果模型检查

最近检查：2026-11-06 15:58（Asia/Taipei），E4审阅后续轮结束。E1–E6及≤5mm静态跟随要求全部通过，v1.2已打本地冻结标签。以下保留各轮历史，最新状态在文末。

## 范围与方法

恢复指纹一致的冻结v1物理模型，只增加κ的pinch_coeff与独立抓取效果模块；旧表层源码及结果保留。正式运行均用`flock /tmp/claude-gpu.lock`排队；Python为`/home/ubuntu/Dev/Envs/mjwarp/bin/python -B -u`，入口`contents/segment/validate_pinch.py`，新run-id拒绝覆盖。

| 检查命令（入口后的参数） | 结果 | 证据 |
|---|---|---|
| `units --run-id e1_formal --workers 1` | 31/31通过，8.029s | [日志](../contents/segment/results/pinch_effect_v1_2/e1_formal.log) |
| `response --run-id e2_reference --reference --workers 4`及`response --run-id e2_candidate --workers 4`；`compare --run-id e2_comparison --workers 1` | 12/12；全部物理结果、编译XML和采样qpos/qvel字节摘要相同 | [对比](../contents/segment/results/pinch_effect_v1_2/e2_comparison/summary.json) |
| `r4 --run-id e3_formal --count 3 --workers 3` | 9/9；硬0，易弯1→1.007514，易压2→2.009549，误差均<1% | [原始记录](../contents/segment/results/pinch_effect_v1_2/e3_formal/summary.json) |
| `units --run-id e1_final_logging --workers 1` | 记录范围修正后31/31通过，8.071s | [日志](../contents/segment/results/pinch_effect_v1_2/e1_final_logging.log) |
| `scan --run-id e4_formal --workers 8` | **未通过**，513/513完成、执行异常0；易压默认0/81成功 | [日志](../contents/segment/results/pinch_effect_v1_2/e4_formal.log) |

E1覆盖原24项以及双指顶面接触/3mm压入/刚体拒绝/原生weld承载/连续20ms超载解除/等效弹簧/采样配对等7项。R4合并weld与真实竖直接触反力，未用设定阈值伪造测量。非顶面抓的适配器、本构、测法源码与v1相同。

用户确认的弹簧只在捏取条件成立后启用：40N/mm、从压入后实际间距起再收拢10mm才受力，控制/承载/挤压用当前反力。试跑发现易压物提起不足50mm；加硬抓取weld未解决且破坏R4命中，已撤回到通过E3的设置。探索数值选择和全部失败输出保留，详见[主报告](../test/content_model/REPORT.md)末节。滑移日志已排除松开后的正常分离，不改动力学。

候选源码归档及SHA在[指纹](../contents/segment/results/pinch_effect_v1_2/candidate_source.json)，不是冻结标签。v1包SHA仍为`a3c78255e885ec2bf339012abfa1ad6ce0599d531f217b941199194105cf49f5`。没有安装包、改共享库或修改其他线代码。

## v2历史E4证据与当时限制

- 易弯pinch_coeff=1/0.5/0.25成功27/81、11/81、2/81，同种子成功集合满足包含关系，且都有超载解除；硬物27/27夹取失败。
- 易压pinch_coeff=2/1/0.5全部0/81成功。默认系数2没有超载解除，提起保持时COM升高30.46–43.06mm，最高43.055mm，未达到协议50mm。
- 易弯w40/d30mm的27次压缩越限、w60/d30mm的27次弯曲越限均在press、绑定前发生；共54例，不能当作E5随机200组失败率。未改阈值、删断言或排除失败动作。
- E5、E6未运行；不声称随机稳定≤1%或速度≤1.5倍，不用探索和并行墙钟替代正式速度。下一轮优先在既有协议内补查未覆盖的3–9mm浅压入及顶面基准，再由总体判断审阅行程和深压入问题（I-33）。本轮已停止，不将本次网格无解推广到全部连续动作。

[验收状态](../contents/segment/results/pinch_effect_v1_2/acceptance.json) · [完整成功区域](../contents/segment/results/pinch_effect_v1_2/analysis/success_region.csv) · [513次动作与结果](../contents/segment/results/pinch_effect_v1_2/analysis/feasibility.csv) · [54例失败参数/阶段/证据索引](../contents/segment/results/pinch_effect_v1_2/analysis/simulation_failures.csv)。每例原始JSON保留κ和失败时qpos/qvel/警告计数。

静态[审计](../contents/segment/results/pinch_effect_v1_2/analysis/audit.json)确认：正式E4源码与当前一致，候选归档逐文件SHA一致，四个普通物理核心模块与v1逐字相同，E3的pinch模块与最终相同，513例原始记录齐全。冻结v1包与Newton换算表SHA均不变。归档为候选记录，**未达到v1.2发布和正式训练条件**；策略线和Isaac Lab线继续固定v1。结论已回写HANDOFF/segment_model，progress为4行，跨线问题写入I-33。

交付静态检查：本轮报告/流水的21个本地链接均存在；动作CSV为513行、仿真失败索引54行、分组成功区域162行，与原始数据一致。无活动正式实验遗留。

## v2.1续轮（2026-11-06，正式结束）

用户与总体判断明确升级协议：提起250mm、提起保持后按实际桌面接触判断、两指真实竖直接触反力合计20N提前停止下压。v1本构/连接参数、夹紧弹簧、pinch_coeff范围均不改，扫描d新增5mm；全部旧源码包与失败结果保留。

| 检查 | 正式命令（同入口，均持共享锁） | 当前结果 |
|---|---|---|
| 相关单元 | `units --run-id v21/units_formal --workers 1` | 32/32，8.066s；新增原生桌面接触/完全离桌检查 |
| 共用R4受影响路径复核 | `r4 --run-id v21/r4_regression --count 1 --workers 3` | 三锚点各1次通过，测得0/1.007401/2.009549 |
| E4 | `scan --run-id v21/e4_formal --workers 8` | **通过**，684例、执行异常0、确认仿真失败0 |
| E5 | `random --run-id v21/e5_formal --count 200 --seed 20261119 --workers 8` | **未通过**，20/200=10%确认失败、执行异常0 |
| E6 | 上项通过才运行 | **未运行**，按E5失败即停 |

小试验证深压入约8.9mm即因20N停止，易弯完成搬运；易压完全离桌、悬挂伸长约66mm>厚度一半60mm，已记录为诊断，不改模型。小试耗时不作速度验收。悬挂诊断以同一材料点的静置位置作基准，同时记录指尖中点、材料点和约束跟随误差，未进入提起保持则记未测得。

新结果/候选指纹在[本轮目录](../contents/segment/results/pinch_effect_v1_2/v21/candidate_source.json)。主报告续节先写计划，现已填入正式结果；v1包/换算表仍为正式基线。

### v2.1关键数据与限制

- E4默认系数：易弯52/108、易压12/108成功；减小系数后分别32/108、3/108与6/108、0/108，成功集合单调包含，出现超载滑脱；硬物36/36夹取失败。实际桌面接触判据替代COM高度，未改本构。
- 成功易压案例的保持平均悬挂变化70.58–71.41mm>厚度一半60mm。材料点–COM变化11.18–13.88mm、约束跟随偏差57.52–59.40mm；不能全部归因于材料拉伸。全部453例进入保持的诊断已保留，本轮不改模型。
- E5随机200组静置全部通过，顶面抓确认失败20例：易弯18/68、易压2/56、硬物0/76。14例在approach确认、6例在close确认；压缩越限10、弯曲越限7、相邻重叠>50%三例，均有参数和完整状态，均未绑定pinch weld。有效180例、成功3例，成功/有效1.67%，按全部尝试为1.50%；这是随机动作稳定测试，不是策略性能。
- 20N停止当前只覆盖双指接触后的继续下压；单指先接触的接近入口与无有效捏取时合拢的真实载荷尚有执行边界，已记I-34，未自行追加规则或删去失败动作。全部失败保留，未因速度诊断直接判失败。
- E6未运行，未宣称速度达标，未打v1.2标签。策略/Isaac Lab线继续用v1；HANDOFF/segment_model已回写，progress续轮为4行。

[验收状态](../contents/segment/results/pinch_effect_v1_2/v21/acceptance.json) · [E4完整区域](../contents/segment/results/pinch_effect_v1_2/v21/analysis/e4_success_region.csv) · [684次动作记录](../contents/segment/results/pinch_effect_v1_2/v21/analysis/e4_feasibility.csv) · [悬挂诊断](../contents/segment/results/pinch_effect_v1_2/v21/analysis/suspension.csv) · [E5逐例失败参数/阶段/状态索引](../contents/segment/results/pinch_effect_v1_2/v21/analysis/e5_simulation_failures.csv)。

[静态审计](../contents/segment/results/pinch_effect_v1_2/v21/analysis/audit.json)确认：本轮4个正式阶段的源码/协议指纹均匹配当前文件；候选归档逐文件SHA一致，仅3份Python相对v2改变；普通物理核心4模块、v1冻结包SHA和换算表SHA均不变；684+200份原始记录及20份失败状态齐全。无包安装、共享库改动或其他线代码改动。

v2.1交付静态检查：19个本轮本地链接均存在；684行动作、216行分组区域、453行保持诊断、20个唯一失败索引与原始记录一致；流水4行，正式批次均已结束。

## v2.2续轮（2026-11-06，正式结束）

按用户审阅补齐执行：任一指接触起受20N竖直限制；真实横向反力平均与等效弹簧反力相加，5ms力控和20ms过度挤压均使用总读数，当前承载上限也使用总夹持力。仅将效果weld改为solref=5ms/solimp=.99/torquescale=1，内容物本构和采样不改。静态跟随误差用保持末0.1s中已绑定材料点相对指尖中点的三维偏差，门槛5mm。

| 检查 | 正式命令（同validate_pinch.py入口、均持共享锁） | 当前结果 |
|---|---|---|
| 相关单元 | `units --run-id v22/units_formal --workers 1` | 34/34，9.360s；新增任一指接触与真实力/弹簧力合成检查 |
| R4 | `r4 --run-id v22/r4_formal --count 3 --workers 3` | 9/9；硬0，易弯.978470，易压2.010443，误差−2.153%/+0.522% |
| E4 | `scan --run-id v22/e4_formal --workers 8` | **未通过**；684次完整结束、执行异常0、确认仿真失败0；成功集合有反例 |
| 静态跟随 | 同E4，最后0.1s已绑定材料点三维距离 | 易弯180例最大0.5402mm、易压264例最大1.5686mm，均≤5mm；硬物不绑定，未测 |
| E5/E6 | E4通过后以新种子20261126跑200，再跑三类各10次串行速度 | **未运行**，E4失败即停；种子未使用，速度未验收 |

参数探索保留在v22/weld_pilot*.log及各组合目录。最硬组合曾使R4易弯误差超10%，未选用；最终参数小试两软物偏差约.48/1.41mm，边界组合约.52–1.39mm。旧执行失败六例小试五例不再确认失败，116仍失败，保留记录，不据此排除样本；正式结论只看已规划的完整E4及新种子E5。

本轮[源码指纹](../contents/segment/results/pinch_effect_v1_2/v22/candidate_source.json)，候选包SHA为`e75f2a8cc718bcde7d99b6ea34afa170e712784e8960482eec62937232e23919`。此前结果、v1正式包和换算表保持不变。

E4系数从0.25×→0.5×→1×，易弯成功29→30→56/108、易压0→34→60/108，硬物36/36拒绝；易弯有2个相邻系数反例（同一实际动作的d20/30mm请求），易压有4个落点反例。低系数解除weld后真实接触仍能携物，高系数延后解除或保留到open，改变了后续运动；本轮保留协议判定，未把成功总数单调替换严格集合包含。原始记录[684行动作](../contents/segment/results/pinch_effect_v1_2/v22/analysis/e4_feasibility.csv)、[216行成功区域](../contents/segment/results/pinch_effect_v1_2/v22/analysis/e4_success_region.csv)、[反例索引](../contents/segment/results/pinch_effect_v1_2/v22/analysis/monotonicity_counterexamples.csv)（相邻6行、含跨档共8行）。

当前易压已测悬挂变化最大56.130mm<60mm，易弯19.955mm<25mm；完整[保持时序](../contents/segment/results/pinch_effect_v1_2/v22/analysis/hold_diagnostics.csv)同时记录材料点与约束误差。不把锚点扫描0次确认失败推广为新随机200组通过，也不把并行耗时当E6。旧116探索失败未删；v2.1 E5失败历史保留。

静态检查方法：`python3 -B contents/segment/results/pinch_effect_v1_2/v22/analysis/analyze.py`，只汇总已有JSON并核对SHA，不运行仿真。[审计](../contents/segment/results/pinch_effect_v1_2/v22/analysis/audit.json)通过：三项正式源码/协议一致、候选归档31文件一致、四个普通物理核心与v1相同、684份原始记录与汇总逐项一致、v1包/表SHA不变。E2沿用已审阅12/12逐位一致，本轮未重跑。仅4个Python文件相对v2.1变化，结果分析脚本在本轮结果目录内。

本轮[验收状态](../contents/segment/results/pinch_effect_v1_2/v22/acceptance.json)为not_frozen，无v1.2标签；[主报告v2.2续节](../test/content_model/REPORT.md#v12-效果模型验收续协议-v22)、HANDOFF、设计与progress已回写，跨线审阅问题I-35。策略/Isaac Lab线继续冻结v1；后续先审阅集合单调要求，不带未通过项进入下一阶段。

交付静态检查：本次核对的39个本地文件链接均存在；动作/成功区域/反例CSV分别684/216/8行，保持诊断9486行，与原始记录一致；当前候选源码及归档SHA仍匹配，E5/E6目录未创建。记录为 `v22/analysis/delivery_check.json`；文档更新没有再改运行源码或增加仿真。

## E4审阅后核对与E5/E6（2026-11-06，正式结束，已冻结）

本轮先在主报告写计划，采用总体判断与用户明确修订的E4标准；结果放新目录 `contents/segment/results/pinch_effect_v1_2/v22_review/`，保留旧严格集合失败记录，不改物理源码。

| 检查 | 命令/方法 | 结果 |
|---|---|---|
| E4现有数据核对 | 持锁 `review.py audit-e4`；核对684份JSON与feasibility | **通过**：未解开且挺过提起保持集合易弯48→48→84、易压51→84→102，均包含；成功数不减，反例6/216=2.778%≤5% |
| w60资格/接触诊断 | 小试一次后，持锁 `review.py diagnose --name ... --w ... --d ...`，w60/d5、w60/d10、w40/d5、w80/d5，F均5N，各1次 | 全部绑定。源码已支持跨段；w60真实力先达目标、未走满10mm弹簧行程，提起后力衰减并超载，不是同段拒绝；未改资格条件，无E4组合需重跑 |
| E5入口小试 | `validate_pinch.py random --run-id v22_review/e5_smoke --count 3 --seed 20261126 --workers 1`，非正式 | 3例完成、执行异常0、确认失败0；不作速度结论 |
| E5正式 | 持锁 `random --run-id v22_review/e5_formal --count 200 --seed 20261126 --workers 8` | **通过**，确认仿真失败0/200、执行异常0、静置200/200通过；成功8、夹取失败131、过度挤压59、掉落2 |
| E6入口小试 | `speed --run-id v22_review/e6_smoke --count 1 --workers 1`，非正式 | 三类各1次完成，0.282/8.903/11.673s；不作速度结论 |
| E6正式 | 持锁 `speed --run-id v22_review/e6_formal --count 10 --workers 1` | **通过**，三类各10次串行CPU，中位0.281240/8.920371/11.707560s，侧夹比0.4049/1.3135/1.4857，均≤1.5；执行异常/确认仿真失败均0 |

[E4核对](../contents/segment/results/pinch_effect_v1_2/v22_review/e4_review.json) · [逐例保持事件](../contents/segment/results/pinch_effect_v1_2/v22_review/e4_lift_hold_events.csv) · [w60诊断](../contents/segment/results/pinch_effect_v1_2/v22_review/w60_diagnosis.json)。w60两例未经滤波真实力峰值91.56/65.70N、20ms读数峰值4.295/5.118N、弹簧峰值0；w40/80弹簧峰值5.90/5.93N、完成搬运。协议与材料参数均未调整。

E5类别样本数硬77/易弯56/易压67，均无确认失败；[200行参数/动作索引](../contents/segment/results/pinch_effect_v1_2/v22_review/analysis/e5_cases.csv)、[失败索引](../contents/segment/results/pinch_effect_v1_2/v22_review/analysis/e5_failures.csv)0行（仅表头）。每份原始JSON均保留κ、动作及静置/抓取结果；旧失败历史保留。E6硬物10次为提前夹取失败；易弯8成功、2次搬运掉落，仅成功8次中位8.926189s；易压9成功、1次放下后落点失败，仅成功9次中位11.681066s。所有30次纳入速度中位，详见[逐次时间](../contents/segment/results/pinch_effect_v1_2/v22_review/analysis/e6_times.csv)。易压中位距11.820042s上限约0.1125s，速度余量较小。

持锁 `python -B .../v22_review/finalize.py` 汇总与归档，未运行额外仿真；[审计](../contents/segment/results/pinch_effect_v1_2/v22_review/analysis/audit.json)确认5个正式阶段源码/协议一致、候选归档31文件一致、v1普通核心四文件及v1包/表SHA不变。运行源码相对已完成E1/R4/E4的v22没有变化；E2仍沿用已审阅12/12逐位一致，本轮未重跑，不宣称重跑。

[验收状态](../contents/segment/results/pinch_effect_v1_2/v22_review/acceptance.json)全部通过；[本地v1.2标签](../contents/segment/results/pinch_effect_v1_2/v22_review/release_v1_2/v1.2.json)及同目录[冻结源码包](../contents/segment/results/pinch_effect_v1_2/v22_review/release_v1_2/v1.2.tar.gz)已生成，SHA256 `e75f2a8cc718bcde7d99b6ea34afa170e712784e8960482eec62937232e23919`；Newton表版本mapping_pass1。此项目无Git元数据，按既有v1方式用源码包及SHA标签冻结。策略/Isaac Lab线可固定此包做协议v2.2接入/训练；w60路径敏感、随机成功8/200、旧116探索失败、形状暂停、无实测/批量GPU吞吐等限制保留。没有安装包、修改共享库或其他线代码；报告、HANDOFF、设计及progress已同步。

最终交付静态检查通过：冻结标签、归档31文件、当前源码、验收JSON与4份依据SHA相符，v1包及表不变；本次检查61个本地文件链接均存在，保持事件/E5案例/确认失败/E6时间索引为684/200/0/30行。记录 `v22_review/analysis/delivery_check.json`。没有再增加仿真或修改运行源码，正式作业均已结束。

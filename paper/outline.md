# RSS 2027 扩展摘要：大纲与各节要点

2026-11-05（Asia/Taipei）。论文线维护。主张、假设和对照组以 [论文计划](rss2027_plan.md) 为准，本文件不改它们；与计划不一致或计划已过时的地方列在文末“需要总体判断会话确认”，并已写入 `../log/ideas.md` I-30。

正文用英文写在 `main.tex`（官方模板 IEEEtran，待建）；本文件是写作用的工作稿。所有数字只取各线报告的正式结果，注明出处；没有结果的地方写 TODO。

## 0. 题目、摘要与篇幅

**暂定题目**（沿用计划）：Response-Space Randomization of Reduced-Order Package Models for Sim-to-Real Grasping

**摘要草稿**（约 190 词；结果出来前不写任何数字）

> Robots that sort logistics parcels must grasp soft packages whose contents they cannot see: folded clothes that sag, pillows that crush, books that do not deform, and loosely filled bags whose contents slide inside the film. Measuring and modeling each item is infeasible, and high-fidelity deformable simulation is too slow and too fragile for large-scale randomization. We hypothesize that grasping such packages does not require accurate deformation, but only a randomized model that reproduces what the gripper feels. We propose a reduced-order package template in which contents are rigid segments connected by compliant six-DoF joints, a tightly packed bag is folded into an equivalent skin, and a loosely packed bag is represented explicitly. Object parameters are defined by whole-body responses under three fixed probes (squeeze, lift, transport) rather than by internal stiffnesses, and are reparameterized so that every sample is physically valid. Randomization ranges come from gripper limits and common-sense physical bounds instead of per-item measurement. We state four falsifiable hypotheses and a pre-registered evaluation that compares response-space randomization against rigid-body randomization and direct joint-parameter randomization on unseen real packages, together with ablations of individual response dimensions. Preliminary simulation results validate the template's stability and response control. TODO（定稿时按实际结果改写最后一句）

**篇幅分配**（正文 5 页 + 参考文献 1 页；模板为双栏）

| 节 | 页 | 图表 |
|---|---|---|
| I Introduction | 0.6 | 图 1（方法示意，占一栏上半） |
| II Related Work | 0.6 | — |
| III Problem Statement and Hypotheses | 0.5 | — |
| IV Method | 1.5 | 表 1（κ 参数、约束与映射，精简版） |
| V Experimental Design | 1.0 | 表 2（训练条件 × 检验的假设） |
| VI Preliminary Results | 0.5 | 图 2（响应空间覆盖）或表 3（模型验证汇总） |
| VII Expected Outcomes, Limitations | 0.3 | — |

## I. Introduction

要点（按段）：
1. **问题**：物流分拣中的软包，外包装相近、内容物差异大；夹爪感受到的差别是“顶住还是压扁、提起是否下垂、搬运时是否滑脱或晃动”。逐件测量不可行，所以用随机化；但随机化什么、随机到多宽，没有现成答案。
2. **现有两条路线的缺口**：刚体域随机化只随机尺寸、质量、摩擦，表示不了压扁、下垂和袋内滑动；高精度可变形仿真（FEM、MPM、薄壳袋子）每个物体都要标定材料参数，且慢、在接触下不稳定，难以做大规模随机化。引用：动力学随机化 [Peng18]，软物抓取的 MPM 随机化 [Stress-guided]，可变形仿真器的仿真–真实差距 [Blanco-Mulero24]。
3. **核心想法**：不追求形变精度，只保留夹爪能感受到的整体响应。A = g_B(X)：一个固定模板 B，一组按响应定义的参数 X；参照跨形态控制中“模板 + 物理一致参数化 + 随机化”的做法 [XHugWBC]。
4. **研究问题**（原文照计划）：少量与抓取相关的整体响应，能否构成足够有效的随机化空间，让策略迁移到没见过的真实软包？
5. **贡献**（与计划 §1 一致，第 3 条写成“评估设计”，不写成已完成）：
   - 按响应定义参数的降阶软包模板（分段刚体 + 弹性关节；满装并入外皮，松装显式袋子；重参数化保证合法）；
   - 只用夹爪能力和常识物理界限确定随机范围的方法；
   - 一套预先写定的评估：四个可证伪假设、对照组 R0/R1/RJ/R2/R3 与消融、开发集与测试集分开、配对统计；以及仿真中的初步证据。
6. 图 1 放这里。

## II. Related Work

RSS 第一阶段只看直接相关性。分四段，每段最后一句写清本文的区别。候选文献见第 8 节；**只有在 `references.bib` 和 `reference_check.md` 里核对过原文的条目才写进正文**。

1. **域随机化与随机范围的选取**：视觉和动力学随机化 [Tobin17, Peng18]；随范围自动扩展或用真机数据调整分布 [OpenAI19 ADR, SimOpt, BayRn, DROPO]。区别：本文不用评估物品的真机数据调范围，范围来自夹爪能力和物理界限；随机化的对象是物体的响应，不是仿真器参数。
2. **形态随机化与物理一致参数化**：跨形态策略 [GenLoco, URMA, XHugWBC]；惯性参数的物理一致性和光滑参数化 [Wensing17, Rucker22]；把物体分布本身当作随机化对象的抓取 [Tobin18, EGAD]。区别：本文把“模板 + 重参数化”从机器人搬到被抓物体上，参数按夹爪层面的响应定义。
3. **可变形与易损物体的抓取**：FEM 抓取仿真 [DefGraspSim]；MPM 仿真加材料参数随机化、零样本迁移 [Stress-guided]；以物理参数为条件的策略 [GenDOM]；MuJoCo 中带物体软度随机化的夹持力控制 [Lach24]。区别：它们随机化本构参数（杨氏模量等）或接触柔度，本文随机化测得出的整体响应，并覆盖下垂和袋内滑动。
4. **降阶模型与袋子、包裹操作**：刚性段 + 弹簧关节的软体机器人仿真 [SoMo, SoMoGym]；模型降阶 [Goury18]；软体机器人的仿真到真实 [Schaff23, Tiboni23]；袋子操作与基准 [DeformableRavens, AutoBag, SLIP, Bag-All-You-Need]；内容物质心会移动的袋子抓取 [Coinbot]；快递软包的视觉抓取与真实软包的参数辨识 [Wang19, Manyar25]。区别：降阶的对象是被抓物体；袋子按满装、松装区分处理，松装时袋内滑动是显式建模的对象。

## III. Problem Statement and Hypotheses

- **任务**（照 `../test/policy/README.md` v2）：单个软包放在桌面上，平行夹爪（行程约 95 mm）从上方抓起，水平搬运 0.5 m，放入 0.2 × 0.2 m 目标框。软物和松装袋主要用顶面抓（指尖压入表面捏住一块，带起整个物体）；硬物只包括至少一个水平尺寸 ≤ 80 mm、能侧夹的小件。
- **动作**：一次输出一组抓取参数（抓取方式、夹持中心、朝向、下降前开口 w、压入深度 d、目标夹持力 F、提起速度、搬运峰值加速度），由固定流程执行，力控维持 F。
- **观测**：俯视有向包围盒（长、宽、高）和两个形状特征；可选夹紧前 10 mm 的力–开口曲线。质量、刚度不作为输入。
- **结果判定**：成功、掉落、夹取失败、过度挤压（易压物 15 N、易弯物 25 N、硬物以夹爪上限为界）；按事件先后。安全阈值对所有训练条件相同（I-13）。
- **形式化**：物体 A = g_B(κ)，κ = h(θ)；训练分布 p_c(θ) 随条件 c 变化；目标是最大化未见真实物体上的期望成功率。
- **假设 H1–H4**：照计划 §2 的表，每条写“预测 + 检验方式 + 不成立时能得到什么”。
- 写法注意：H1 的比较对象是同结构、同预算、只换物体分布的 R1；RJ 单独检验“按响应定义参数”本身的价值（I-17）。

## IV. Method

### A. 软包模板 A = g_B(X)
- 三部分：内容物、袋子、二者之间的接触与摩擦。物体参数只放物理量，求解器配置所有物体共用一份。
- 满装：袋子效果并入等效外皮（表面摩擦、可选外皮柔度、袋重计入总质量）。松装：袋子显式表示（见 D）。

### B. 按响应定义的参数与重参数化
- κ_content 的关键量用固定测法定义（`../test/content_model/README.md`）：R1 夹紧（2% 到 min(15%, 0.6 × 顶住压缩比) 的割线刚度；顶住压缩比）；R2 提起（下垂比）；R3 搬运（滑移、摆动，只检验不定义参数）。仿真和实物用同一测法。
- 重参数化：正量用 exp，区间量用 sigmoid，“小于另一参数”用比例 sigmoid，袋子尺寸由内容物导出；在 θ 空间均匀采样，即刚度类对数均匀、比例类 logit 均匀。与 [Rucker22] 的伪惯性 Cholesky 参数化对照着写：软包没有对应的矩阵约束，改用逐项映射。
- 响应 → 内部参数的换算：解析初估（串联弹簧、悬臂梁）+ 仿真扫描建表、插值；表外目标拒绝，不外推。
- 采样约束：自重压缩、接缝不落在指尖中心、屈曲可达、下垂 < 0.05 按近刚性。
- 表 1：精简版参数表（尺寸、形状 p/q、质量、质心、squeeze_y/z、lock_y/z、droop、ζ、活动质量、surface_friction、段数）。

### C. 降阶内容物模型
- 长方体段（v1.1 起裁成超二次曲面包络），相邻段之间 6 自由度弹性连接：连接面势能给弹性，零刚度的原生约束给限位、剪切锁死和压密支撑，原生隐式阻尼；关节锁死即硬物；可选活动质量。
- 压密用段间重叠近似，到限位时“顶住”；压密支撑由外部围压门控，卸载后释放（I-24）。
- 每段质量和惯量由尺寸和质量份额算出，天然物理一致。
- 平台：CPU MuJoCo，0.4 ms，Newton 求解器（`../test/content_model/REPORT.md`“冻结 v1 补充验收”）。
- v1.2 顶面抓所需的表层细分段：写成计划中的模型改动，结果出来前只描述设计（TODO）。

### D. 满装外皮与松装袋子表示
- 满装：并入外皮。是否成立待袋子线 F4（TODO），在论文中作为 H3 的一部分检验，不写成已知。
- 松装：当前候选为 MuJoCo flex 薄膜。第 1 轮显示它与 Newton 参考袋子定性一致，但速度不够训练，第 2 轮在提速（TODO）。薄板片已放弃，理由可写进论文作为“便宜表示需要能起皱”的一句观察。

### E. 策略
- 学抓取参数，不做端到端 RL；同一策略结构、同样的样本数，只换训练物体分布。
- 第 1 轮策略线用的是“成功概率模型集成 + 交叉熵法选动作”（`../test/policy/REPORT.md` 第 1 轮“实现”）。正式算法以 Q2 实际使用的为准（TODO）。

## V. Experimental Design

- **表 2**：训练条件 R0、R1、RJ、R2、R3 与消融（去掉压缩特性、下垂、活动质量、外皮；随机范围收窄一半），每行写它检验哪条假设。
- **控制变量**：同网络、同样本数、同奖励和安全阈值；R1 保留类别阈值；RJ 与 R3 共享几何和质量的采样入口。
- **仿真交叉评估**（策略线 Q3）：三个策略 × 三个分布，配对评估，按类别分开统计，各 3 个训练种子。
- **真机评估**（计划 §4.2）：约 30 件未见软包（满装约 18、松装约 12），评估前列出并封存；每件每条件 10 次；指标为成功率、掉落率、滑移、过度挤压率、节拍。
- **统计**：每个条件的成功率给 Wilson 95% 区间 [Wilson27]；条件之间用按物品配对的成功率差，区间用物品级（物品内重复成组）自助法 [Efron93]；次数、种子在实验前写定（I-18 的处理方式待总体判断会话确认）。
- **仿真失败的报告**：总尝试数、有效数、仿真失败数，以及把仿真失败计为失败的成功率下限。
- **H4**：评估物品按 R1–R3 测响应，只用于分析，计算其到训练分布的覆盖程度与成功率的相关性。
- **开发集与测试集分开**（I-19）：任务和范围只用仿真物体和开发用真实物品调。

## VI. Preliminary Results

只写正式运行的结果。每条注明出处；版本不同的结果不混用（第 1、2 轮的桁架实现已被替换，其数字不用于描述 v1）。

| 证据 | 结果 | 出处 | 写进论文的方式 |
|---|---|---|---|
| 内容物 v1：随机稳定 | 新种子 200 组（硬物/易弯物/易压物 68/72/60），静置 + R3，确认仿真失败 1/200；另有 5 例物理掉落、45 例搬运碰桌，均保留在分母 | `../test/content_model/REPORT.md`“冻结 v1 补充验收 / 正式结果与 v1 标签”“稳定性、失败证据和入口修复历史” | 支撑“随机抽到的物体能稳定仿真”；同时说明数值稳定率不等于任务成功率 |
| 内容物 v1：响应命中 | 6 个软物配置 × 3 项，15/18 误差 ≤ 20%；未命中 3 项（1 项表外，易压物两种分段的下垂 +139.0%、−31.6%）；表外目标拒绝外推 | 同上“正式结果与 v1 标签”“求解器差异、映射和未命中项” | 支撑“参数可以按响应控制”，并如实写出下垂控制的限制 |
| 内容物 v1：侧夹任务 | 易弯物锚点中心横夹 20 N，10/10 成功；易压物锚点 10 N 中心横夹 10/10 夹取失败（质心只抬升 10.623 mm） | 同上“正式结果与 v1 标签”“任务执行、速度和边界” | 说明不同类别在同一动作下结果不同；易压物侧夹失败也是协议 v2 改顶面抓的动机之一（需与总体判断会话确认这样写是否准确） |
| 内容物 v1：成本 | CPU 单环境 evaluate 中位：硬物 0.695 s、易弯物 6.791 s、易压物 7.880 s | 同上“任务执行、速度和边界” | 训练成本的依据；MJWarp 批量后端未验收，不写吞吐 |
| 松装袋：袋子必要性 | Newton 牛皮纸袋松装 vs 无袋硬盒（各 1 次，定性）：R2 盒子沿袋滑动约 82 mm vs 0.1 mm；R3 下沉 30 mm vs 0.6 mm | `../test/bag_model/REPORT.md`“第 1 轮结果 / C1” | H3 的初步证据（只是单次定性对照，正文要写明） |
| 松装袋：便宜表示 | flex 薄膜行为与参考一致（夹中心能夹起、袋内滑动、夹空袋端失败），但 28 进程只有 0.016 次 evaluate/s；薄板片更快但行为不对，已放弃 | 同上“C2”“C3”“与判断标准对照” | 作为现状和风险如实写；提速结果待第 2 轮 |
| 满装能否并入外皮 | 第 1 轮未回答（满装盒子不贴合，且 R3 穿透失败） | 同上“C1” | TODO：F4 |
| 形状 v1.1、顶面抓 v1.2 | 未运行 | 内容物报告“v1.1 验收”“v1.2 计划” | TODO |
| 策略训练与交叉评估 Q2/Q3 | 未运行 | `../test/policy/REPORT.md` 第 2 轮计划 | TODO：这是 H1、RJ 对照在仿真中的第一份证据 |
| 真机小规模试验 Q4 | 未运行 | 同上 | TODO；赶不上 12/04 就写成第二阶段计划 |

**图 2 的候选**（按数据到位情况二选一）：
- (a) 响应空间覆盖：R3 采样与 RJ 采样各抽 N 个物体，用同一测法测 squeeze_y 和 droop，按类别着色画散点（对数轴）。回答“直接随机关节参数会把样本集中到哪里、有效样本率多少”，直接服务 RJ 对照。需内容物线提供数据（I-30）。
- (b) 若 Q3 结果赶得上：R1、RJ、R3 三个策略在三类物体上的成功率与 Wilson 区间。

## VII. Expected Outcomes and Limitations

按假设逐条写成功与失败各能得到什么（RSS 接受有分析的负面结果）：
- H1 不成立：刚体随机化对这类任务已足够；报告在哪类物体上没有差别，柔性建模的价值需要另找任务。
- RJ 与 R3 无差别：有柔性就够，按响应定义参数不带来额外好处；论文主张相应收窄。
- H2 部分不成立：可删去对应维度，模板更简单。
- H3 不成立（松装上外皮与袋子表示无差别）：袋子表示可以省去，训练更便宜；若松装表示始终达不到训练速度，H3 只在真机上测试，作为限制报告。
- H4 不成立：迁移效果不主要由这几项响应决定，需要找遗漏的因素（如外形、表面纹理、多件物品）。
- 限制：所有参数未经实物标定；强弯曲面外失稳未修复（内容物报告 I-24 正式结果“未关闭项”）；外部围压按整个物体计算；单件、桌面场景。

## 8. 文献（第一批，未完成核对）

按用户 11-05 的安排，文献调研暂停、先完成大纲。已做的部分：
- 已用 arXiv API 核对 36 个编号的题名、作者、首发日期和 arXiv 注明的发表处，全部与题名对上；用 Crossref 核对了 Rucker22、Goury18、Yin21、MuJoCo（Todorov12）、Wilson27、Sanchez18、SoMoGym、SoMo、Schaff23、Manyar25、Manyar24、Efron93 的 DOI 与卷期页。
- 读了原文相关段落、确认要引用的说法：XHugWBC（模板机器人 + 伪惯性 Cholesky 参数化，7 台真机零样本，§3.1、§4）、Stress-guided（Genesis MPM，随机化摩擦、杨氏模量、泊松比，平行夹爪抓豆腐，零样本，§IV-D、§V-A）、Lach（MuJoCo 中用接触柔度表示物体软度并随机化，平行夹爪力控零样本）、Coinbot（MuJoCo composite 袋子，质心移动导致掉落）、SimWeaver（带塑料袋任务的 RGB 零样本）。
- 发现：计划 §5 把 Schaff 等记为“TTIC 2023”，Crossref 显示为 *Autonomous Robots* 47(8):1195–1211, 2023（写入 I-30）。
- DBLP 有反爬验证，未使用；OpenAlex 限流，发表处核对未完成。
- 下一步：补完发表处核对，选 30–35 条写 `references.bib` 和 `reference_check.md`。

## 9. 需要总体判断会话确认（已写入 `../log/ideas.md` I-30）

1. 计划 §4.3 第 1 条仍是第 1 轮桁架实现的数字（1700 倍、14/18、6.5%），与冻结 v1 不是同一模型。稿件将改用 v1 的数字（15/18、1/200）。
2. 计划 §3.3 写“在 MJWarp 中批量仿真”，但 MJWarp 后端未验收，策略线计划在 CPU 上训练。稿件写 CPU MuJoCo，除非另有决定。
3. 计划 §3.2 的随机范围“尺寸不超过最大开口”与协议 v2（顶面抓、软包宽 160–500 mm）矛盾，需要说明顶面抓下的尺寸范围依据。
4. 统计方法：计划写 Wilson 区间 + 按物品配对比较；I-18 建议直接给配对成功率差的区间。稿件拟两者都写（每条件 Wilson，条件间物品级自助法），请确认。
5. Schaff 等的发表处更正（见第 8 节）。
6. 易压物 v1 侧夹 10/10 夹取失败，能否在论文中写成改顶面抓的动机之一。

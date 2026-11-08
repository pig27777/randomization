# 内容物 Warp / MJWarp 检查

最近检查：2026-11-06 18:50（Asia/Taipei）；本轮正式仿真于18:44:13结束，此后仅处理已存证据和静态检查。

**最新结论：硬物R1y下限按审阅仅记录；易弯R1y/R1z/R2通过，R3滑移/摆幅/力峰峰差9.619%/2.316%/23.114%超2%，立即停止。已修复GPU接触编号读回缺陷；修复后的R3是响应超线，两端均无确认发散。当前候选未满足训练标准，GPU E4/E5、硬物任务力峰及S6未运行。** S2维持通过（CPU类别E4 100%、E5前20κ 95%），CPU3倍与单独v1.2.1发布维持取消；冻结归档及提取运行副本不变。

以下记录保留历史；轨迹逐位/1e-9标准已作废，单测量网格差异与硬物刚度下限已按审阅处理，最新检查见末节。

## 范围和执行

环境：`/home/ubuntu/Dev/Envs/mjwarp/bin/python`，MuJoCo/MJWarp 3.17.0、Warp 1.19.0、RTX5090；没有安装或升级依赖。以下命令均以项目根为工作目录、使用共享锁；`PY`表示上述Python绝对路径。

| 检查 | 命令（省略共同前缀`flock /tmp/claude-gpu.lock "$PY" -B -u`） | 结果 |
|---|---|---|
| 隐式tendon阻尼先行风险 | `contents/segment/validate_acceleration.py risk --run-id risk_formal_checked` | 退出0；三类各750步/0.3s短静置，原失败判据0次；CPU反馈求连接力，不是完整GPU后端 |
| S1六次evaluate剖析 | `contents/segment/validate_acceleration.py profile --run-id s1_profile` | 退出0；三类各一次侧夹/顶面抓，排他计时与cProfile原始文件齐全 |
| 相同状态连接力预检查 | `contents/segment/validate_acceleration.py force-pilot --run-id s2_force_pilot_compile` | 退出0；两软类共62个状态，最大力差8.327e-17，压密度差≤1e-9；仅局部预检查 |
| S2原流程回归 | `contents/segment/validate_acceleration.py force-regression --run-id s2_bushing_regression` | **退出2**；计划65例，已执行6例：5个硬物案例逐位相同，首个易弯R1y不等价，余59例未运行 |
| 已存证据审计 | `contents/segment/results/cpu_equivalence_v1_2_1/analyze.py`（同锁、同Python） | 退出0；只读取已有记录，无新仿真；冻结包SHA及原31文件SHA不变，侧夹外部只读依赖SHA一致 |
| 交付静态检查 | 对新增3个Python文件执行`ast.parse`，检查本报告本地链接及候选源码SHA | 通过：语法3/3、失效链接0、候选指纹与运行记录相同；不新增物理实验 |

## 关键数据

- 软物侧夹MuJoCo占50.66%/58.18%，顶面抓35.23%/40.03%；连接力占23.15%–27.24%、压密切换7.40%–10.13%、顶面捏取效果4.11%/4.37%。这些是带剖析开销的单次占比，不是速度验收。
- S1全机忙碌3.681%，平均1.178个逻辑CPU；load1/5/15由0.576/0.621/0.519至0.836/0.686/0.547。S2忙碌3.677%、约1.177个逻辑CPU，load由0.083/0.350/0.435至0.288/0.383/0.445。未停止策略训练；未固定单核，也未开展S3受控中位测速。
- S2易弯R1y刚度：冻结0.5011508539825948，候选0.49658196416970884 N/mm；绝对差0.004568889812886001 >1e-9。轨迹12493/12483步；公共前缀第2步非零差3.587e-21，第18步超过容差，qpos最大差0.005733、qvel最大差3.731323。两次均无确认仿真失败，失败项目为数值等价。
- v1.2归档SHA256：`e75f2a8cc718bcde7d99b6ea34afa170e712784e8960482eec62937232e23919`，与冻结标签相同。未改步长、求解设置、模型参数、阈值、协议或采样范围。

## 必要失败与修复记录

- 风险入口首试因验证器为MuJoCo3.17的12维eq_data额外补列而异常；修正数组形状后才完成三类正式短静置。原日志`risk_initial_shape_error.log`保留，首试不是物理结论。
- 连接力原型先遇到两处Warp编译错误（数组元素多目标赋值、索引引用局部赋值），修正后完成62状态局部预检查；`s2_force_pilot.log`、`s2_force_pilot_checked.log`保留。
- 正式S2失败后没有改原型重跑、放宽标准或继续S3–S6。原34项单元未重跑；本轮没有修改原运行入口，不声称本轮重新通过全部旧单元或完整回归集。

## 限制与证据

压密weld及捏取weld/承载上限/等效弹簧的Warp迁移尚未开始。局部力近似相同不能证明耦合轨迹等价；具体舍入算子与原生接触/求解分支的因果尚未分离。短静置结果不能证明受夹持工况不发散，也不能替代2s静置、S4–S5或完整协议v2.2。GPU训练可用性未建立，evaluate/s和显存吞吐未测；CPU≥3倍未建立。

- [主题报告“CPU 等价提速”](../test/content_model/REPORT.md#cpu-等价提速)
- [汇总审计与源码指纹](../contents/segment/results/cpu_equivalence_v1_2_1/analysis.json)
- [S1原始汇总](../contents/segment/results/cpu_equivalence_v1_2_1/s1_profile/summary.json)
- [S2原始汇总](../contents/segment/results/cpu_equivalence_v1_2_1/s2_bushing_regression/summary.json)
- [首个失败逐字段比较及邻接的两份轨迹](../contents/segment/results/cpu_equivalence_v1_2_1/s2_bushing_regression/case_0005/comparison.json)

全部结果及详细日志统一位于`contents/segment/results/cpu_equivalence_v1_2_1/`。

## 修订标准续轮（2026-11-06）

本轮结果与日志在`contents/segment/results/warp_backend_v1_2/`；标准来自主题报告“S1/S2审阅”：算子绝对差≤1e-12，R1–R4主要响应相对差≤1%，E4/E5类别一致率≥95%。原判法、步长、求解设置及任何参数未改。

| 检查 | 命令（同上共享锁和Python） | 结果 |
|---|---|---|
| 入口小试 | `contents/segment/validate_warp.py smoke --run-id s2_smoke` | 退出0；298项，检查kernel入口和适配器 |
| 算子初验 | `contents/segment/validate_warp.py operators --run-id s2_operators` | 退出0；418项。随后补足5mm初始指尖间隙下的压入深度与接触门槛边界，未改算子或阈值 |
| 最终算子验证 | `contents/segment/validate_warp.py operators --run-id s2_operators_boundaries` | **退出0，436/436通过**，最大绝对差1.7763568394002505e-15，离散输出全部相同 |
| 响应回归 | `contents/segment/validate_warp.py responses --run-id s2_responses` | **退出2**，已执行7/15项：前6项通过，第7项易弯R1z顶住压缩比超1%；余8项未运行 |
| 离线审计 | `contents/segment/results/warp_backend_v1_2/analyze.py` | 退出0，不推进仿真；原31文件/冻结包SHA不变、候选4文件与运行指纹相同、5份源文件语法通过 |
| 交付链接与指纹复核 | 检查本报告的本地文件链接，核对4份候选/验证文件的正式运行SHA | 通过；失效文件链接0，源码仍与已运行版本相同 |

新增Warp实现覆盖围压/滤波/压密weld迟滞和激活位姿、顶面资格/候选段、捏取weld激活/解除/反力/当前承载上限和20ms超载、40N/mm等效弹簧；CPU显式适配器启用这些kernel，原默认入口不变。算子验证包括连接力62状态、压密实际状态62及阈值位姿16、顶面资格42、捏取位姿/反力/超载/解除218、弹簧36项。

响应结果：硬物R1y/R1z/R2/R3/R4主要输出相同；易弯R1y刚度差0.91168%、顶住压缩比相同，通过。易弯R1z刚度0.301981963964→0.302177456450 N/mm，差0.06474%；**顶住压缩比0.1525→0.1550，差1.63934%，未通过**。两次均未确认仿真失败，离散状态一致。

原`metrics.r1_metrics`使用0.0025应变网格判定首次切线刚度>5倍初始刚度；本例恰差一格，而1%允许绝对差仅0.001525。没有改网格或标准、重跑挑选结果、或用S5的2%替代S2的1%。这是测量输出的单格偏移证据，不足以定位具体接触/求解分支的因果；原始结果只有降采样曲线，不伪称保留完整力曲线。

E4/E5类别对照、MJWarp批量后端与S4–S6未运行，不能给95%一致率、GPU训练可用性、evaluate/s或显存结论。原34项单元未额外重跑，不声称本轮重新通过全部旧测试。CPU速度与单独发版目标按审阅取消，策略线继续使用冻结v1.2。

环境未安装新依赖；响应轮load1/5/15为0.196/0.215/0.177→0.693/0.328/0.216，平均忙碌约1.263个逻辑CPU；算子短窗口约2.057个。均持锁，无测速结论。

[本轮审计](../contents/segment/results/warp_backend_v1_2/audit.json) · [算子结果](../contents/segment/results/warp_backend_v1_2/s2_operators_boundaries/summary.json) · [响应结果](../contents/segment/results/warp_backend_v1_2/s2_responses/summary.json) · [首个失败](../contents/segment/results/warp_backend_v1_2/s2_responses/case_0006/comparison.json)。

## 测量分辨率审阅后续轮（2026-11-06）

执行新规则：离散读数的允许绝对差为`max(原相对容差×|冻结值|, 一个实际网格)`，连续量不变。所有正式入口仍用`flock /tmp/claude-gpu.lock`与mjwarp环境Python；CPU配对/本轮GPU入口均设置`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。本轮没有改物理参数、步长、求解设置或协议，没有安装依赖。

| 检查 | 验证入口与参数 | 结果 |
|---|---|---|
| 已存数值响应重判 | `validate_warp.py reassess --run-id resolution_review` | 退出0；7项重判通过，R1z单格差标注按分辨率通过。无新增仿真；按审阅接受既有数值结论，原计划另8项未伪称已执行 |
| 类别入口小试 | `validate_warp.py classes-smoke --run-id classes_smoke --workers 3` | 退出0；三类各1对，3/3类别相同 |
| S2正式类别对照 | `validate_warp.py classes --run-id s2_classes --workers 8` | **退出0，150对完整**；E4 90/90=100%，E5 57/60=95%，异常0；E5两侧各60次2s静置均无失败 |
| 真GPU步进入口 | `validate_warp.py gpu-rest --run-id gpu_rest_smoke --seconds .02 --nworld 2` | 退出0；三类双环境短静置通过，不作正式稳定/速度结论 |
| S4真GPU静置 | `validate_warp.py gpu-rest --run-id s4_gpu_rest --seconds 2 --nworld 2` | **退出0**；三类各5000步、CPU/2个GPU环境均无原判据失败；保留所有阻尼 |
| 受载/S5数值前置检查 | `validate_warp.py gpu-responses --run-id s4_gpu_responses --anchors hard bendable compressible --tests r1y r1z r2 r3` | **退出2**；首个硬物R1y刚度差15.2636%>2%，停止余11项，未进入R4/完整抓取协议/后续S5/S6 |
| 已存证据与来源审计 | `results/warp_backend_v1_2/audit_resolution.py` | 同锁，退出0，无步进；六阶段源码归档逐文件SHA正确，当前5个源文件与最后运行指纹相同，语法通过；原31文件/冻结包SHA不变，硬物R1y与冻结XML完全相同 |
| 交付静态检查 | 复核5个源文件语法/运行SHA，以及本检查报告本地文件链接 | 通过；失效文件链接0；未新增物理实验 |

表中入口均位于`contents/segment/`，结果统一在`contents/segment/results/warp_backend_v1_2/`各run-id，日志同名`.log`。没有重复已通过算子层的436项仿真。

**E5全部不一致**：本轮索引129/130/131，对应原κ索引13、seed20261139/40/41，均为易压。冻结夹取失败且未绑定，候选绑定第33段后在lift过度挤压；滤波夹持力峰14.493327→15.312891N，15N阈值两侧余量+0.506673/-0.312891N。候选记录的承载余量仍+5.212N、无overload事件，不能称为承载上限临界导致。差异在资格/接触及过度挤压分支附近，但具体触发因果未分离；未剔除这些样本。事件早于随机搬运，因此三个seed重现相同前段，不是三个独立κ。

**GPU超线**：硬物R1y CPU161.96591339936433、GPU186.6877670586538 N/mm，绝对差24.721854，允许3.239318；刚度连续计算，分辨率规则不适用。两侧均无确认仿真失败、均50N封顶且刚度为下限。硬物模型无连接力/tendon/weld（ntendon=0、neq=0），原XML相同，故不能把差异归咎于自定义算子或旧阻尼风险。已有降采样曲线的首次接触力8.04903/1.98695N、接触参考宽度差约2.02μm，仅作为接触起点/下限测量敏感的线索，未证明具体数值原因。原线搜索20次到顶提示完整保留，未改参数或重跑挑选。

GPU静置由原生MJWarp和同一双精度连接力/压密kernel在设备上执行，已移除CPU求力桥接；受载诊断保留冻结CPU控制器和逐步读回。当前只支持同模型批量步进，未完成协议v2.2全流程、异κ批量随机化、weld不同激活模式的环境隔离验收。**当前候选不能接入本项目GPU训练；没有S6 evaluate/s、峰值显存或CPU28进程对照数据。** 不把诊断RTF当吞吐。

负载：S2配对平均约7.808个逻辑CPU；GPU静置/受载约1.152/1.169个。受载load1/5/15为0.125/1.247/1.213→0.399/1.251/1.214。正式运行全程持共享锁，未干预其他线；本轮无性能结论。

[本轮总审计](../contents/segment/results/warp_backend_v1_2/audit_resolution.json) · [S2类别结果](../contents/segment/results/warp_backend_v1_2/s2_classes/summary.json) · [GPU静置](../contents/segment/results/warp_backend_v1_2/s4_gpu_rest/summary.json) · [GPU首个失败](../contents/segment/results/warp_backend_v1_2/s4_gpu_responses/hard_r1y/comparison.json)。

## GPU首项审阅后续轮（2026-11-06）

范围：按审阅排除硬物封顶刚度下限，软物R1–R4主要响应≤2%；离散量仍按测量分辨率规则。正式入口均使用共享锁、mjwarp环境Python，并限制数学库线程数为1。未修改参数、测法、协议或阈值。

| 检查 | 命令后缀（入口为`contents/segment/validate_warp.py`） | 结果 |
|---|---|---|
| 硬物旧数据重判 | `gpu-first-review --run-id hard_lower_bound_review` | 退出0；仅记录下限，无新增仿真，不代表硬物任务层通过 |
| 软物首轮 | `gpu-responses --run-id s5_soft_responses --anchors bendable compressible --tests r1y r1z r2 r3` | 退出2；易弯R1y/R1z通过；R2接触反力为零/滑移缺失，定位为读回缺陷，未认定物理失败 |
| 原生读回复现与修复 | `gpu-readback --run-id s5_readback_repair` | 退出0；同一接触现代编号`[0,1]`、旧字段`[0,0]`，同步后相同 |
| 修复后受影响响应 | `gpu-responses --run-id s5_bendable_readback_fixed --anchors bendable --tests r2 r3` | **退出2**；R2通过，R3三个连续幅值超2%，停止余下所有仿真 |
| 来源与证据审计 | `contents/segment/results/warp_backend_v1_2/audit_gpu_review.py`（同锁，无步进） | 退出0；四阶段归档SHA匹配、5份当前源码语法/SHA通过，修复仅涉及接触编号读回；冻结归档/提取副本31文件/工作区原内容物文件不变 |

修复位置是新增`mjwarp_backend.DiagnosticSim`适配器：MJWarp3.17的`get_data_into`填写`contact.geom`，冻结测量函数读取独立的`geom1/geom2`，现读回后同步。旧异常R2完整保留为无效测量证据；修复后R2承重比0.999137，离散状态与CPU一致。R1使用现代编号，无积分或材料改动，未重复已通过R1。没有删除断言、放宽阈值或选择较好的重跑结果。审计首试的函数间空行边界比较错误已修正，严格SHA核对保留。

有效通过：易弯R1y刚度差0.5672%、R1z差0.08259%、R2下垂差0.1651%，顶住压缩比相同。**R3首次真实超线**：

- 滑移0.738057633→0.667063190mm，绝对差0.070994443mm，**9.6191%**；允许0.014761153mm。
- 摆幅0.143136203→0.146450549mm，绝对差0.003314346mm，**2.3155%**；允许0.002862724mm。
- 力峰峰0.162323024→0.124803474N，绝对差0.037519549N，**23.1141%**；允许0.003246460N。
- 衰减时间0.9920→0.9908s通过，两端均观察窗未完成；`fail=None/dropped=false/table_touch=false`。连续幅值不适用网格豁免。

负载：首轮平均约1.2033个逻辑CPU、修复后约1.1793个，修复后load1/5/15为0.153/0.417/0.597→1.277/1.025/0.826。全部正式运行持共享锁；这些是验证负载，不作速度结论。

限制：只完成4个有效软物响应配置（3通过、1失败）；易弯R4、易压5项、完整GPU v2.2执行器/焊接隔离、GPU E4的30组合/E5前50κ、硬物类别及夹持力峰中位数均未验收。**没有64/256/1024环境evaluate/s、峰值显存或CPU28进程对照**，不得引用诊断RTF代替。当前候选不满足GPU训练接入标准，具体数值差异来源尚未分离，不能称为旧阻尼发散。

工作区`test/policy/README.md`与冻结标签不同，非本轮改动，未回滚；提取运行副本31文件仍逐项一致。未重跑原34项单元或已通过436项算子测试，不宣称新增覆盖。HANDOFF/progress及主题报告已同步。

交付静态检查：6份Python语法、5份当前运行源码SHA、42个本地文件链接均通过，失效链接0；没有新增物理实验。[检查记录](../contents/segment/results/warp_backend_v1_2/s5_bendable_readback_fixed/delivery_checks.json)。

[本轮结果](../test/content_model/REPORT.md#gpu首项审阅后结果2026-11-06-1844软物r3超线停止) · [R3逐字段证据](../contents/segment/results/warp_backend_v1_2/s5_bendable_readback_fixed/bendable_r3/comparison.json) · [读回复现](../contents/segment/results/warp_backend_v1_2/s5_readback_repair/summary.json) · [审计](../contents/segment/results/warp_backend_v1_2/audit_gpu_review.json)。

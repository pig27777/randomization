# 交接：物流软包的统一物理建模

更新：2026-11-10（v1.3 CPU仍冻结；硬物R3保持力差按审阅只记录，易弯R1y五次正式通过。GPU G3已有E4复核为85/90=94.44%<95%，按要求停止，G4/G5未进入，尚不能供GPU训练。）

**2026-11-09 统一YAM平台补充（ubun）**：冻结v1.3已通过独立适配器测试原14条件，4任务success、5抓取失败、5初始化失败；9次动力学无模型失效/警告，四个任务success的跟踪峰值约42–43mm，不能认作完整物理验收。两类0°/45°的系数1可完成任务，系数0对照均提起失夹。模型本构/冻结包不改；另有整体响应κ的CPU交互预览（材料边界实验，非抓取验收）。[平台报告](../test/flex3d/REPORT.md#分段内容物-v13-接入本轮计划2026-11-09ubun) · [κ界面入口](../test/content_model/TOOLS.md)。

## 1. 目标

用一套物理语言，通过换参数生成不同的物流软包（A = X · B）：

- **B**：固定的物理结构和力学规律，所有物体共用
- **X**：每个物体的参数（形状 + 行为量）
- **A**：某一个具体物体的模型

具体形式参照 XHugWBC（arXiv 2602.05791）：结构固定的模板、只含物理量的参数向量 κ（即 X），以及重参数化 θ ↔ κ。规格见 `package_params.md`：κ_content、约束采样与分段生成已部分实现，完整 θ ↔ κ 和袋子参数化仍未实现；参数满足代数约束不代表动态稳定，后者须单独验收。

建模原则是**按夹爪和袋子感受到的效果建模**，不追求外观：压缩曲线、迟滞、残余、蠕变、局部性、质心移动、提起时的下垂。

最终用途（2026-11-03 明确）：物流分拣中软包的内容物种类很多，软的硬的都有。目标是对内容物做大规模随机化，让策略在仿真里学习，在真实场景中更鲁棒。因此判断标准是：随机抽到的每个 κ 都能稳定仿真、行为覆盖真实范围、单卡批量吞吐足够；单个物体的精确标定是次要的。内容物不需要精细建模，物理上合理即可。

## 2. 一句话现状

**2026-11-09 大软枕头与四面体视图**：用户明确240×160×50mm/0.29kg，恢复大尺寸并降到E5kPa，采用189节点/576四面体新模板。三例保持原YAM/GPU Newton/pinch0：StVK/ν.3在4.371s翻转，ν.45在3.312s塑性非有限；实验性原生SNH/ν.3跑到7.939s，无数值几何失效、overflow或warning，Jmin.4328、穿透峰.354mm，但提起失夹，尚无大软枕头成功。SNH只能mjSpec编译，完整模型存model.mjb，回放优先读MJB；网格/本构变化不能沿用旧B验收。用户要求四面体视图，已隐藏两相机外表面、保留体积网格和机械臂。按计划停止三例探索，不加硬/缩小兜底，不额外QA。结果见[连续体报告](../test/content_model/CONTINUUM_REPORT.md#大软枕头抓取本轮计划2026-11-09ubun运行前)，数据`continuum-pillow-grasp-r1/`。

**2026-11-09 连续体首个原生接触成功例**：同B/GPU Newton、辅助捏取0，120×45×50mm、40.5g、E100kPa/ν.1/yield.2的窄实体15.026s完整提起搬运放下松开。无倒置/overflow/warning，Jmin.7596、穿透峰.8828mm，跟踪峰40.60mm；本次plastic=0，是弹性响应实例，不能外推到大软物或随机κ成功率。前一120×50mm/E10kPa样本翻转失败保留，网页有标红慢放；[本轮结果](../test/content_model/CONTINUUM_REPORT.md#成功例及停止探索)。

**2026-11-09 分段/连续体GPU成本与视频补充**：同YAM、尺寸/质量的单world推进1s仿真，当前分段/连续体hard为0.900/4.998s、易弯8.651/3.335s、易压13.005/3.052s；分段六个1/32world组均有线搜索上限警告，完整诊断保留valid=false，不作有效吞吐或等响应加速结论。用户指定agent完成4单片+1原生接触并排，均为旧轨迹CPU回放、事件醒目标注；[对照与视频记录](../test/content_model/CONTINUUM_REPORT.md#分段物gpu速度对照本轮计划2026-11-09ubun运行前) · [视频检查](../reports/contents-model-videos-test.md)。

**2026-11-09 连续体候选（ubun，用户要求实施）**：新`continuum/`采用MuJoCo/MJWarp4原生3D StVK/J₂、GPU Newton，固定60节点/144四面体B；X含真实圆润/鼓包/渐变体积形状与材料，不加压密。三材料×四形状构建、四形状短入口及调参抓取UI闭环已完成。完整三组抓取为硬盒提起失夹、两软形状塑性非有限；当前复测另有残差门槛失败，不能进入训练。纯GPU接触静置24κ×1/32world、各有效组3次×1s：19/24与13/24有效；有效κ中位588.02 step/s与9162.14 world-step/s，配对合计增益中位22.106倍，失败组不计速度。UI单次上限10s仿真；不是纯GPU闭环抓取控制器。独立`.venv-continuum`不升级旧环境；[模型/UI说明](continuum/README.md) · [主题报告](../test/content_model/CONTINUUM_REPORT.md) · [检查报告](../reports/content-continuum-test.md)。

**使用判断（2026-11-05 Newton 补充验收）**：内容物 **v1 已冻结**，本地标签 `segment/results/freeze_v1_newton/release_v1/v1.json`、源码包同目录 `v1.tar.gz`（无 Git 元数据）。Newton 上限 100、线搜索 20、容差 1e-8，步长 0.4 ms；M3 **15/18**，新种子 20261107 确认失败 **1/200**，I-22 **10/10**，CPU 三类中位 **0.695/6.791/7.880 s**，24 项单元检查通过。默认表是 Newton `mapping_pass1/table.json`。错误入口 20261106 批的 5/200 失败已保留；漏传原自动固定上限的问题修复，原五例复核 0/5 失败。v1.1按审阅暂停；旧v1.2局部表层已归档。新顶面捏取效果模型v1.2已于2026-11-06冻结：34项单元、R4与≤5mm静态跟随通过，E4按审阅修订标准核对通过，E5新种子0/200，E6三类速度比0.405/1.314/1.486，策略/Isaac Lab线可固定v1.2按协议v2.2接入/训练（旧v2.1失败历史保留）。I-26 不宣称关闭；I-29 硬物任务力峰 >50 N 交策略线，不能把执行器返回 success 理解为真实夹爪合规。详见内容物报告“冻结 v1 补充验收”。

**分段内容物 v1.3 已冻结；GPU训练平台指定MJWarp自建后端，当前候选尚未通过。** MuJoCo3.17 CPU v1.3保留为参考；Isaac Lab不再作为内容物平台候选。v1为长方体刚体段+弹性关节，关节锁死表示硬物；参数按整体响应定义。v1.3保留本体及等效夹紧弹簧，顶面捏取使用捏点处connect连接及5ms合力低通超载解除，不加表层/形状。设计见 [segment_model.md](segment_model.md)，参数见 [package_params.md](package_params.md)，测法与逐轮结果见 [内容物报告](../test/content_model/REPORT.md)。

**冻结版本边界（2026-11-06，v1.2/协议v2.2）**：标签与源码包在 `segment/results/pinch_effect_v1_2/v22_review/release_v1_2/v1.2.json`、同目录 `v1.2.tar.gz`，包SHA256为 `e75f2a8cc718bcde7d99b6ea34afa170e712784e8960482eec62937232e23919`（无Git元数据，沿用v1归档/SHA标签方式）。本构/换算表不改；任一指接触起20N下压力限，总夹持力为真实横向力平均+弹簧反力，效果weld为5ms/imp.99/torquescale1。34项单元、R4±10%、三维跟随最大0.540/1.569mm通过；E4按已授权修订标准通过：成功总数不减、未解开且挺过提起保持集合包含（48→48→84、51→84→102），全任务配对反例6/216≤5%，默认成功56/108和60/108、硬物36/36拒绝。E5新种子20261126确认失败0/200；E6三类各10次CPU中位0.281/8.920/11.708s（侧夹比0.405/1.314/1.486），全部通过。易压悬挂变化最大56.130mm<半厚60mm。形状暂停、表层不继续；旧v2.1失败/偏软跟随及旧116探索失败均保留，不宣称全域稳定。策略/Isaac Lab线可固定v1.2接入/训练，原v1包/表不改；新结果/审计见 `v22_review/`。

**Warp/MJWarp本轮结论（2026-11-06）**：S2算子436项最大差1.776e-15≤1e-12，已存7项数值响应按分辨率审阅接受，原另8项未补跑且未声称已执行；CPU类别E4 90/90=100%、E5前20κ 57/60=95%，S2通过。真GPU步进三类各2环境2s静置已通过。硬物R1y的161.9659→186.6878N/mm按最新审阅只记封顶刚度下限，不再判失败；任务层类别与力峰仍待验。新增适配器已修复MJWarp只更新新版接触编号、冻结测量读旧字段的问题，原异常R2保留。有效易弯R1y/R1z/R2差0.5672%/0.08259%/0.1651%通过，**R3滑移/摆幅/力峰峰差9.619%/2.316%/23.114%超2%，已立即停止**；绝对差0.071mm/0.00331mm/0.0375N，连续量不适用网格豁免。两端均无确认发散、掉落或碰桌，具体数值来源未分离，不能称为旧tendon阻尼发散。**当前MJWarp候选未满足训练接入标准**；易压响应、R4、完整v2.2/异κ批量/焊接隔离、GPU E4/E5与硬物力峰、S6吞吐/峰值显存及CPU28进程对照均未验，诊断RTF不可当吞吐。冻结归档、提取运行副本31文件和工作区原内容物文件不变；工作区策略README存在非本轮改动，未回滚。策略线继续CPU v1.2，CPU3倍与单独v1.2.1发布取消。新结果 `segment/results/warp_backend_v1_2/`；详见[GPU审阅及续节](../test/content_model/REPORT.md#gpu-首项审阅总体判断会话2026-11-06)、[本轮审计](segment/results/warp_backend_v1_2/audit_gpu_review.json)和[检查报告](../reports/content-model-cpu-equivalence-test.md)。

**用户指定的第 1、2 项曾在 50 μs 版本的规定工况验收通过**：M1 的 42 个刚度模式误差最大 0.067%，有限变形/闭环块的势能梯度与内力矩检查通过；两个软锚点中心/端部夹紧提起 **4/4 通过**，最大接触点运动 3.10 mm，保持期桌面反力为 0，指垫承重误差最大约 0.15%。两锚点静置/能量、硬物夹爪和硬物/易压物搬运入口回归通过。结果为 `segment/results/modeling/verified_50us/`，25 μs 对照在 `verified_25us/`；50/25 μs 同参考点的滑移差 0.000068 mm，通过预定一致检查。此后源码已改变，历史验收不能直接用于当前版本；当前版本的验收见下一段。任务检查见 [segment-modeling-test.md](../reports/segment-modeling-test.md)。

**I-24 历史基线：CPU bushing、0.4 ms，指定修复与回归通过**（`segment/results/modeling/release_repair_400us/`，0.2 ms 对照 `release_repair_200us/`）。连接面弹性势能、原生内外纤维与全向剪切限位、原生隐式 tendon 阻尼保留；压密 weld 按任一端在其他轴上的压密度 ≥0.9 开启、<0.8 释放，另外要求存在外部围压（5 ms 滤波、≤0.05 N 时全部释放），weld 时间常数改为 2 ms，几何限位仍为 1 ms。两个步长 M1 各 10 项通过，含实际双轴压紧释放与开启/卸载能量检查；四项 M2 均通过，0.4 ms 滑移为 1.174/0.060/2.458/4.437 mm，桌面反力 0。原步长一致线全部通过，最大转角差 0.728°、滑移差 0.5961 mm、承重比差 0.003105。两个锚点各连续 2 次完整抓放全部通过，松开后 weld=0、指垫力=0，均能再夹起。静置/能量、采样器与 4 项公共入口通过；29 份正式 JSON 指纹与该轮交付源码一致。该轮未证明所有局部载荷或随机 κ 稳定。详见报告“I-24 正式结果”及 [该轮检查](../reports/content-model-release-repair-test.md)。

当前仍采用上述 I-24 后本构与 0.4 ms；R2/R3 提起后维持弱侧 20 N，材料参考在 lift 前捕获，接触体等载时固定选择，必须使用 `Sim.step()`。当前 Newton 表为 `segment/results/freeze_v1_newton/mapping_pass1/table.json`，M3 15/18，三项未命中保留；表外拒绝外推。易压物 10 N 中心横夹仍 10/10 物理夹取失败。CG 候选和上一轮 9.215 s 未通过的记录在 `segment/results/freeze_v1/`，是历史结果，不能当作当前状态。所有旧表、truss、round2、50 μs、方向 A、I-24 与错误入口批次保留。Newton 最终失败 #57 为 R3 合拢时 y 压缩 1.20353 倍限位；未证明 I-26 因果关系。本轮未做 MJWarp。详见 [Newton 检查](../reports/content-model-newton-v1-test.md)。

2026-11-04 接手原会话未分析的后台结果：**S1、S5、S6 和 S3 均已运行结束**，两段程序退出码均为 0，结果在 `segment/results/round2/`。S1 新随机扫描 **14/200 = 7%**，标准 ≤ 1%；旧 13 个失败参数重跑 12 个未失败、1 个仍失败，但其中 **10 个被新约束排除**。T5 为 **11/18 项命中、5 项超差、2 项提起失败**；易压物 R3 碰桌，指垫承重仅约重量的 1%。S3 软物接触回放出现非有限状态或极端位移，易压物批量 22/1024 个环境非有限。**程序跑完不等于验收通过，接口可调用不等于模型动态稳定。** 完整证据在内容物报告“第 2 轮最终结果与交接”；任务检查见 [segment-round2-test.md](../reports/segment-round2-test.md)。

满装袋子的效果并入等效外皮；`surface_friction`、`skin_compliance` 已加入 κ，但外皮尚未完整验收。松装袋子的表示仍由袋子线对比实验决定（`../test/bag_model/REPORT.md`）；第 2 轮（2026-11-05）只保留 flex 并提速到 28 进程 0.068 次 evaluate/s（目标 0.3）；硬盒行为保持、穿透 0，但软内容物穿透 4–5.6 mm、顶面抓提不起松装袋、满装贴合时袋被压穿，按计划表松装暂不进入训练，待用户决定。松装必须覆盖，末端只用平行夹爪。论文目标与计划见 `../paper/rss2027_plan.md`。策略线仍使用自己的 `policy/content_snapshot/`，它不会自动变成本轮模型；更新快照前须明确模型版本和共享夹爪协议。**所有参数仍是设定值，没有实测对比。**

此前的四面体路线（压密本构、刚体压四面体的接触修复、隔膜压四面体基准）结果保留，但已退出关键路径，见 §6。

## 3. 环境

| 环境 | 路径 | 版本 | 用途 |
|---|---|---|---|
| mjwarp | `/home/ubuntu/Dev/Envs/mjwarp` | MuJoCo 3.17.0、MuJoCo Warp 3.17.0、Warp 1.19.0 | **当前内容物环境**：分段模型 CPU 验证；MJWarp 后端未验收；旧粒子查看器仍可用 |
| mujoco_env | `/home/ubuntu/Dev/Envs/mujoco_env` | MuJoCo 3.14.0 | 早期的硬物模型；`real2sim` 袋子资产只能在 3.14 上运行 |
| newton_latest | `/home/ubuntu/Dev/Envs/newton_latest` | Newton 1.7.0、Warp 1.19.0 | 接触复测与修复（`../test/contact_transfer/REPORT_newton170.md`、`REPORT_repair.md`） |
| newton 1.6.1 | `/home/ubuntu/Dev/Projects/real2sim/previews/newton-official-bag/.venv` | Newton 1.6.1 | 第一轮 Newton 实验 |
| bubble venv | `/home/ubuntu/Dev/Projects/real2sim/bubble_mailer/.venv` | Newton 1.6.1、Warp 1.18（无 MuJoCo） | 袋子线 Newton 后端；接触基准、夹取测试、`newton/` 下的接触诊断 |

硬件：RTX 5090（32 GB），32 核 CPU。桌面显示为 `:1`。

## 4. 文件地图

模型代码按方案和实际版本放在 `contents/`，版本索引见 [README.md](README.md)。独立测试、验收、分析、测速与查看器在 [test/content_model/](../test/content_model/TOOLS.md)，Newton 接触诊断在 `test/contact_transfer/`；新结果按 [日期与对象](../test/runs/README.md)管理。历史冻结包与旧结果保留原路径。

```
contents/
├── HANDOFF.md
├── package_params.md   参数向量 κ 规格（设计）
├── segment_model.md    分段刚体内容物模型（设计，当前主线）
├── segment/      当前主线（模型、采样器、MuJoCo 适配、换算表；第 2 轮结果在 segment/results/round2/）
├── legacy/particle/   粒子键模型（已停止，MuJoCo / MJWarp）
├── newton/     旧 Newton 实验、接触诊断和暂停的压密本构；四面体生成器供代理测试线参考
└── legacy/rigid/      硬物模型和最早的统一结构原型
```

独立脚本的当前路径与运行说明统一见 [测试入口表](../test/content_model/TOOLS.md)；下表的历史文件名用于对应既有轮次。

### `segment/`：当前实现与接入约定

| 入口 | 用途与限制 |
|---|---|
| `model.py` | `Kappa`、三个锚点、`build`、质量/惯量、关节描述、换算表读取。常规 `build(k)` 对新表的表外目标明确报错；历史表的旧行为保留供复现 |
| `sampling.py` | `sample_kappa(rng, category)` 返回合法 κ，`k.category` 带类别元数据；`sample_rj` 返回直接随机内部参数的描述，与 R3 配对几何/质量。类别为 `hard` / `bendable` / `compressible` / `mixed` |
| `bushing.py` / `mj_adapter.py` | 当前默认 CPU bushing、0.4 ms，I-24 与指定夹持回归已通过；原生限位/阻尼，外部围压门控的 2 ms 压密 weld，使用 `scene()` + `Sim.step()`。原 truss 保留 0.5 ms 历史对照 |
| `mj_adapter.object_mjcf` | 仅原生 XML；软物新默认明确报错，避免丢失自定义力。显式 `mode="truss"` 可复现旧版，但不代表新模型已接入策略 |
| `protocols.py` / `metrics.py` | R2/R3 保持 20 N；在 lift 前捕获接触材料点，切向运动为 slip，旧内部点运动另存。R2 用 `suspended` 明确记录承重/离桌 |
| `freeze_v1.py` / `results/freeze_v1_newton/` | M3、200 组、冻结任务 evaluate 的可复现入口；`--output-root` 选择阶段目录，新 run-id 拒绝覆盖。Newton v1 默认 `mapping_pass1/table.json`，15/18，标签 `release_v1/v1.json`；原 CG 结果在 `results/freeze_v1/` |
| `calibrate.py` / `results/table.json` | 旧表保留；单独运行旧标定入口须给新 `--output-dir`，不会再覆盖当前候选表 |
| `run_round2.py` | S1、S5、S6；S6 复用第一轮测试函数，T7 使用 S1 结果 |
| `validate_model.py` / `tests/test_bushing.py` | M1（刚度、梯度、限位、阻尼、weld 开关）、M2 分层夹紧提起、`regressions` 公共入口回归，`--dt` 步长对照；独立结果写 `results/modeling/<run-id>/`，不覆盖 round2。`stability_estimate.py` 复核步长；`view_m2.py` 在 DISPLAY=:1 实时查看 M2，不写结果 |
| `mjwarp_validate.py` | 新 bushing 模式明确拒绝运行（无 GPU 连接力实现）；显式 truss 可复现旧 S3，旧回放未通过 |
| `results/round2/` / `results/logs/` | 本轮 JSON；`round2.log`、`s3_mjwarp.log`、`chain2.log`。本轮已结束，不要重启同一批覆盖产物 |

接入例子（用于开发，不代表已获准正式训练）：

```python
import numpy as np
import mj_adapter as adapter
from model import build, load_table
from sampling import sample_kappa

table = load_table()
kappa = sample_kappa(np.random.default_rng(2026), "compressible", table=table)
desc = build(kappa, omega_max=adapter.OMEGA_MAX, table=table)
sim = adapter.Sim(adapter.scene(desc, (0, 0, kappa.size[2] / 2)))
sim.step()  # 必须由该入口计算连接力；直接 mj_step 不包含 bushing 力
```

`scene` 可接收两指垫定义并生成显式接触对，`Sim` 传入对应 pad_names；默认步长为 `adapter.BUSHING_DT`。原版 XML 接入显式指定 `object_mjcf(..., mode="truss", dt=adapter.DT)`；这种导出仅用于历史对照。表面摩擦与指垫力控制都必须与策略协议一致。新模型正式测试状态见上文，策略快照不改。

I-27：`scene(..., worldbody="", contact="", equality="", tendon="", pad_contact=True, table_contact=True, solver=None)`，外加片段不带外层节标签；`solver="Newton"`（默认）或 `"CG"`。开关只禁内容物直触，外来 tendon 按名称与内容物分开，不再补零 `TendonInfo`。袋子线请删旧 `bag_rope` 补丁；装袋组合仍由袋子线验收。CPU 构建必须保留 `omega_max=adapter.OMEGA_MAX`，否则不是验收配置。

### `legacy/particle/`：粒子键模型（已停止）

| 文件 | 作用 | 状态 |
|---|---|---|
| `particle_soft.py` | 模型本体：节点 + 键 + 接触表皮；叠平（`build_flat`）、揉团（`build_blob`）、枕头（`build_pillow`）三种生成器 | 能运行；揉团和枕头受压会塌 |
| `particle_tests.py` | 测试：静置、悬臂、挤压、手指、夹边提起、从上捏起 | 能运行；从上捏起的动作设计有问题 |
| `calibrate.py` | 闭环标定压缩曲线、蠕变、迟滞的增益 | 三种形状都不收敛；保存的 `gains_*.json` 已发散，**不要直接使用** |
| `view_soft.py` | 交互查看器（见 §9） | 可用；使用未标定的默认参数 |
| `mjwarp_soft.py` | 同一个模型在 MJWarp 上的批量版本，力学规律写成 Warp kernel；`check` 和 `speed` 两种模式 | 已验证与 CPU 一致 |
| `results_particle/` | 测试结果（json、png）和标定记录 | |

### `newton/`：Newton 实验与接触诊断

| 文件 | 作用 |
|---|---|
| `unified_soft.py` | **分层四面体生成器**（表皮、芯层、间隙；叠平、揉团、枕头），保留供代理测试线参考 |
| `compaction_model.md` | 内容物压密本构设计（已搁置） |
| `unified_tests.py` | Newton 上的测试（夹爪受力已扣除驱动阻尼） |
| `newton_soft.py` | Newton 真实布料（折叠、揉团）和四面体对比实验；生成揉团的状态 |
| `render_newton.py` | 用 MuJoCo 渲染 Newton 存下的状态 |
| `rigid_tet_check.py` | 第 0 步初步探索（只看瞬时下陷，结论已作废，仅作记录） |
| `i11_finger_off.py`、`ef_*.py` | 边/面接触抖动的诊断脚本，调用 `../test/contact_transfer/benchmark.py`；说明见 `../test/contact_transfer/REPORT_diagnosis.md` |
| `results_newton/`、`results_unified/` | 结果 |

### `legacy/rigid/`：硬物与早期原型

| 文件 | 作用 | 状态 |
|---|---|---|
| `unified_content.py`、`run_tests.py` | 硬物模型：刚性核心 + 表面单元 + 活动质量（书、衣物、瓶子三组预设） | 书基本可用；瓶子未通过；最终版接触参数下抬升力抖动。压缩单元的做法（串联弹簧 + 限位锁死 + 摩擦迟滞）可供分段模型参考 |
| `render_items.py` | 渲染三组预设 | |
| `results/` | 硬物测试结果（`summary.json` 为三件物品的完整结果） | |

### 相关的外部目录

- `/home/ubuntu/Dev/Projects/real2sim/soft-package-assets/`：PE 袋、牛皮纸袋、纸板文件封、旧实验袋的资产。只能在 **MuJoCo 3.14** 上运行（`runtime.py` 有版本检查，3.15 起移除了它用到的 `internal` 属性）
- `/home/ubuntu/Dev/Projects/real2sim/bubble_mailer/`：气泡信封的参数辨识脚本
- `../test/contact_transfer/`：刚体/膜片压四面体的接触传力基准和修复；`REPORT.md`（1.6.1 正式验收）、`REPORT_position_control.md`、`REPORT_newton170.md`、`REPORT_repair.md`
- `../test/gripper_grasp/`：位移控制夹爪夹取均匀海绵，`REPORT.md`
- `../test/bagged_contact/`：隔膜压四面体基准（已退出关键路径）
- `../test/content_model/`：分段内容物模型的响应定义 `README.md` 和测试 `REPORT.md`
- `../test/bag_model/`：袋子对比实验（满装、松装时袋子的影响，松装袋子的便宜表示），`REPORT.md`
- `../test/policy/`：抓取任务与评估接口定义 `README.md`、策略测试 `REPORT.md`（策略线）
- `../test/proxy_eval/`：代理真实测试集（Newton 高精度物体），`REPORT.md`（接触线）
- `../paper/`：RSS 2027 论文计划
- `../log/`：进度流水、想法池

## 5. 已验证的事实

| 事实 | 数据 |
|---|---|
| 当前分段模型构建与锚点静置 | 第 2 轮 T0 的 8 例构建检查、T1 三个锚点静置/能量通过，不能推广为随机模型稳定 |
| 当前约束采样仍有动态失败 | 新种子 2026：14/200 = 7%，全部为软物；18 项软物响应仅 11 项命中。见内容物报告第 2 轮 |
| 当前分段模型 MJWarp 未通过 | 易压物 R1 回放非有限；软物 R2 出现极端位移；批量易压物 22/1024 个环境非有限。不能引用旧粒子的一致性结论 |
| MuJoCo 里刚体能正确压缩粒子软体 | 2 kg 方块使软块下陷 10.8 mm；同样的力直接加在粒子上是 9.8 mm |
| 旧粒子模型 MJWarp 和 CPU 一致 | GPU 4 个环境下陷 10.78–10.86 mm，CPU 10.82 mm；不适用于当前分段模型 |
| 旧粒子模型 MJWarp 吞吐 | 256 环境 919、512 环境 1045 env-帧/秒（每帧 10 ms）；1024 环境显存不足 |
| CPU 单进程速度 | 粒子模型约 32–37 帧/秒 |
| 粒子模型局部性（叠平） | 压入 2 mm 时，20 mm 手指 5 N，压板 104 N |
| 提起边角 | 揉团承担 47% 重量；枕头承担 38%，远端留在桌上 |
| 现有袋子资产速度 | PE 约 0.06 帧/秒，牛皮纸约 1 帧/秒，纸板约 0.39 帧/秒（单进程，MuJoCo 3.14） |
| Newton 四面体本身承力正确 | 同一网格直接加压：Newton 下陷 4.506 mm，MuJoCo 3.17 为 4.503 mm（`../test/contact_transfer/REPORT.md`） |
| Newton 刚体压四面体的抖动机制 | 全表面接触对每条边/面只取一个“最深点”，平工具压平边/面时该点在两端间跳，力轮流压到不同节点；与迭代、刚度、阻尼无关，1.6.1 和 1.7.0 相同（`../test/contact_transfer/REPORT_diagnosis.md`） |
| 平面压头积分接触修法 | 按覆盖面积积分、刚度按面积加权，压板、节点上/节点间手指均单次通过原验收（需 5–15 s 静置）；仅支持竖直平面工具压顶面（`../test/contact_transfer/REPORT_repair.md`） |
| 位移控制夹爪能夹起硬海绵 | μ=6000 Pa 海绵质心升高 92–95%；μ=1500 Pa 软海绵翻转或滑脱（`../test/gripper_grasp/REPORT.md`） |
| MuJoCo 版本事实 | flex 隐式积分在 3.5.0 加入；3.13.0 起 flex 弹性要求 `integrator="discrete"`；3.15.0 移除了 `internal`；未发布版本将加入 flex `yield` 塑性 |

## 6. 已否定或暂停的方向

| 方向 | 结论 | 原因 |
|---|---|---|
| Newton 四面体软体 + 刚性夹爪 | **搁置（2026-11-03）**：内容物不需要精细建模，改用分段刚体模型。原状态：未否定；平台验收未通过 | 早先“刚体压不动四面体”的结论已推翻：四面体承力正确，增加迭代后平均力也能对，问题是边/面接触取点跳动造成持续抖动和局部力不平衡（只用顶点接触时小工具会从节点间穿过）。平面压头的积分接触修法已通过平压基准，尚未推广到夹爪、膜片和任意工具；压板粗细网格下陷差 6.56%，未收敛。见 §5 所列报告 |
| Newton 真实布料 | 只作行为参考 | 刚体和布料之间可以传力，折叠布料能提起一角；但挤压曲线无法按参数控制 |
| 扁平双层壳 | 放弃 | 只能表示叠平，不能表示揉团 |
| 刚性核心 + 表面单元表示衣物 | 放弃（单段） | 提一角时整件像木板一样翘起。分段刚体模型用多段核心解决这一点 |
| 两两相连的键网络表示填充物 | **停止**，由分段刚体模型取代 | 受压像沙堆一样塌，刚度不随参数变化。模型含抗拉参数，不能简单说“键不抗拉”；抗拉、抗剪支撑偏弱和缺少可靠体积支撑是候选原因，未逐项确认（`../log/ideas.md` I-07） |

## 7. 下一步（v1.3已冻结，GPU候选待验收）

-1. **v1.3 GPU后端停在G3（2026-11-10，ubun）**：候选接口`segment/mjwarp_v13.py::run_specs(specs)`，`specs`为含`kappa`、`grasp`、`seed`的字典列表；验收驱动已移至`test/content_model/validate_tasks_v13.py`。**尚未就绪，袋子线不能将其当作已验收的软内容物后端。** G1通过；硬物R3保持力18.924→7.420N按11/8审阅归于测法Closer增益，只记录，不改冻结版本。易弯R1y补4次后，5次中位误差−0.44738%、每次最大绝对误差3.49685%，G2阻塞解除。原正式G3的90项E4保留后，E5因异κ共享字段构建报错中断；本次先复核E4，硬物30/30、易弯28/30、易压27/30，合计85/90=94.44%未过95%。硬物任务力峰中位差0.00014372%通过，E4无sim_fail/溢出。A0显式共享字段修复及两κ小试只验证局部构建/行为，不构成完整G3/G4通过；新输出位于`../test/runs/2026-11-10/contents/v13-a0/`。下一轮先逐状态隔离两项易弯未绑定、三项易压未超载的分歧，再重过G3；E5、G4/G5及A1–A6未继续，没有训练吞吐结论。I-37/I-38留待v1.4；GPU平台固定MJWarp、CPU v1.3作为参考。[报告](../test/content_model/REPORT.md) · [检查](../reports/content-model-gpu-v1_3-test.md)。

0. **v1.3 已冻结（2026-11-06），各线改用 v1.3。** 标签 `segment/results/pinch_connect_v1_3/release_v1_3/v1.3.json`，归档 SHA256 `940efb2f…415e`。相对 v1.2 只改顶面捏取：单段 weld 改为 connect 点连接，捏点取指尖中点的实际深度，与指尖间带（含 20 mm 指尖宽）相交的顶层段分担载荷；connect 合力经 5 ms 低通，超过 `pinch_coeff × F` 即全部解开（取消“持续 20 ms”）。P1–P7 全部通过：R4 +3.59%/+1.69%；悬挂偏差 ≤0.74 mm、倾角 ≤2°；E4 修订单调性通过，反例 0.93%；E5 新种子 1/200；E6 1.16/1.38 倍；位置扫描趋势合理。与 v1.2 相比：悬挂倾角中位 11.5°/30.0° → 0.25°/0.16°；急停摆幅中位 0.99/10.3 → 4.4/25.5 mm；易弯 w60 成功 8/36 → 13/36。接口不变（`top_grasp.evaluate`、`sample_kappa` 等），GPU 版按 v1.3 规则另写；工作区的 `warp_*`、`validate_warp.py`、`validate_acceleration.py` 只适用于 v1.2。详见报告“v1.3 验收（续）”。**审阅（总体判断）**：接受冻结；GPU 只重写捏取部分（connect、覆盖判定、低通解开），S2 已通过的其余 kernel 复用，按统一五层标准做 G1–G5，11/13 前出结论；I-36 偏心捏取可能偏苛刻，列入真机第一批；合拢阶段压缩越限两种子合计 5/400（I-26），GPU 训练须标记剔除并检查是否集中在某个 κ 区域。见报告“v1.3 冻结审阅”。

1. **v1 冻结确认有效，原包不动。** `segment/results/freeze_v1_newton/release_v1/v1.tar.gz` 保留为本体/侧夹基线；M3 15/18、随机确认失败1/200、I-22 10/10、软物中位6.791/7.880s，既有未命中与I-29保留。顶面抓使用下述v1.2。
2. **v1.1 形状暂停。** 恢复时按报告审阅加入精确长方体、形状条件化标定，并采用审阅明确的新回归标准；本轮不做这些修改，也不把原 V2 失败等同模型错误。
3. **v1.2效果模型已冻结，E1–E6全部通过。** E4按审阅取消严格全任务集合要求，改为成功数不减、保持事件包含、反例≤5%，核对通过；w60当前已支持不同段，两指均能绑定，低成功率来自格距附近真实力先达目标而弹簧未聚料，按要求不改规则。E5新种子0/200，E6易压中位1.4857倍、余量约0.1125s，需保留机器/执行路径说明。标签位置见第2节，报告“E4审阅后核对与E5/E6”、`v22_review/acceptance.json`及同一检查报告给出完整证据；I-35本轮审阅已解决，旧失败不删除。
4. **论文独立数据。** 已从独立解包且指纹一致的v1完成R3/RJ各200物体、800次R1y/R2测量；数值失败7/200与10/200，完整软物二维有效57/133与58/133，CSV与分位数见 `segment/results/top_grasp_v1_2/paper_analysis/`。这是R1y/R2工况，不能替代v1静置+R3稳定性结论。
5. **版本接入与未验收范围。** 策略/Isaac Lab线可固定v1.2包接入与训练：`sample_kappa(rng,category)`返回含pinch_coeff的κ（当前总夹持力的承载系数、硬物0），RJ保留`sample_rj`完整描述；使用`scene()`和`Sim.step()`，内容物验收执行为`top_grasp.evaluate(k,grasp,seed)`。各线按协议v2.2移植并自行验证速度/数值，活动工作区不替代冻结包。未做MJWarp，无实测标定，装袋组合由袋子线验收。

代码只改 `segment/`，数据放各阶段的新结果目录；正式运行排队共享锁。详见内容物报告“v1.1 审阅与下一步”“v1.2 效果模型验收”。

## 8. 踩过的坑

- **`pgrep -f` / `pkill -f` 会匹配到自己。** 命令行里含目标字符串时，会把执行它的 shell 也算进去，导致等待脚本永不结束，或者把自己杀掉。要用 `^/home/.../python` 这种锚定开头的写法，或者按进程号等待。
- **用驱动反推受力时要扣除阻尼。** 压头或夹爪运动时，驱动阻尼项（kd × 速度）会被误当成接触力。压头以 1 cm/s 下降时偏差约 20 N。
- **MJWarp 的 `nconmax` 不能按实际接触数设置。** flex 表皮的候选碰撞对远多于实际接触，设成 600 会大量丢接触；目前用每个环境 4000。`njmax` 设 2000 就够（实际最大约 1590 行），设成 12000 会显存不足。
- **查看器需要 `DISPLAY=:1`**，否则会立刻退出。
- **碰撞分组**：`TOOL_CONTACT`（contype=1、conaffinity=2）和地面之间互相不碰撞。需要同时碰地面和软体的刚体，要用 contype=3、conaffinity=3。
- **GPU 上的 Newton 结果不完全可重复**：同一脚本可能一次夹住、一次掉落，所以夹取测试要重复多次统计成功率。
- **只看平均下陷会漏掉振动。** 第 0 步初探就是这样把“平均力对了”当成“力传对了”。判断接触要同时看节点速度（中位和最大）、力平衡残差和末段漂移。
- **Newton 1.6.1 的 `rigid_contact_k_start` 不起作用**：默认罚刚度递增系数 beta=0 时被忽略（`solver_vbd.py:995`），设成 1e12 也一样。

## 9. 常用命令

历史轮次的入口现已迁至 `test/content_model/`（工作目录为仓库根目录；列出用于定位与复现，不是仍待执行的任务。直接重跑会覆盖本轮结果，须先规划下一轮输出目录）：

```bash
flock /tmp/claude-gpu.lock /home/ubuntu/Dev/Envs/mjwarp/bin/python -B test/content_model/run_round2.py s1 s5 s6 --workers 30
flock /tmp/claude-gpu.lock /home/ubuntu/Dev/Envs/mjwarp/bin/python -B test/content_model/mjwarp_validate.py --nworld 1024
```

以下为旧路线命令，保留用于查看历史模型，不是当前分段模型的验收入口：

```bash
C=/home/ubuntu/Dev/Projects/random/contents
PY=/home/ubuntu/Dev/Envs/mjwarp/bin/python

# 交互查看（flat | blob | pillow）
DISPLAY=:1 $PY -B $C/legacy/particle/view_soft.py pillow

# 单项测试（shape: flat | blob | pillow；test: settle | cantilever | squeeze | finger | edge_lift | top_pinch）
$PY -B $C/legacy/particle/particle_tests.py pillow squeeze

# 闭环标定
$PY -B $C/legacy/particle/calibrate.py pillow --rounds 4

# MJWarp：验证 GPU 与 CPU 一致，测批量速度
$PY -B $C/legacy/particle/mjwarp_soft.py check
$PY -B $C/legacy/particle/mjwarp_soft.py speed --worlds 1 64 256 512

# Newton 实验（用 Newton 1.6.1 环境；1.7.0 不接受标量重力，需改成向量）
/home/ubuntu/Dev/Projects/real2sim/previews/newton-official-bag/.venv/bin/python -B $C/../test/contact_transfer/unified_tests.py folded_clothing speed

# 硬物（用 MuJoCo 3.14 环境）
/home/ubuntu/Dev/Envs/mujoco_env/bin/python -B $C/legacy/rigid/run_tests.py
```

## 10. 尚未确认的假设

- 真实平行夹爪的型号、控制方式、法向力读数和指垫几何尚未对齐；当前内容物协议使用 40×40 mm、圆角 4 mm 指垫。
- 物流内容物的分布（叠平与揉团的比例、是否真空压缩）都是假设，没有数据。
- 粒子模型在 MJWarp 上的速度瓶颈（稠密矩阵、显式步长、求解迭代次数）是推测，没有逐项测过。

## 11. 目录里其他东西

- `/home/ubuntu/Dev/Projects/random/soft-deformable-four-assets-only-20261022/`：不是本项目创建的，没有动。
- `/home/ubuntu/Dev/Projects/random/MUJOCO_LOG.TXT`：MuJoCo 运行时自动写的日志，可以删。

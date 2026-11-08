# 软包参数向量 κ 规格

2026-11-03（Asia/Taipei）。状态：设计，尚未实现。由总体判断会话维护（`../AGENTS.md` 第 8 节）。

## 目的

参照 XHugWBC（arXiv 2602.05791，第 3.1 节）的做法：一个结构固定的模板，一个只含物理量的参数向量 κ，再加一层重参数化 θ ↔ κ。这样在无约束的 θ 空间里随便扰动，得到的 κ 都是物理上合法的物体。论文用伪惯性矩阵的 Cholesky 分解保证 J ≻ 0；软包没有对应的矩阵约束，改用下文的逐项映射。

本文件规定三件事：κ 包含哪些参数，每个参数的约束和映射方式，以及哪些东西不属于 κ。它是袋子线、内容物线和基准共用的参照；参数改名或加减时，先改这里。

## 总体结构

```
κ = [κ_content, κ_bag, κ_interface]                         只含物理量
platform = 求解器、步长、网格分辨率、接触罚刚度、摩擦正则化  所有物体共用一份
```

三条原则：

1. **κ 只放物理量。** 步长、迭代、网格单元数、接触罚刚度这类为了让仿真跑得动而选的量，放进平台配置。现在 `bag_sim/bag_model.py` 的 `DEFAULTS` 把两者混在一起，要拆开（见“现有参数的归属”一节）。
2. **一个物理量只有一个独立参数。** 质量由密度和几何算出，不单独随机。袋子用面密度；内容物用总质量和质心偏移，尺寸扰动时按密度不变同步缩放。每段惯量由段的尺寸和质量份额算出，和论文“惯性参数由几何和密度一致导出”是同一个意思。
3. **数值可行性单独记录。** 物理上合法的 κ 不一定能在给定平台上仿真（会翻转、抖动或要求极小步长）。可行域由实测扫描得到，不写进映射（见“数值可行性”一节）。

## 重参数化规则

| 约束类型 | κ 由 θ 得到 | 反变换 θ 由 κ 得到 |
|---|---|---|
| 必须为正 | `κ = exp(θ)` | `θ = log κ` |
| 开区间 (a, b) | `κ = a + (b − a)·sigmoid(θ)` | `θ = logit((κ − a)/(b − a))` |
| 下界 c，无上界 | `κ = c + exp(θ)` | `θ = log(κ − c)` |
| 小于另一个参数 y | `κ = y·sigmoid(θ)` | `θ = logit(κ / y)` |
| 递增序列 x₁ < x₂ < … | `x₁ = exp(θ₁)`，`x_{k+1} = x_k·(1 + exp(θ_{k+1}))` | 逐项反解 |
| 单位向量 | 用两个角度表示 | — |

闭区间的端点（例如允许为 0 的比例）在 θ 空间里取不到；需要端点值时，用离散开关表示“这一项关闭”，不用把 θ 推到无穷。

扰动在 θ 空间进行：`θ' = θ_anchor + ε`，ε 每个分量独立，标准差按参数单独设定。锚点是现有预设（4 种袋子、3 种内容物）。**扰动范围目前全部待定**，要靠实测（`../log/ideas.md` I-06）给出依据；在那之前只能用于检查映射和可行性，不能当作物理分布。

## κ_content：内容物（分段刚体模型）

模型见 [segment_model.md](segment_model.md)。内容物参数**由夹爪层面的整体响应定义**，每个量的意义由 [响应定义](../test/content_model/README.md) 的固定测法确定，仿真和实物用同一种测法。模型内部通过换算表把目标响应变成关节参数，所以关节刚度不出现在 κ 里。

**几何、质量与分段**

| 参数 | 单位 | 约束 | 映射 | 备注 |
|---|---|---|---|---|
| `size` (Lx, Ly, Lz) | m | > 0 | exp | 包络尺寸（v1 为长方体） |
| `shape_p`、`shape_q` | — | [2, 12] | 2 + 10·sigmoid | v1.1 新增：超二次曲面包络的横截面指数和长度方向指数，见 `segment_model.md`“外形（v1.1）” |
| `mass` | kg | > 0 | exp | 尺寸扰动时按密度不变同步缩放，避免两者独立变化 |
| `com_offset` (x, y, z) | 包络尺寸的比例 | (−0.5, 0.5) | sigmoid 缩放 | 模型换算成各段的质量份额 |
| `segments` (nx, ny, nz) | — | 每个方向 ∈ {1, 2, 3, 4, 6}，上限 6 × 4 × 3 | 离散 | 只决定局部分辨率；采样约束：`droop_ratio` 越大，最少段数越多（阈值第 1 轮定） |
| `rigid` | — | 开关 | 离散 | 硬物整体锁死 |

**整体响应**

| 参数 | 单位 | 约束 | 映射 | 测法 |
|---|---|---|---|---|
| `squeeze_stiffness_y`、`squeeze_stiffness_z` | N/mm | > 0 | exp | R1：压缩 5%–15% 的割线斜率 |
| `lock_strain_y`、`lock_strain_z` | — | (0, 1) | sigmoid | R1：切线刚度超过初始 5 倍时的压缩比 |
| `droop_ratio` | — | [0, 1) | sigmoid（0 用 `rigid` 开关） | R2：远端下降量 / 长度 |
| `damping_ratio` | — | > 0 | exp | 关节阻尼比，全部自由度共用 |
| `pinch_coeff` | — | ≥ 0（硬物恒为 0） | exp（0 用开关） | v1.2 新增：顶面捏取的承载系数，R4：拉脱力 / 夹持力 |

**活动质量（可选单元，默认关闭）**

| 参数 | 单位 | 约束 | 映射 | 备注 |
|---|---|---|---|---|
| `mobile` | — | 开关 | 离散 | |
| `mobile_frac` | — | (0, 1) | sigmoid | 活动质量占总质量的比例 |
| `mobile_range` | m | (0, 包络最短边的一半) | “小于另一参数” | 活动范围 |
| `mobile_freq` | Hz | > 0 | exp | 活动质量的固有频率 |

**等效外皮（满装包裹的袋子效果，第 1 轮之后加入）**

| 参数 | 单位 | 约束 | 映射 | 备注 |
|---|---|---|---|---|
| `surface_friction` | — | > 0 | exp | 包裹表面的摩擦系数，取值覆盖 PE 到牛皮纸 |
| `skin_compliance` | m/N | ≥ 0 | 开关 + exp | 可选，表示气泡袋这类缓冲层；关闭时为 0 |

拉伸与压缩、剪切与压缩、扭转与弯曲之间第一版用固定比例，不单独随机；第 1 轮若三类不够分明再放开。迟滞、蠕变、精确体积变化等真实任务暴露需求后再加，加入时在这里补参数和测法。

## κ_bag：袋子

**第一版不使用**（2026-11-03）：满装包裹的袋子效果并入 κ_content 的等效外皮；松装包裹的袋子表示由袋子线的对比实验决定（`../test/bag_model/REPORT.md`）。下表保留，用于以后需要膜表示时。


当前由 `bag_sim/profiles/*.json` 加上 `bag_model.DEFAULTS` 给出。下表只列物理量。

| 参数 | 单位 | 约束 | 映射 | 备注 |
|---|---|---|---|---|
| `membrane_model` | — | elastic / inextensible / tension_only | 离散 | |
| `areal_density_gsm` | g/m² | > 0 | exp | 取代 `mass_kg`，质量由面积导出 |
| `young_pa`、`membrane_thickness_m` | Pa、m | > 0 | exp | 膜刚度实际只取决于乘积 E·t；建议 κ 里直接用 E·t，厚度单独留给接触厚度 |
| `poisson` | — | [0, 0.5) | sigmoid 缩放到 (0, 0.5) | |
| `membrane_damping` | — | ≥ 0 | exp | |
| `bending_rigidity_nm` | N·m | > 0 | exp | |
| `region_bending_scale` | — | 每个区域 > 0 | exp | |
| `anisotropy_axis` | — | 单位向量 | 两个角度 | |
| `anisotropy_ratio` | — | > 0 | exp | |
| `bending_knee_deg`、`bending_post_ratio` | deg、— | 开关；> 0 | 离散 + exp | `None` 表示关闭 |
| `plasticity` 及 `plastic_yield_angle_deg`、`plastic_hardening_ratio` | —、deg、— | 开关；(0, 180)；[0, 1) | 离散 + sigmoid | |
| `memory_ratio`、`recovery_tau_s` | —、s | ≥ 0；> 0 | exp | |
| `precreased_stiffness_ratio` | — | (0, 1] | sigmoid | 1 用开关表示 |
| `crease_softening` 及 `crease_onset_deg`、`crease_width_deg`、`crease_max_softening` | —、deg、deg、— | 开关；> 0；> 0；[0, 1) | 离散 + exp/sigmoid | |
| `bending_damping_nms` | N·m·s | ≥ 0 | exp | |
| `collision_radius_m` | m | > 0 | exp | 表示壁厚的一半，作物理量保留 |
| `cushion` 及 `cushion_range_m`、`cushion_soft_modulus_pa_per_m`、`cushion_lock_ratio`、`cushion_damping_ratio` | — | 开关；range < collision_radius；> 0；≥ 1；≥ 0 | 离散 + “小于另一参数”、exp、“下界”、exp | 气泡缓冲层是真实结构 |
| 几何（袋子尺寸、开口） | m | 见“跨组约束” | — | 目前是固定网格文件，见“已知缺口” |

## κ_interface：界面

| 参数 | 约束 | 映射 | 备注 |
|---|---|---|---|
| `friction`（袋–袋） | ≥ 0 | exp | |
| 袋–内容物摩擦 | ≥ 0 | exp | 目前隔膜基准用 0.5 |
| 袋–桌面摩擦（`table_friction`） | ≥ 0 | exp | |
| 内容物–桌面摩擦 | ≥ 0 | exp | |

夹爪与袋之间的摩擦属于场景（夹爪），不放进物体的 κ。接触罚刚度和摩擦正则化 ε 是数值选择，归平台配置。

## 跨组约束

| 约束 | 处理方式 |
|---|---|
| 袋子能包住内容物：袋子每个内尺寸 ≥ 内容物尺寸 + 2 × 接触间隙 | 袋子尺寸 = 内容物尺寸 + 2 × 间隙 + exp(θ)，即袋子尺寸随内容物导出，不独立采样 |
| 袋子初始位置不触发自接触势垒：相对两壁距离 ≥ 接触距离 + 0.2 mm | 由几何初始化保证（`bag_sim` 已有类似处理） |
| 总质量 = 内容物质量 + 袋子质量 | 袋子由面密度导出，内容物质量直接给出，相加即可 |

## 现有参数的归属

`bag_model.DEFAULTS` 里以下各项**不属于 κ**，移到平台配置：

- 时间与求解器：`timestep_s`、`frame_dt_s`、`iterations`、`solver`、`tolerance`、`newton_substep_s`、`newton_substep_courant`、`newton_min_substep_s`、`newton_iterations`
- MuJoCo 专用的约束参数：`edge_solref_s`、`edge_solimp`、`contact_timeconst_s`
- 接触的数值设置：`collision_patch_faces`、`newton_contact_modulus_pa_per_m`、`newton_self_contact_modulus_pa_per_m`、`newton_self_contact_scale`、`newton_self_contact_ramp_s`、`newton_full_surface_contact`
- 数值阻尼：`velocity_damping_per_s`
- 场景开关：`gravity`、`self_contact`、`newton_self_contact`、`self_contact_front_panel_ids`、`self_contact_back_panel_ids`、`precreased_panel_ids`、`precreased_panel_boundaries`（面板编号依赖具体网格，归几何）

`newton_contact_modulus_pa_per_m` 是否部分代表真实壁面柔度，还没定论；在弄清之前按数值设置处理。

## 数值可行性

这一部分是论文没有、软包必须有的。对每个 κ 和一份平台配置，可行的条件是：

1. 跑完静置、平压、夹起三项快速检查，不发散、不翻转；
2. 接触验收达标（标准沿用 `../test/contact_transfer/` 和 `../test/bagged_contact/`）；
3. **请求的 κ 和实际仿真用的 κ 一致。** `bag_sim` 在子步低于 float32 下限时会自动软化膜刚度（`membrane_scale`），这时实际 E·t 小于请求值。这种情况必须记录实际值；超出时判为在该平台上不可行，不能悄悄换成另一个物体。

可行域通过扫描得到：先测每个锚点，再测 θ 空间的角点和随机点，记成一张表。采样时限制在可行域内；可行域太小的地方，说明需要修求解器或本构，而不是收窄物理范围。

## 已知缺口

| 缺口 | 影响 | 处理方向 |
|---|---|---|
| 所有扰动范围没有依据 | 不能当作物理分布使用 | 按响应定义粗测几件最软、最硬的典型物品（I-06）；内容物参数本身就是可实测的响应量 |
| 内容物外形只有长方体包络 | 揉团、枕头鼓起这类外形差异表示不了 | 第 1 轮通过后，按需要加包络形状参数 |
| 袋子几何是每种袋子一个固定网格文件 | 袋子尺寸不能随内容物导出，跨组约束无法满足 | 做参数化的袋子网格生成器 |
| 膜刚度会被自动软化 | κ 不等于实际物体 | 按“数值可行性”第 3 条记录和判定 |
| 内容物模型第 1 轮部分通过 | 随机 κ 下 R3 失败率 6.5%；采样还没有可行域约束（自重、段数、接缝位置） | 见 `../test/content_model/REPORT.md` 第 1 轮结果和“需要用户决定的地方” |
| 换算表只按（锚点几何, 分段数）建了 6 张 | 锚点上 18 项里 14 项误差 ≤ 20%；尺寸随机时的误差未测 | 加密扫描、扩大 z 向下限，表外目标要报出来 |
| 单调性只验证了易弯物锚点 | 夹紧刚度 2 个数量级内单调；下垂在 0.12 以下撞自动锁死出现平台 | 第 1 轮 T6；其他锚点未测 |

## 实现时的验收

实现代码时，单元测试至少覆盖：

- **往返**：κ → θ → κ 与原值一致（相对误差 < 1e-9）；
- **合法性**：随机抽取大量 θ（包括很大的 |θ|），得到的 κ 都通过 `bag_model.parameters()` 和内容物模型的参数检查；
- **连续性**：θ 的小扰动只引起 κ 的小变化；
- **锚点不变**：现有预设经过拆分（κ + 平台配置）后，重新组合得到的参数与原文件一致，原有回归结果不变。

内容物部分随分段模型一起实现；袋子部分在 `bag_sim` 上拆分参数，两者可以并行。

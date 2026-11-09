# 软包统一物理模型 交接文档

2026-11-02 · 状态：Newton 后端第一阶段完成，第二阶段（塑性/记忆/折痕）未开始

四种外层袋子（PE 快递袋、牛皮纸邮寄袋、纸板文件封、气泡信封袋）用**同一套方程**，只靠 `profiles/*.json` 里的参数区分。有两个后端：MuJoCo（对照基准）和 Newton VBD（后续开发方向）。所有参数均为估计值，**未做实物标定**。

本目录自包含：代码、参数、几何、纹理、对照资产和测试记录都在这里，不依赖外部路径。

---

## 1 目录

```
bag_sim/
├── bag_model.py            共用：参数校验 parameters()、几何与材料 load_material()；MuJoCo 后端 BagModel
├── newton_bag.py           Newton VBD 后端 NewtonBag；content_shape_config()/content_mesh() 给刚体配 SDF
├── profiles/               四个参数文件 pe / kraft / paperboard / bubble
├── geometry/               四个袋子的参考网格（npz）
├── textures/               对应纹理
├── make_bubble_geometry.py 用旧袋拓扑生成气泡袋网格（结果与 geometry/bubble_180x240.npz 逐位一致）
├── wrap.py                 把袋网格包到内容物外的初始化 wrap_generic（纯 numpy，Newton 与 MuJoCo 共用）和包含判定 winding_number
├── mj_loose.py             MuJoCo 3.17 松装袋表示（薄板片 plates / flex 薄膜），装 contents/segment 的分段内容物（第 1 轮 C2）
├── self_clearance.py       诊断：VBD 会评估的点-三角形对的最小距离
├── legacy/                 旧袋的初始化与内容物工具（asset_mesh / contents / initialization），原样复制
├── reference/              原 runtime.py 与三类资产（仅供 MuJoCo 回归），LICENSES 与 PROVENANCE
└── outputs/                GIF/PNG、录制 npz（recordings/）、内容物 OBJ（meshes/）；第 1 轮结果在 outputs/round1/
```

独立检查、演示、行为验证、分析和测速已移至 [test/bag_model/](../test/bag_model/README.md)。本目录维护建模、材料、几何与共享后端；新测试结果按 [日期与对象](../test/runs/README.md)管理。

来源：`soft-deformable-four-assets-only-20261022`（与 `real2sim/soft-package-assets` 相同）。原资产未改动；旧工作目录 `assets/unified/` 已并入本目录。

## 建模方案与版本入口

| 版本或方案 | 模型文件与参数 |
| --- | --- |
| 原始四资产 / MuJoCo 3.14 对照 | `reference/`、原资产包，保持原始来源和内容 |
| 统一材料模型 / Newton VBD | `bag_model.py`、`newton_bag.py`、`bending_history.py`、`profiles/` |
| 薄板片与 2D flex 袋 | `mj_loose.py`、`flex_geometry.py`、`wrap.py`、`loose_setup.py` |
| Round 3 效果袋 | `effect_model.py`、`film_pinch.py`，冻结内容物通过 `frozen_content.py` 接入 |
| 完整 flex 与软内容物联合求解 | `soft_flex_adapter.py`、`soft_content_backend.py`、`flex_*` 辅助模块 |
| 静态插值袋候选 | `interpolated_bag.py` |
| 原生 3D flex 内容物 / 整包候选 | `native_flex_contents.py`、`native_flex_gpu.py` |

以上使用已有方案名称，不代表都已验收或冻结；具体边界以本文最新状态和 [实验报告](../test/bag_model/REPORT.md) 为准。`outputs/` 中的旧证据和被显式引用的后端快照保留原位；独立测试代码统一在 `test/`。

## 2 环境与运行

两个 Python 环境（都已在本机）：

| 用途 | 解释器 | 版本 |
|---|---|---|
| MuJoCo 后端、渲染 | `/home/ubuntu/Dev/Envs/mujoco_env/bin/python` | MuJoCo 3.14.0（代码强制） |
| Newton 后端 | `/home/ubuntu/Dev/Projects/real2sim/bubble_mailer/.venv/bin/python` | Newton 1.6.1、Warp 1.18、Python 3.11、RTX 5090 |

从仓库根目录运行：

```bash
NP=/home/ubuntu/Dev/Projects/real2sim/bubble_mailer/.venv/bin/python
MP=/home/ubuntu/Dev/Envs/mujoco_env/bin/python
$NP -B test/bag_model/demo_contents.py book --profile bubble --seconds 1.5          # 装内容物静置；--set '{"key": value}' 临时覆盖参数
$NP -B test/bag_model/demo_newton.py pe --seconds 2                                  # 空袋掉落
$MP -B test/bag_model/render_demo.py frames bag_sim/outputs/bubble_book.npz                  # 渲染帧（写 *_frames.npy，约 35 MB/段）
$NP -B test/bag_model/render_demo.py gif bag_sim/outputs/x.gif --seconds 1.5 bag_sim/outputs/bubble_book_frames.npy
$NP -B test/bag_model/check_newton.py cantilever grid 20 | press 20 120 2 | profile pe 0.3
$MP -B test/bag_model/check_unified.py regression pe                                 # 应输出 max_position_difference_m: 0.0
```

代码调用：

```python
from newton_bag import load_profile_newton
bag = load_profile_newton("bubble", params={"friction": 0.6})   # params 只覆盖本实例
for _ in range(100):
    bag.step()                                                   # 一帧 = frame_dt_s（0.01 s）
print(bag.state_report())
```

`extra=callback(builder, contact)` 可在建模前往同一场景加刚体；刚体形状请用 `content_shape_config()`（基本体）或 `content_mesh()`（网格），否则全表面接触会因缺 SDF 报错。

## 3 模型

每个袋子是三角形中面网格，节点三个平移自由度。参数文件对五部分取值：

| 部分 | 共用计算（`load_material`） | MuJoCo 后端 | Newton 后端 |
|---|---|---|---|
| 质量 | 节点集中质量；可按 `mass_kg` 或 `areal_density_gsm` 重算 | 同左 | 同左（覆盖 Newton 自算质量） |
| 膜 | — | `inextensible` 边长等式 / `tension_only` 单侧腱 / `elastic` StVK | Neo-Hookean，E·t 与泊松比换 Lamé；无硬约束，`inextensible` 也按 E·t；无 `tension_only` |
| 弯曲 | 铰链刚度 K = D·3l²/(A₁+A₂) × 分区 × 方向性 × 预折线 | Python 回调，含拐点、塑性、SLS 记忆、折痕软化 | 每边 `edge_ke = K/l`、`edge_kd = c/l`（VBD 乘以边长），**仅线性** |
| 厚度/缓冲 | — | 软接触 solref 按面密度缩放（仍与网格相关，不建议用） | 粒子半径 + 线性罚函数 ke = 模量 × 平均顶点面积 |
| 接触 | — | 前/后片组自接触 | VBD 全自接触（含同片折叠）；刚体可碰边和面（全表面接触） |

### Newton 专用参数（`bag_model.py` DEFAULTS，MuJoCo 忽略）

| 参数 | 默认 | 作用 |
|---|---|---|
| `newton_substep_courant` | 0.7 | 子步 = 系数 × 膜波步长 h·√(ρ/μ)（h 为边长中位数） |
| `newton_substep_s` | 0.001 | 子步上限 |
| `newton_min_substep_s` | 5e-5 | 子步下限（float32）；不够时自动软化膜，`state_report()["membrane_scale"]` 给出倍数 |
| `newton_iterations` | 10 | VBD 迭代次数 |
| `newton_contact_modulus_pa_per_m` | 2e8 | 无缓冲时的接触模量 |
| `newton_full_surface_contact` | true | 刚体与袋子边、面也接触（防尖角穿出） |
| `newton_self_contact` | true | Newton 自接触开关（PE 设为 false） |
| `newton_self_contact_scale` | 2.0 | 自接触距离 = 系数 × 碰撞半径（2 = 壁厚 2r） |
| `newton_self_contact_ramp_s` | 0.1 | 启动时自接触距离从 0 按二次曲线增至目标值 |

### 各参数文件要点

| 键 | 几何 | 质量 | 自动子步 | 备注 |
|---|---|---|---|---|
| `pe` | 350×450 mm，549 节点 | 15.3 g | 0.068 ms | E·t = 1e4 N/m；**Newton 自接触关闭** |
| `kraft` | 250×50×350 mm，413 节点 | 28.0 g | 0.109 ms | 侧褶预折线刚度 1%；塑性参数在 Newton 被忽略 |
| `paperboard` | 325×230 mm，268 节点 | 41.4 g | 0.1 ms（参数文件设上限） | 膜用默认 E·t = 8e3 N/m，偏软（见已知问题 3） |
| `bubble` | 180×240 mm，570 节点 | 12 g（按 110–140 g/m² 估） | 0.05 ms | 膜被自动软化到 31%；缓冲 5e6 Pa/m 为估计值 |

## 4 验证结果

### MuJoCo 回归
PE、牛皮纸、纸板跑 40 子步，与原 `runtime.py` 位置差 **0.0**（`check_unified.py regression`）。

### Newton 弯曲（120 mm PE 悬臂，D = 2e-6 N·m，46 g/m²，解析弹性线为参考）

| 网格 | 下垂误差 20 / 10 / 5 mm | 伸出误差 20 / 10 / 5 mm |
|---|---|---|
| 方格（边沿折轴） | +2.9 / +1.2 / −0.2 % | −7 / +15 / +28 % |
| 三角（边不沿折轴） | −10 / −6.6 / −2.4 % | +177 / +107 / +53 % |

三角网格粗时偏硬（膜锁定），随加密收敛。伸出量本身仅约 9 mm，对误差敏感。

### Newton 缓冲（压板，全表面接触开启）
实际刚度约为参数的 **8.7 倍**，在 10 与 20 mm 网格上一致；与面密度无关（未开全表面接触时验证）。按常数补偿在 6 kPa 时压穿且出现质量依赖，故未采用。

### Newton 速度（单进程、RTX 5090、当前默认）

| PE | 牛皮纸 | 纸板 | 气泡袋 |
|---|---|---|---|
| 0.026 | 0.035 | 0.058 | 0.012 |

实时率；耗时主要在 VBD 每次迭代的接触力计算，CUDA graph 只提速约 10%。

### 装内容物静置 1.5 s（最终版本，录制在 `outputs/recordings/`）

穿出量为内容物顶点超出袋面中线的最大距离，负值表示在袋内。

| 袋子 | 书 250 g | 瓶 300 g | 球 150 g | 圆环 120 g（非凸） |
|---|---|---|---|---|
| 气泡袋（自接触 2r） | −0.68 | −1.27 | −1.30 | −1.54 |
| PE（自接触关） | −0.22 | +0.08 | −0.20 | −0.16 |
| 牛皮纸 | −0.32 | +0.14 | −0.33 | −0.27 |

| 纸板（子步上限 0.1 ms） | +0.15（平移 25 mm，袋仍以 0.34 m/s 晃动） | 未测 | 未测 | 未测 |

纸板另测 8 mm 文件（200×150 mm，250 g，更符合文件封用途）：−0.09 mm，平移 0.7 mm，边拉伸 0.2%，静止。

全部内容物留在袋内，无发散；除纸板装书外，1.5 s 时内容物速度 ≤ 0.017 m/s。正值均小于各自的接触半径（PE 0.25 mm、纸类 0.35 mm）。

## 5 已知问题（按严重程度）

1. **PE 的 Newton 自接触不可用。** 前后片在参考形状中重合，VBD 自接触在半个作用距离以下是对数势垒，包住内容物后边-边对仅相距 0.018 mm，起步即弹飞。拓扑过滤放宽、参考形状排除半径、自接触距离爬升 0.3 s 都未解决。当前 `pe.json` 设 `newton_self_contact: false`，袋壁折叠时可互相穿过。
2. **第二阶段未实现。** 拐点、塑性、SLS 记忆、折痕软化在 Newton 中被忽略（`state_report()["ignored_until_stage2"]`）。纸袋弯后完全回弹；纸板包 3 cm 书会把书弹起。
3. **纸板膜刚度未设定。** 参数文件沿用默认 E·t = 8e3 N/m（远软于纸板），波步长规则因此给出 0.167 ms 子步，装文件时穿出 +0.36 mm、边拉伸 4.8%、装书穿出 +0.44 mm。现已在 `paperboard.json` 设 `newton_substep_s: 0.0001`，文件回到 −0.09 mm、0.2%。根本做法：填入实测级 E·t（届时子步会由波步长规则变小，需重测）。纸板装 3 cm 书时仍被推开 25 mm，属已知问题 2。
4. **缓冲层未标定，且只有线性段。** 全表面接触下实际刚度 ≈ 8.7 × 参数；无压实段。需用开启全表面接触的状态对实测力-间隙曲线标定。
5. **气泡袋膜被软化到 31%。** float32 下限 5e-5 s 与 13 mm 网格共同所致。
6. **无夹爪、无提起测试。** `../newton_soft.py` 内的夹取提升此前全部失败（提升率 0），迁移前须查明。（2026-11-04 补充：接触线在 `../proxy_eval/` 用本后端做了装硬盒/海绵的夹取—提起—搬运试验，未修改本目录代码；结果是耗时长，且挤压空袋边时常有膜节点略超 1 m/s，见 `../test/proxy_eval/REPORT.md` 第 1 轮和 `../log/ideas.md` I-23。）（2026-11-05 补充：`c1_newton.py` 用本后端做了装硬盒的 R2、R3：松装时能隔着袋子夹住硬盒，盒子在袋里滑动；R2 提起 0.49 m 不掉。膜节点仍有 1.01–1.02 m/s 的诊断标记，实时率约 0.01，见第 1 轮报告 C1。）
7. **速度。** 单袋实时率 0.012–0.058；提高吞吐需在同一 GPU 上并行多个场景（未做）。装内容物并加两个运动学指垫后，牛皮纸袋、气泡袋都按 50 µs 子步运行，实时率约 0.005–0.014（2026-11-04，代理测试集第 1 轮）。
8. **MuJoCo 版气泡袋不可用。** 第一步即 `mjWARN_INERTIA`，未修（Newton 版可用）。
9. **全部参数未经实物标定。**

## 6 关键结论（踩过的坑）

- **VBD 弯曲依赖子步。** 逐点求解看到膜的面外 Hessian，迭代严重欠松弛：单铰链 1 ms 子步下 1/10/100 次迭代只保留 0.7/6/48% 刚度。子步须小于膜波步长 h·√(ρ/μ)；网格越细、膜越硬，子步越小。
- **float32 冻结。** 子步低于约 5e-5 s 时位移被舍入，VBD 由位置差求速度，整片冻住（10 mm 条带 2.4e-5 s 完全不动）。
- **VBD 自接触是对数势垒。** 初始任何点-三角形或边-边对小于半个作用距离都会被弹开；`self_clearance.py` 只查点-三角形，边-边可用 `solver.trimesh_collision_detector.collision_info.edge_colliding_edges_min_dist` 读。
- **刚体-粒子接触缓冲区** 默认 256，袋子贴地即超过；已设 4096。
- **`rigid_contact_k_start` 在当前配置下不起作用**：它只是 AVBD 罚刚度递增的起始值，仅在 `rigid_avbd_beta`（或 `rigid_avbd_linear_beta`）> 0 时使用；默认 beta = 0，此时接触刚度直接取材料刚度，与该值无关（Newton 1.6.1 `solver_vbd.py:995`）。`newton_bag.py` 设的 1e12 是无效设置，传负数会报错。
- **全表面接触需要 SDF**：网格用 `mesh.build_sdf()`，基本体用 `cfg.configure_sdf()`。
- **CUDA graph** 要求每帧子步数为偶数（状态缓冲交替）；自接触距离爬升期间不能录图。
- **MuJoCo 软接触不是弹簧**：同样 1 N，0.2 g 与 200 g 物体穿透相差约 180 倍，缓冲层不能靠调 solref 实现。
- **旧袋 140 g** 约为实物 12 倍（疑为把 140 g/m² 当成总质量）；`drag=120` 使自由落体极限速度仅 0.08 m/s。

## 7 下一步（建议顺序）

1. PE 自接触：改初始几何（前后片留出 ≥ 作用距离的间隙）或在初始化中加入边-边分离投影。
2. 纸板：填入合理 E·t 后重跑内容物测试（当前靠 0.1 ms 子步上限维持）。
3. 第二阶段：每帧按当前角度更新每条边的静止角（塑性）与刚度（折痕软化），记忆合并为 K(1+m)、静止角 (p+m·z)/(1+m)。
4. 夹爪与提起测试；修复 `newton_soft.py` 提升率为 0 的问题。
5. 实测：称重；纵横向悬臂（D）；条带拉伸（E·t）；夹爪压缩 1–20 N 并保持（缓冲）；各接触对起滑力（摩擦）；带测力的夹取提升（仅验证）。
6. 多场景并行提吞吐。

## 8 输出说明

- `outputs/{bubble,pe,kraft,paperboard}_contents.gif/.png`：上表对应的最终版本。
- `outputs/newton_drop.gif/.png`：**早期版本**（未开全表面接触、子步固定 0.1 ms），仅作形态参考。
- `outputs/recordings/*.npz`：每段的帧、内容物位姿和检查结果（`checks` 字段，JSON）。

## 9 MuJoCo 3.17 松装表示（第 1 轮 C2、第 2 轮 F1–F5，2026-11-05；未交策略线）

**第 2 轮现状**：
- 薄板片已放弃，只保留 flex。
- 选定配置：`LooseBag(kind="flex", edge=0.035, radius=0.005)`，刚性内容物用 `dt=0.0008`，分段软内容物用 0.4 ms。`edge` 让 `make_kraft_coarse.py` 按袋子尺寸生成封口、带折边的牛皮纸网格（250 × 350 mm 时 200 节点）；折边深度取 max(25 mm, ½ 内容物高 + 2 r)。
- 速度：28 进程每秒 0.068 次 evaluate（第 1 轮 0.016，目标 0.3）。
- 硬盒的定性行为与 Newton 参考一致，穿透 0 mm。易弯物穿透 4–5.6 mm；满装贴合时会被压穿；顶面抓提不起松装袋。
- 结论：松装暂不进入训练（[第 2 轮结果](../test/bag_model/REPORT.md)）。
- 新增：`make_kraft_coarse.py`；`loose_mjwarp.py`（MJWarp 开环对照）；`mj_loose.fingertip_xml`（v2 指尖）；`LooseSim.penetration()`（穿透深度）。`loose_protocols.evaluate` 接受 8 个参数的顶面抓（w、d 用 m）；侧夹合拢时限 8 s。
- 场景改用内容物 `scene()` 的片段接口（I-27）。

**第 3 轮（F6/F7，限时 48 小时；2026-11-07 补记）**：
- F6/F7 已于11/7 07:52限时内完成。F7原计划216次加3个低承载对照：中部63/108成功，空袋端0/108且无上层膜连接；最大穿透0、确认仿真失败0。F7不通过，按原表 flex 退出训练候选，保留可视化/参考；转 B1–B4。
- 内容物现只从 v1.3 冻结归档导入（`frozen_content.py` 校验归档及所有源文件 SHA256），不再导入内容物线的工作副本。
- 入口：`round3_flex.py`（缓存对照、CPU 行为/速度、F7 网格）、`flex_batch.py`（完整侧夹流程的 MJWarp 批量）、`film_pinch.py`（上层膜 connect + v1.3 弹簧和过载释放）；输出统一在 `outputs/round3_flex/`。`loose_mjwarp.py` 仍是第 2 轮开环对照入口。
- 缓存/不缓存 9 对输入类别和逐步状态均相同；F6c 精度与 F6d 分区网格没有通过原行为范围，未采用。当前 v1.3 的基准 R2/R3 也未复现第 2 轮数值，原标准未改。完整结论和原始数据见[原报告](../test/bag_model/REPORT.md)“第 3 轮并行项”。
- GPU64/256/1024各九输入吞吐为0.10233/0.29139/0.59355次/s，峰值全卡显存2162/2230/8210MiB，均零溢出、多数类别9/9与CPU一致；最大穿透2.63/15.25/17.47mm均超标。1024达到速度目标，行为未通过；最终GPU R3下沉中位12.84mm、穿透2.17mm，也未通过原标准。
- 修复 flex 接触 `geom=-1` 的指尖力误索引；F7 先前 72 次保留为无效诊断数据，修正后重跑。当前仍未交策略线。
- `effect_model.py` / `round3_effect.py` 为 B1–B4 的原生空间 tendon 效果模型：三类CPU和硬盒GPU静置已完成、无发散或中心越界；软内容物GPU仍依赖已验收冻结后端。B2下沉0.068mm、空袋端0/3完成，未通过。B3修复中部碰撞路径后，易弯/易压中部1/3、2/3成功，空袋端均失败，侧夹均超动作范围，未完整通过。`effect_batch.py`硬盒B4带袋256/1024环境为59.51/160.58次/s，裸物221.81/660.91次/s，均零溢出；这是全部夹取失败的动作吞吐，不能作可用训练性能。剩余B1/B4软GPU待冻结后端，当前不能交策略线。[检查报告](../reports/bag-model-round3-effect-test.md)。

以下为第 1 轮的原文。

为松装包裹（袋与内容物相对滑动、夹到空袋边、提起袋角）找一种比 Newton 袋子便宜的表示，装 `contents/segment` 的分段内容物。本轮结论：两种都不够用（[第 1 轮结果](../test/bag_model/REPORT.md)）。flex 行为对但和 Newton 一样慢；薄板片比 flex 快约 25–35 倍，但空边压不出褶皱。下面的接口留作记录和后续试验，不是交给策略线的版本。

环境：`/home/ubuntu/Dev/Envs/mjwarp/bin/python`（MuJoCo 3.17）；本节的脚本在本目录下运行。

```python
from mj_loose import LooseBag, LooseSim, loose_scene
from model import ANCHORS, build                      # contents/segment，由 mj_loose 加入 sys.path
desc = build(ANCHORS["bendable"])
lb = loose_scene(desc, LooseBag(kind="plates", fill=0.75, compress=0.0), pads=[("pA", 1, (0, -1, .5)), ("pB", 1, (0, 1, .5))])
sim = LooseSim(lb, ["pA", "pB"]); sim.wrap()          # 合上袋缝并静置；之后 t = 0
sim.step(); sim.pkg_com(); sim.bag_failure(); sim.footprint()
```

| 参数（`LooseBag`） | 含义 |
|---|---|
| `kind` | `plates`：薄板片；`flex`：参考网格做 dim-2 flex |
| `size` / `fill` | 平放袋身 (L, W)；或由 fill 算：L = (Lx + Lz)/fill，W = (Ly + Lz)/fill（≈0.95 满装，C1 松装牛皮纸袋约 0.43） |
| `profile` | 摩擦、质量、弯曲刚度 D、预折线比例和 flex 网格的来源（默认 kraft） |
| `offset` | 内容物在袋里的位置，按自由余量的一半归一化 |
| `cell`、`thickness`、`gusset`、`crease`、`curl` | 薄板片：板片尺寸 50 mm、碰撞厚度 2 mm、两侧折边 25 mm（kraft）、折线刚度比例、背片外圈上翘静止角 0.2 rad |
| `compress`、`compress_k` | 薄板片受压可缩短的比例和刚度；第 1 轮正式结果用 `compress=0`（面内刚性），默认值 0.9/2 N/m 是已弃用的变体 |
| `membrane` | flex 的 E·t（软化到 1e3 N/m）；弯曲按 D 选等效厚度 |

任务与测试：`loose_protocols.py eval|r2|r3 --bag plates|flex --content box_loose|bendable|compressible [--set JSON]`；`evaluate(object_spec, grasp_params, seed)` 的 `object_spec` 比策略线多一个 `"bag"` 字段（`LooseBag` 的字典）。

已知问题：

1. **薄板片空边不起皱**（主要问题）。面内刚性时，空边被推成刚性板片堆，提起时滑出；装硬盒的闭合袋像一根硬管，夹空端也能抬起盒子。受压可缩短后空边平躺，指垫只顶着棱边，同样夹不住。30 mm 板片也没有解决，且慢 5 倍以上。
2. **装袋场景改用 Newton 求解器**。CG 100 次在轻袋片与重内容物同场时不收敛（I-27）。Newton 的耗时大致随 nv³ 增长，因此板片数决定速度。
3. **flex 只能用 `discrete` 积分器**：MuJoCo 3.17（CPU 和 MJWarp）都不允许 flex 弹性与 implicitfast 同用，内容物在该积分器下未验证。第 2 轮查明：
   - 慢的原因是 Newton 求解器随 nv 急增：413 节点时每步 78 ms，其中求解 69 ms。
   - 盒角“穿过三角面”是真实穿透：第 1 轮配置按深度口径 R3 为 12.7 mm。原因是软化膜（E·t 1e3 N/m）、粗网格和夹持力共同作用。碰撞半径 5 mm 时硬盒为 0 mm。
   - 大于约 0.8 ms 的步长会让轻节点的接触被步长上限压软，于是穿过。
4. **内容物只通过字符串替换接入 `scene()`**（改 conaffinity、删指垫对、换求解器、补 TendonInfo），内容物接口一变就可能失效，接口需求已写进 I-27。
5. 平放时边缘完全贴桌会卡死，所以背片外圈有 0.2 rad 的上翘静止角；空袋平放时边缘约翘起 8 mm。

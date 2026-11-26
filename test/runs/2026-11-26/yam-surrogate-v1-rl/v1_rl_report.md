# Yam_B Policy V3 低保真代理预训练

## Preflight Discovery

2026-11-27 北京时间，主机 `ubun`，RTX 5090（驱动 595.80，32,607 MiB）。Git 起点 `52fbb71f`，读取远端后本地和 `origin/main` 同步。共享工作树已有其他任务的修改。

| 核查项 | 状态 | 结果 |
| --- | --- | --- |
| 分段刚体袋体 | PASS | 找到 `bag_sim/mj_loose.py` 的 `LooseBag(kind="plates")`。前后薄板用弹性铰链和可选压缩关节连接，自由盒状内芯通过板片碰撞承载。历史报告记录片间面内刚性会使整袋接近硬管。该方案已按用户要求排除。 [构建源码](../../../../bag_sim/mj_loose.py) · [历史薄板片对照](../../../bag_model/REPORT.md#第-1-轮-2026-11-05-执行结果) |
| 周一双柔索版 | PASS | 用户指定的模型复用 `lumped_contact_proxy.py`、`macro_parcel_sim.py` 和 `unified_payload.py`。周一有 98 mm 悬停版和 250 g 居中复放，两者共享冻结力学。该模型的训练资格见下方 C0 检查。 [周一模型说明](../../../../reports/pe-yam-film-comparison-test.md) · [250 g 说明](../../../../reports/real-grasp-skirt-c-test.md) |
| 其他代理审阅 | FAIL | `effect_model.py` 以六个直接 tendon 表达外壳与每段内容物的关系，历史抓取路径还启用夹爪连接。分段内容物 v1.3 的历史适配器也有 `connect` 捏取。两个入口的既有抓取分支均无法提供要求的原生接触证据。 [载体](../../../../bag_sim/effect_model.py) · [分段内容物](../../../flex3d/segmented.py) |
| Yam_B 资产与工位 | PASS | 原生资产 `experience/yam_platform/assets/i2rt-yam-menagerie/upstream/yam_linear_4310.xml`，同时使用包内 STL。`yam_model.py` 加载机械臂与名义惯量，ABC 工作站从冻结的三个 Route B 配置装配。仓库当前原生入口通过 MJCF，标称 URDF 残差通道待确认。 [资产](../../../../experience/yam_platform/assets/i2rt-yam-menagerie/README.md) · [接口](../../../../experience/YAM_PURE_SIM.md) |
| Python 环境 | PASS | `/home/ubuntu/workspace/gzl/randomization/.venv-physical-parcel/bin/python`，Python 3.14.7、MuJoCo/MJWarp 4.0.1、Warp 1.19.0、Torch 2.15.0+cpu。`.venv/bin/python` 和 `.venv-continuum/bin/python` 可发现仿真模块，缺少 Torch。项目环境可用于原生 C0、MJWarp 仿真和 CPU 学习器。 |
| 多保真定位 | PASS | 本任务对代理预训练做门禁验证。后续路线为代理预训练、MJWarp Flex 零样本评估、高保真微调与从零训练对照、真机零样本部署。计划内的训练信号全部来自仿真。 |

## C0 本轮计划

在现有 `loose_scene()` 与 `LooseSim.wrap()` 上先测包覆与自由内芯受力，确认模型允许机器人经袋片接触传力后，再做完整 Yam_B Try 与入篮。试件使用同一个 350 × 250 mm 外袋，PE 工程先验，原生 MuJoCo、0.4 ms、零渲染。四种固定代表工况为松装左偏、松装右偏、中等居中和扁平满装。每种用原有可压缩关节配置、关闭压缩的对照各一次。前者是已有的最小物理修正，存在袋片之间的单向受拉限位。跟踪模型编译时内芯自由度、机器人碰撞位通路、初态包覆与接触法向承载、穿模、求解器告警及用户禁止的约束是否存在。采用袋体既有包覆方法，并加原生内芯与板片接触的独立复核。板片中心或内芯角点位于重建外表面之外时记录距离，原生接触重叠单独记录。试验从 `wrap()` 后的静止状态再走 0.20 s，每 40 ms 采样一次。内芯必须保持包覆，且无明显穿模、发散、直触或人工承载。仅当包覆与传力合格才适配 Yam_B 并验证四类工况的可部署 Try 和约十条真实入篮轨迹；存在可重复的模型结构失效则终止正式训练。

## C0 计划修订（用户指定周一版本）

用户排除薄板片袋体，并指定采用 2026-11-23 的四块内容物、双松弛柔索模型。原薄板片短试保存在 [`excluded-plates/`](excluded-plates/)，不计入所选模型的 C0。用户确认的物理模型为 Fix 4.1 冻结方程及周一 UMI 软包对照中的适配。98 mm 柔索来自 65 g 悬停验帧，250 g 动态对照使用同一方程，膜颈名义长度为 0，由原捕获方程确定接触参考长度。 [周一夹口分屏](../../2026-11-23/bag/umi-pe-yam-hold-r1/hold_comparison.png) · [周一 250 g 动态分屏](../../2026-11-23/bag/real-grasp-skirt-c-r2/grasp-visibility-r1/compare_real_grasp_skirt_c.mp4)

先复跑原版 65 g 和 250 g 各一次，仅观察每个 2 ms 子步的原始柔索向量、仿真支撑力、净空和捕获状态。65 g 用于核对用户选定的 98 mm 版本，250 g 用于指定质量范围内的诊断。每次使用周一归档的原关节或指垫运动边界，模型与动作保持不变。分别量化 8.0 s 悬停附近的柔索竖直净力/包裹重量，核对原有轨迹输出是否逐元素一致，保留静止阶段原生接触是否存在的源级证据。仅当所选版本有独立外袋接触与可检查的内芯传力路径、无需软件绑定夹爪，才进入完整 C0。存在结构性传力旁路时，结果作为 `surrogate approximation` 归档，停止后续正式训练。

## C0 结果

**FAIL。周一模型按原样无法通过接触传力门禁。** 两条轨迹在 `ubun` 用原生 MuJoCo 4.0.1 和 CPU Torch 复跑，数值均有限，轨迹数组与归档逐元素一致。7.5–8.5 s 的 500 个子步均离台，无台面支撑。 [诊断汇总](monday-rope-audit.json) · [复跑日志](monday-rope-audit.log) · [源与输入指纹](source_sha256.txt)

| 复放 | 重量（N） | 柔索平均竖直净力（N） | 柔索承重占比 | 峰值净空（mm） |
| --- | ---: | ---: | ---: | ---: |
| 周一 98 mm，65 g | 0.637650 | 0.637639 | 99.998% | 164.72 |
| 周一居中，250 g | 2.452500 | 2.452422 | 99.997% | 262.17 |

载荷由“指垫轨迹、代理聚拢/捕获、双柔索、四块内容物”传递。归档脚本的夹口法向力可由 `touch | previously` 与夹口宽度生成，持夹时无需独立的指垫—外袋接触。柔索末端通过 `distribute_anchor_forces()` 向四块内容物施力。按 C0 要求，这属于 `surrogate approximation`。这里存在可量化的直接代理承载，原生 `robot_payload_direct_force` 审计为 **NOT TESTED**。复放使用预录的机械臂运动边界，无法给出闭环机器人直触测量。 [250 g 夹口方程](../../2026-11-23/bag/real-grasp-skirt-c-r2/proxy_contact.py#L68) · [柔索到内容物的力分配](../../../../src/low_order_parcel/macro_parcel_sim.py#L285) · [双柔索方程](../../../../src/low_order_parcel/lumped_contact_proxy.py#L140)

四类工况的可部署安全 Try、25 Hz `H_try`、撤回 `q_hover`、原生提拉搬运入篮均为 **NOT TESTED**。冻结源码、Route B / Plan B 配置路径检查为 **PASS**。既有 12 项 `test_v3.py` 单测与同目录的 4 项 `test_yam_substep.py` 单测全部通过。 [冻结检查](frozen-check.log) · [单测日志](policy-v3-regression.log)

## 交付问题逐项结论

| 问题 | 状态 | 结果 |
| --- | --- | --- |
| 1. 最优 `num_envs` 与峰值显存 | NOT TESTED | C0 停止后没有启动 512–4096 路扩容。 |
| 2. 端到端有效宏回合吞吐 | NOT TESTED | 两条周一轨迹复放用于诊断受力。 |
| 3. C0 / C1 / C2 / C3 / C4 | FAIL | C0 传力检查失败。C1、C2、C3、C4 与最终留出评测均为 NOT TESTED。 |
| 4. Informed 对 Matched Blind 是否显著 | NOT TESTED | 无同初态匹配评测。 |
| 5. Learned Try 对 Fixed/Random Try 是否显著 | NOT TESTED | 无本轮训练策略与试探对照。 |
| 6. Effort 相对 w/o Effort 的额外收益 | NOT TESTED | 无本轮闭环消融。 |
| 7. 作弊、直触或代理载荷代偿 | FAIL | 周一双柔索直接向内容物承载，悬停近 100% 重量。原生直触指标为 NOT TESTED。 |
| 8. 可迁移到 MJWarp Flex 的权重与接口 | NOT TESTED | 本轮未生成权重或归一化统计量。既有 13D Try、11D 连续抓取和 25 Hz Yam_B 历史格式已在源码定位。跨保真兼容、FiLM 与新增候选无泄漏回归待通过 C0 后验证。 |
| 9. 主要失败及阻塞 | FAIL | 使用原周一物理模型时，独立接触无法接管其直接柔索承载。继续需为同一单向柔索方程提供可碰撞的独立外袋端点，并重新接受 C0，或由用户明确变更代理预训练的传力验收范围。 |

`summary.json` 记录测试状态和实测载荷。C0 后的模型检查点、数据摘要、学习曲线、策略对比图和合格入篮演示视频均为 **NOT TESTED**。现有周一分屏是模型预览。

## 复现

从仓库根目录执行，`out` 指向一个已创建的空目录。

```sh
run=test/runs/2026-11-26/yam-surrogate-v1-rl
flock /tmp/claude-gpu.lock env PYTHONPATH=src:experience:. \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-physical-parcel/bin/python -B "$run/audit_monday_rope.py" "$out"
env PYTHONPATH=src:experience:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-physical-parcel/bin/python -B -m unittest discover -s tests/policy_v3 -v
```

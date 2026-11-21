# 2026-11-21 试验记录

统一四块 XPBD 与 Stage 2 冻结代理，`ubun`，RTX 5090，PyTorch 2.11.0+cu130。版本为 2 ms 模板接入和 [统一 X 配置](../../../configs/payload_template_X.yaml)。执行命令见 [任务检查报告](../../../reports/unified-payload-template-test.md#step-3--接入与最终回归)。

| 运行 | 问题与结果 | 数据 |
| --- | --- | --- |
| unified-payload-proxy-r1 | 名义开环动作的异质完整回合。全部步进，稳定回合有效吞吐不足。 | [P6 日志](bag/unified-payload-proxy-r1/p6.log) |
| unified-payload-proxy-r2 | 增加夹爪位置反馈。完整回合的有效吞吐通过。 | [P6 日志](bag/unified-payload-proxy-r2/p6.log) |
| unified-payload-proxy-r3 | 最终模板接入回归。P1–P6、挂点虚功和既有 20 项整包测试通过。 | [CPU](bag/unified-payload-proxy-r3/cpu.log) · [GPU](bag/unified-payload-proxy-r3/cuda.log) · [整包](bag/unified-payload-proxy-r3/physical-parcel.log) · [验收](../../../docs/PAYLOAD_TEMPLATE_REPORT.md) |
| unified-payload-proxy-r4 | 补充初始余料的填充率上限，P1–P6 与挂点虚功复测通过。 | [CPU](bag/unified-payload-proxy-r4/cpu.log) · [GPU](bag/unified-payload-proxy-r4/cuda.log) |
| unified-payload-proxy-r5 | Stage 2.5 同袋分岔、侧壁推跑、预算接口与整批验收，`ubun`。 | [CPU](bag/unified-payload-proxy-r5/cpu.log) · [GPU](bag/unified-payload-proxy-r5/cuda.log) · [Stage 2](bag/unified-payload-proxy-r5/stage2-proxy.log) · [整包](bag/unified-payload-proxy-r5/physical-parcel.log) · [验收](../../../docs/PAYLOAD_TEMPLATE_REPORT.md#stage-25-同袋异物结果) |
| stage3a-dual-arm-r1 | `ubun`，RTX 5090。基于 `96a896c1` 双臂封装、独立 YAM 估力与虚拟 RGB，Step 1 通过。 | [命令与测试报告](../../../reports/stage3a-dual-arm-test.md) · [主题报告](../../low_order_parcel/STAGE34_REPORT.md#stage-3a-双臂-step-12026-11-21ubun) · [运行数据](policy/stage3a-dual-arm-r1/) |
| physics-realism-audit-r2 | Stage 1–2.5 冻结包裹物理审计，含共享 YAM 闭环。ubun，RTX 5090，基线 `96a896c1`。P6、三张相图及原回归跑完。捕获无源性失败，训练 NO-GO。 | [报告](../../../docs/PHYSICS_REALISM_AUDIT_REPORT.md) · [检查与命令](../../../reports/physics-realism-audit-test.md) · [数据](bag/physics-realism-audit-r2) |
| stage3b-corners-r1 | `ubun`，RTX 5090，基于 `85cb6f0e` 的三层分布与 C1–C10，Step 2 通过。 | [标准与命令](../../../reports/stage3b-corners-test.md) · [主题报告](../../low_order_parcel/STAGE34_REPORT.md) · [候选、日志与压力跑](policy/stage3b-corners-r1/) |
| parcel-passivity-fix-r1 | `ubun`，RTX 5090。从 `09a0901e` 修复捕获、侧接触与顶压能量，六组规定门禁通过，仿真训练条件准入。 | [计划与结论](../../../docs/PHYSICS_REALISM_AUDIT_REPORT.md#修复与复验结论) · [命令及日志](../../../reports/physics-realism-audit-test.md#修复后全量复验) · [数据](bag/parcel-passivity-fix-r1/) |
| yam-microlift-heldout-r1 | ubun，修复版 d6e519bc，三种控制规则 × 九个同袋留出配置。27/27 完成，三组均未达到 3 s 稳悬。 | [计划和实验报告](../../policy/PROBE_REPORT.md#配对结果) · [命令和检查](../../../reports/yam-microlift-heldout-test.md) · [归档](policy/yam-microlift-heldout-r1/) |
| parcel-pad-torsion-r1 | ubun，RTX 5090，d6e519bc 加有限指垫抗转。19 个能量工况、纯开环 P6、C1–C10、三张相图和回归完成，冻结供 Stage 4 仿真训练。 | [冻结结论](../../../docs/PHYSICS_REALISM_AUDIT_REPORT.md#fix-4-冻结结论) · [命令及测试证据](../../../reports/physics-realism-audit-test.md#fix-4-复验与冻结) · [数据](bag/parcel-pad-torsion-r1/) |
| yam-fix4-25mm-r1 / yam-fix4-sized-r1 | ubun，Fix 4，完整 YAM 两批各 27 回合。25/90 mm 最长稳悬分别 2.702/2.882 s，均未过 3 s。 | [计划、归因和结果](../../policy/PROBE_REPORT.md#原始失败归因与-fix-4-配对结果) · [25 mm](policy/yam-fix4-25mm-r1/) · [尺寸行程](policy/yam-fix4-sized-r1/) |
| yam-microlift-rgb-validation-r1 | ubun，Fix 4 与 90 mm 微提，八个新完整 YAM 对照检验 RGB/等效电流；2 完全承重、4 部分承重、2 空夹。 | [数据](policy/yam-microlift-rgb-validation-r1/) · [命令与检查](../../../reports/yam-microlift-heldout-test.md) |
| stage4-step1-r1 | ubun，RTX 5090，Fix 4 `046384ab`。两个宏步环境各 4,096 路，C10 退化拉拔、观测与 PPO 梯度门禁通过。 | [计划和结论](../../low_order_parcel/STAGE34_REPORT.md#stage-4-step-12026-11-21ubun) · [检查与命令](../../../reports/stage4-rl-step1-test.md) · [日志](policy/stage4-step1-r1/) |

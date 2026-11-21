# 2026-11-21 试验记录

统一四块 XPBD 与 Stage 2 冻结代理，`ubun`，RTX 5090，PyTorch 2.11.0+cu130。版本为 2 ms 模板接入和 [统一 X 配置](../../../configs/payload_template_X.yaml)。执行命令见 [任务检查报告](../../../reports/unified-payload-template-test.md#step-3--接入与最终回归)。

| 运行 | 问题与结果 | 数据 |
| --- | --- | --- |
| unified-payload-proxy-r1 | 名义开环动作的异质完整回合。全部步进，稳定回合有效吞吐不足。 | [P6 日志](bag/unified-payload-proxy-r1/p6.log) |
| unified-payload-proxy-r2 | 增加夹爪位置反馈。完整回合的有效吞吐通过。 | [P6 日志](bag/unified-payload-proxy-r2/p6.log) |
| unified-payload-proxy-r3 | 最终模板接入回归。P1–P6、挂点虚功和既有 20 项整包测试通过。 | [CPU](bag/unified-payload-proxy-r3/cpu.log) · [GPU](bag/unified-payload-proxy-r3/cuda.log) · [整包](bag/unified-payload-proxy-r3/physical-parcel.log) · [验收](../../../docs/PAYLOAD_TEMPLATE_REPORT.md) |
| unified-payload-proxy-r4 | 补充初始余料的填充率上限，P1–P6 与挂点虚功复测通过。 | [CPU](bag/unified-payload-proxy-r4/cpu.log) · [GPU](bag/unified-payload-proxy-r4/cuda.log) |
| unified-payload-proxy-r5 | Stage 2.5 同袋分岔、侧壁推跑、预算接口与整批验收，`ubun`。 | [CPU](bag/unified-payload-proxy-r5/cpu.log) · [GPU](bag/unified-payload-proxy-r5/cuda.log) · [Stage 2](bag/unified-payload-proxy-r5/stage2-proxy.log) · [整包](bag/unified-payload-proxy-r5/physical-parcel.log) · [验收](../../../docs/PAYLOAD_TEMPLATE_REPORT.md#stage-25-同袋异物结果) |
| physics-realism-audit-r2 | Stage 1–2.5 冻结包裹物理审计，含共享 YAM 闭环。ubun，RTX 5090，基线 `96a896c1`。P6、三张相图及原回归跑完。捕获无源性失败，训练 NO-GO。 | [报告](../../../docs/PHYSICS_REALISM_AUDIT_REPORT.md) · [检查与命令](../../../reports/physics-realism-audit-test.md) · [数据](bag/physics-realism-audit-r2) |

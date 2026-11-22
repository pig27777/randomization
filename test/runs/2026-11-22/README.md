# 2026-11-22 运行索引

| 运行 | 对象、问题与状态 | 记录 |
| --- | --- | --- |
| stage4-policy1-entry-r1 | ubun RTX 5090。Fix 4 与 Step 1 固定 25 mm，CUDA Graph 对照、入口和 CPU 回归通过。 | [计划](../../low_order_parcel/STAGE34_REPORT.md#stage-4-step-22026-11-22ubun) · [数据](policy/stage4-policy1-entry-r1/) |
| stage4-policy1-ours-s17 | ubun RTX 5090。Step 2 30 轮完成，留出成功 367/4,096，Step 3 熔断。 | [计划](../../low_order_parcel/STAGE34_REPORT.md#stage-4-step-22026-11-22ubun) · [检查](../../../reports/stage4-policy1-training-test.md) · [数据](policy/stage4-policy1-ours-s17/) |
| stage4-policy1-replay-r2 | ubun RTX 5090。原 4,096 路角点成功数完全复现，C3／C6 对照录像已播放。 | [检查](../../../reports/stage4-policy1-replay-test.md) · [视频与轨迹](policy/stage4-policy1-replay-r2/) |
| stage4-c6-sway-audit-r1 | ubun。复算同一 C3/C6 录像，C6 末段持续捕获且包裹与夹爪存在横向相对摆动。 | [报告](../../../reports/stage4-c6-sway-test.md) · [图与数据](policy/stage4-c6-sway-audit-r1/) |
| route-b-step0-r1 | ubun RTX 5090。Isaac Lab 原生 PhysX 与冻结包裹双向 CUDA 缓冲区检查通过。YAM 指垫几何与双臂、腕相机标定待补齐。 | [计划](../../isaaclab/REPORT.md#路线-b-step-02026-11-22ubun) · [检查](../../../reports/route-b-step0-test.md) · [日志](isaaclab/route-b-step0-r1/) |
| route-b-abc-scene-r1 | ubun RTX 5090。ABC 静态桌面与两台 I2RT YAM 的 2 ms 装配入口通过。 | [计划](../../isaaclab/REPORT.md#路线-b--abc-场景外参2026-11-22ubun) · [验收](../../../reports/route-b-abc-scene-test.md) · [日志](isaaclab/route-b-abc-scene-r1/) |
| route-b-fix41-r1 | ubun RTX 5090。Fix 4.1 功率共轭膜颈阻尼与默认 25 mm 视觉兼容。能量、固定轨迹、相图和回归全部通过。 | [计划](../../low_order_parcel/STAGE34_REPORT.md) · [冻结检查](../../../reports/route-b-fix41-test.md) · [日志](bag/route-b-fix41-r1/) |
| route-b-fix41-step0-r1 | ubun RTX 5090。冻结版 CUDA 复验、ABC 单臂 / 双臂场景复制与双臂空载滑轨扫程通过，最终场景已复验 1,024 路。 | [检查](../../../reports/route-b-abc-scene-test.md) · [日志](isaaclab/route-b-fix41-step0-r1/) |

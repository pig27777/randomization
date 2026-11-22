# 2026-11-22 运行索引

| 运行 | 对象、问题与状态 | 记录 |
| --- | --- | --- |
| stage4-policy1-entry-r1 | ubun RTX 5090。Fix 4 与 Step 1 固定 25 mm，CUDA Graph 对照、入口和 CPU 回归通过。 | [计划](../../low_order_parcel/STAGE34_REPORT.md#stage-4-step-22026-11-22ubun) · [数据](policy/stage4-policy1-entry-r1/) |
| stage4-policy1-ours-s17 | ubun RTX 5090。Step 2 30 轮完成，留出成功 367/4,096，Step 3 熔断。 | [计划](../../low_order_parcel/STAGE34_REPORT.md#stage-4-step-22026-11-22ubun) · [检查](../../../reports/stage4-policy1-training-test.md) · [数据](policy/stage4-policy1-ours-s17/) |
| stage4-policy1-replay-r2 | ubun RTX 5090。原 4,096 路角点成功数完全复现，C3／C6 对照录像已播放。 | [检查](../../../reports/stage4-policy1-replay-test.md) · [视频与轨迹](policy/stage4-policy1-replay-r2/) |
| stage4-c6-sway-audit-r1 | ubun。复算同一 C3/C6 录像，C6 末段持续捕获且包裹与夹爪存在横向相对摆动。 | [报告](../../../reports/stage4-c6-sway-test.md) · [图与数据](policy/stage4-c6-sway-audit-r1/) |
| stage4-main-fix4-merge-r1 | ubun RTX 5090。main 合流后八个角点的全部指标与 4,096 路盲测一致。 | [合流检查](../../../reports/stage4-main-merge-test.md) · [日志](policy/stage4-main-fix4-merge-r1/) |
| route-b-step0-r1 | ubun RTX 5090。Isaac Lab 原生 PhysX 与冻结包裹双向 CUDA 缓冲区检查通过。YAM 指垫几何与双臂、腕相机标定待补齐。 | [计划](../../isaaclab/REPORT.md#路线-b-step-02026-11-22ubun) · [检查](../../../reports/route-b-step0-test.md) · [日志](isaaclab/route-b-step0-r1/) |
| route-b-abc-scene-r1 | ubun RTX 5090。ABC 静态桌面与两台 I2RT YAM 的 2 ms 装配入口通过。 | [计划](../../isaaclab/REPORT.md#路线-b--abc-场景外参2026-11-22ubun) · [验收](../../../reports/route-b-abc-scene-test.md) · [日志](isaaclab/route-b-abc-scene-r1/) |
| route-b-fix41-r1 | ubun RTX 5090。Fix 4.1 功率共轭膜颈阻尼与默认 25 mm 视觉兼容。能量、固定轨迹、相图和回归全部通过。 | [计划](../../low_order_parcel/STAGE34_REPORT.md) · [冻结检查](../../../reports/route-b-fix41-test.md) · [日志](bag/route-b-fix41-r1/) |
| route-b-fix41-step0-r1 | ubun RTX 5090。冻结版 CUDA 复验、ABC 单臂 / 双臂场景复制与双臂空载滑轨扫程通过，最终场景已复验 1,024 路。 | [检查](../../../reports/route-b-abc-scene-test.md) · [日志](isaaclab/route-b-fix41-step0-r1/) |
| route-b-main-merge-r1 | ubun RTX 5090。Fix 4.1 与 ABC 场景并入 main，CPU/CUDA 回归和双臂场景加载通过。 | [合并检查](../../../reports/route-b-main-merge-test.md) · [日志](isaaclab/route-b-main-merge-r1/) |
| route-b-step1-r1 | ubun RTX 5090。Step 0 复验通过。真实滑轨运动下仅腕部回写导致子步端口功越限，Step 1 暂停。 | [门禁报告](../../../reports/route-b-step1-test.md) · [日志](isaaclab/route-b-step1-r1/) |
| route-b-step1-dual-port-r1 | ubun RTX 5090。双通道端口功和静态电流隔离通过，C6 真实关节入口轨迹跟踪未达标，暂停 Step 1。 | [门禁报告](../../../reports/route-b-step1-test.md#双通道续测2026-11-22-2255北京时间) · [轨迹和日志](isaaclab/route-b-step1-dual-port-r1/) |
| route-b-step1-tracking-r1 | ubun RTX 5090。重力前馈和指垫 IK 通过 35／38／41 mm 跟踪，C6 捕获仍未达标，完成三档吞吐与对照。 | [跟踪与吞吐报告](../../../reports/route-b-step1-test.md#重力前馈指垫-ik-与-step-1-余项) · [日志和视频](isaaclab/route-b-step1-tracking-r1/) |
| route-b-step1-workcell-r1 | ubun RTX 5090。右抓／中交／左放工位绑定，滑轨零缝标定与端口复验通过。右抓侧真实建压不足，Step 1 未放行。 | [门禁报告](../../../reports/route-b-step1-test.md#右抓中交左放与滑轨建压复验) · [数据](isaaclab/route-b-step1-workcell-r1/) |

# 2026-11-23 运行索引

| 运行 | 对象、问题与状态 | 记录 |
| --- | --- | --- |
| route-b-film-closure-r1 | ubun RTX 5090，Fix 4.1 Route B。标准动作的薄膜夹口静态预压、空载与双通道回归通过。C6 高入口聚拢不足，右抓门禁未放行。 | [命令与报告](../../../reports/route-b-step1-test.md#薄膜夹口静态承压与-30000-路显存复验) · [复验日志](isaaclab/route-b-film-closure-r1/) |
| route-b-30000-vram-r1 | ubun RTX 5090，同一固定显式 PD 重力前馈。30,000 路双臂空载通过，整卡采样峰值 15.00 GiB。 | [压测命令与结果](../../../reports/route-b-step1-test.md#30000-路显存) · [数据](isaaclab/route-b-30000-vram-r1/) |
| ppo-yam-pe-edge-r1 | ubun，YAM URDF PPO pilot 第 3 轮，C5 PE 袋和折叠衣物。指定边缘的检查点顶抓未离台；受控指垫提起后末帧倾角 52.90°。 | [复现与核验](../../../reports/ppo-pe-soft-edge-test.md) · [轨迹与录像](policy/ppo-yam-pe-edge-r1/) |
| route-b-loaded-fold-r1 | ubun RTX 5090，冻结 Fix 4.1 Route B。压实前侧膜承压、止挡聚拢与重包提拉复验。 | [测试报告与命令](../../../reports/route-b-step1-test.md#有载闭合重包提拉与-c5-界面) · [日志](isaaclab/route-b-loaded-fold-r1/) |
| route-b-c5-ui-r1 | ubun RTX 5090。修复前 C5 PhysX 界面，180 帧可解码，未形成捕获。 | [C5 视频](isaaclab/route-b-c5-ui-r1/c5-grasp.mp4) · [截图](isaaclab/route-b-c5-ui-r1/c5-close.png) |
| umi-skirt-c-r1 | ubun，原生 MuJoCo flex / Fix 4.1 四块代理。UMI 39 × 29 × 2 cm 厚裙同轨迹夹衣，稳悬单帧、边长和防穿模复核；待目视确认。 | [命令与报告](../../../reports/umi-skirt-c-test.md) · [同帧图](bag/umi-skirt-c-r1/skirt_c_8s.png) · [DR 审查](../../../reports/umi-four-modes-dr-test.md) |
| route-b-final-gates-r1 | ubun RTX 5090，Fix 4.1 与 Plan B 单臂 Yam_B。C5 同载荷未放行。端口复验、8,192 路空载和入篮终态逻辑通过。 | [命令与报告](../../../reports/route-b-plan-b-test.md) · [数据](isaaclab/route-b-final-gates-r1/plan-b-step1-summary.json) |
| umi-skirt-c-video-r1 | ubun，沿用 UMI 工况 C 同轨迹物理记录。完整 9.6 s 左右分屏通过逐帧边长和四块防穿模检查。 | [命令与报告](../../../reports/umi-skirt-c-test.md#正式分屏复核) · [视频](bag/umi-skirt-c-video-r1/compare_flex_vs_proxy_side_by_side.mp4) |
| route-b-c5-load-map-r1 | ubun RTX 5090，Fix 4.1 与 Plan B。分指载荷回放、顶压映射 A/B 与端口回归。C5 捕获 2/300，离台同步捕获 0。 | [逐步诊断及回归](../../../reports/route-b-c5-load-map-test.md) · [汇总](isaaclab/route-b-c5-load-map-r1/summary.json) |
| real-grasp-skirt-c-r2 | ubun，250 g 厚裙共享居中顶抓轨迹，原生双指摩擦滑脱，冻结代理稳悬。按要求直出。 | [报告](../../../reports/real-grasp-skirt-c-test.md) · [视频](bag/real-grasp-skirt-c-r2/compare_real_grasp_skirt_c.mp4) |
| route-b-bin-side-demos-r1 | ubun，Plan B 侧抓，种子 1。首批示范 0/8,192。定位接触归属与搬运偏差后 C6 小批完整入篮 1/2。 | [测试与复现](../../../reports/route-b-side-bc-test.md) · [首批结果](isaaclab/route-b-bin-side-demos-r1/s8192-side/result.json) |
| grasp-visibility-r1 | ubun，250 g 居中抓取原始记录，修复 PE 与 CAD 深度遮挡并核对三段夹口。 | [报告](../../../reports/real-grasp-skirt-c-test.md#夹口可见性复核) · [视频](bag/real-grasp-skirt-c-r2/grasp-visibility-r1/compare_real_grasp_skirt_c.mp4) |
| route-b-c5-press-then-close-r2 | ubun RTX 5090，C5 已开爪触袋，压力低于闭爪门槛。 | [两轮实测](isaaclab/route-b-c5-press-then-close-r2/summary.json) · [测试报告](../../../reports/route-b-c5-load-map-test.md#2012-对用户所看录像的复核) |

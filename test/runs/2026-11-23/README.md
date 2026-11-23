# 2026-11-23 运行索引

| 运行 | 对象、问题与状态 | 记录 |
| --- | --- | --- |
| route-b-film-closure-r1 | ubun RTX 5090，Fix 4.1 Route B。标准动作的薄膜夹口静态预压、空载与双通道回归通过。C6 高入口聚拢不足，右抓门禁未放行。 | [命令与报告](../../../reports/route-b-step1-test.md#薄膜夹口静态承压与-30000-路显存复验) · [复验日志](isaaclab/route-b-film-closure-r1/) |
| route-b-30000-vram-r1 | ubun RTX 5090，同一固定显式 PD 重力前馈。30,000 路双臂空载通过，整卡采样峰值 15.00 GiB。 | [压测命令与结果](../../../reports/route-b-step1-test.md#30000-路显存) · [数据](isaaclab/route-b-30000-vram-r1/) |
| ppo-yam-pe-edge-r1 | ubun，YAM URDF PPO pilot 第 3 轮，C5 PE 袋和折叠衣物。指定边缘的检查点顶抓未离台；受控指垫提起后末帧倾角 52.90°。 | [复现与核验](../../../reports/ppo-pe-soft-edge-test.md) · [轨迹与录像](policy/ppo-yam-pe-edge-r1/) |
| c5-pad-skin-r1 | ubun，Fix 4.1 Route B / PPO C5 配置。原穿透溯源、脚本下探、CAD 蒙皮间隙及同轨 MuJoCo flex 分屏核验通过；未捕获。 | [测试报告与命令](../../../reports/c5-pad-skin-test.md) · [数据与视频](isaaclab/c5-pad-skin-r1/) |

# 2026-11-26 测试索引

## RSS 2027 Exp A/B r1

35 × 25 cm PE Flex 袋和 200 g 内芯。主机 `ubun`，CPU 50 µs，参考夹具。正式入口和统计方法见报告。Exp A 135 回合，Exp B 216 回合，已完成。首次 B 接触对归类误报修正后原种子重跑。[报告](rss-exp-ab-r1/exp_ab_report.md) · [数据与图件](rss-exp-ab-r1/) · [测试报告](../../../reports/rss-exp-ab-test.md)

## RSS 2027 Exp A/B r2

35 × 25 cm PE Flex 袋，主机 `ubun`，50 µs CPU 参考夹具。复用原 Exp A 轨迹做左右对称性及观测噪声审计。Exp B 完成独立 108+108 回合，过压对角格 24/24 成功，选择与评测均无缺格。全局无效率 34/216，未通 5% 门禁。[本轮报告](rss-expAB-rework-r2/rework_report.md) · [测试报告](../../../reports/rss-expAB-rework-r2-test.md) · [结果目录](rss-expAB-rework-r2/)

## CPU Flex Policy V3 C1/C2 r1

35 × 25 cm PE 袋、200 g 内芯、50 µs CPU Flex 参考夹具。主机 `ubun`，22 个 CPU 进程。第五轮门禁通过。450 条正式采集、C2 训练、45 个独立状态的闭环已完成。[实验报告](cpu-flex-policy-v3-c1c2-r1/c1c2_report.md) · [测试报告](../../../reports/rss-cpu-flex-policy-v3-c1c2-test.md)


## CPU Flex Policy V3 C2 r2

主机 `ubun`，MuJoCo 4.0.1、50 µs CPU Flex，22 个单线程进程。复用 r1 C1 和 GroupSplit，通过 FiLM 匹配 RGB 门禁，完成 45 个独立状态的闭环，新增 52 个真实物理分支。训练与评测入口、冻结配置及论文结果见[实验报告](cpu-flex-policy-v3-c2-fix-r2/c2_fix_report.md)和[测试报告](../../../reports/rss-cpu-flex-policy-v3-c2-fix-test.md)。

## Yam_B GPU 求解器提速与捏膜提拉 r1

主机 `ubun`，RTX 5090，MuJoCo / MJWarp 4.0.1。800 µs 悬停达标，捏膜与同配置的三位置试探尚未放行训练。命令、接触复测和证据见[实验报告](yam-gpu-speedup-lift-r1/speedup_and_lift_report.md)与[测试记录](../../../reports/yam-gpu-speedup-lift-test.md)。

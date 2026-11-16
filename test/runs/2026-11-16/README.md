# 2026-11-16 运行索引

| 对象与版本 | 问题、机器与状态 | 报告与数据 |
| --- | --- | --- |
| 纸袋、塑料薄膜袋、气泡信封袋分类视频 | ubun，20:27–20:32，CPU；去掉数值标题，塑料两段复用，补录纸袋/气泡各一次；纸袋抬升21.92 cm，气泡抓取失败；视频完整解码通过 | [主题报告](../../bag_model/PHYSICAL_PARCEL_REPORT.md) · [任务检查](../../../reports/physical-parcel-yam-video-test.md) · [分类视频与结果](bag/physical-parcel-category-video-r1/) |
| 最外缘封口双膜片、随机化袋参数及刚体/三段内容物，MuJoCo4.0.1 | ubun，20:04–20:13，CPU；两次准备通过；侧抓抬升21.03 cm、悬挂2.74秒、carry窗口结束；顶面搓起失败；侧抓接触1.1972 mm超预算；视频已导出 | [主题报告](../../bag_model/PHYSICAL_PARCEL_REPORT.md) · [任务检查](../../../reports/physical-parcel-yam-video-test.md) · [随机化录像与配置](bag/physical-parcel-edge-sealed-randomized-r1/) · [封口回归](bag/physical-parcel-edge-sealed-checks-r1/) |
| 既有双膜片，MuJoCo4.0.1，面分离初始化及凸mesh接触 | ubun，13:40–14:08，CPU；r2下压几何超限停止，r3执行至lift接触丢失、grasp_fail；11项回归与凸mesh静置通过；视频已导出 | [主题报告](../../bag_model/PHYSICAL_PARCEL_REPORT.md) · [任务检查](../../../reports/physical-parcel-yam-video-test.md) · [box失败r2](bag/physical-parcel-mj4-yam-video-r2/) · [最终r3](bag/physical-parcel-mj4-yam-video-r3/) · [静置r3](bag/physical-parcel-mj4-face-clearance-r3/) |
| 最外缘封口双膜片，三档E/厚度，统一三段内容物 | ubun，CPU；三档完整顶面动作已录至松爪，均在起抬时抓取失败；4段视频完整解码通过 | [主题报告](../../bag_model/PHYSICAL_PARCEL_REPORT.md) · [任务检查](../../../reports/physical-parcel-top-parameter-video-test.md) · [原停止r1](bag/physical-parcel-top-parameters-r1/) · [完整动作r2](bag/physical-parcel-top-parameters-r2/) |

启动命令、配置与源码快照在各运行目录，录像复用统一YAM平台。动态相交深度与角点包含独立于原生接触点记录，方法和停止标准见主题报告。


## rub-contact-e1-r1：Rub-to-Pinch接触审计

- 对象/版本：袋装包裹physical_parcel，ac0a963a，142节点外缘封口基线；200节点存档作为待运行E1对照。机器ubun，2026-11-16（CST，最近时间见manifest），MuJoCo/MJWarp4.0.1。
- 问题/标准/配置：见[主题报告本轮计划](../../bag_model/PHYSICAL_PARCEL_REPORT.md#rub-to-pinch-phase-0--e12026-11-16ubun)；每配置一次，保持原材料和拓扑。
- 已执行：644帧存档运动学与几何复算、两份Python语法检查、20步CPU短入口。**Phase0审计完成，E1正式动力学受共享锁阻塞，未验证。**
- 命令、机器与源码指纹：[运行目录](bag/rub-contact-e1-r1/)、[manifest](bag/rub-contact-e1-r1/manifest.json)、[待执行E1命令](bag/rub-contact-e1-r1/planned-commands.json)。
- 报告：[主题报告](../../bag_model/PHYSICAL_PARCEL_REPORT.md)、[任务检查](../../../reports/rub-contact-e1-test.md)。

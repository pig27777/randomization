# 2026-11-16 运行索引

| 对象与版本 | 问题、机器与状态 | 报告与数据 |
| --- | --- | --- |
| 最外缘封口双膜片、随机化袋参数及刚体/三段内容物，MuJoCo4.0.1 | ubun，20:04–20:13，CPU；两次准备通过；侧抓抬升21.03 cm、悬挂2.74秒、carry窗口结束；顶面搓起失败；侧抓接触1.1972 mm超预算；视频已导出 | [主题报告](../../bag_model/PHYSICAL_PARCEL_REPORT.md) · [任务检查](../../../reports/physical-parcel-yam-video-test.md) · [随机化录像与配置](bag/physical-parcel-edge-sealed-randomized-r1/) · [封口回归](bag/physical-parcel-edge-sealed-checks-r1/) |
| 既有双膜片，MuJoCo4.0.1，面分离初始化及凸mesh接触 | ubun，13:40–14:08，CPU；r2下压几何超限停止，r3执行至lift接触丢失、grasp_fail；11项回归与凸mesh静置通过；视频已导出 | [主题报告](../../bag_model/PHYSICAL_PARCEL_REPORT.md) · [任务检查](../../../reports/physical-parcel-yam-video-test.md) · [box失败r2](bag/physical-parcel-mj4-yam-video-r2/) · [最终r3](bag/physical-parcel-mj4-yam-video-r3/) · [静置r3](bag/physical-parcel-mj4-face-clearance-r3/) |

启动命令、配置与源码快照在各运行目录，录像复用统一YAM平台。动态相交深度与角点包含独立于原生接触点记录，方法和停止标准见主题报告。

# 随机包裹模型任务检查

最近测试 2026-11-14T18:05+08:00，机器 ubun。

交付包含 GPU 随机内容物、三路径袋体、捕获与滑移力核，及共享 YAM 适配器、材料测试、曲线和视频。实现及范围见 [模型说明](../contents/parcel/README.md)，实测表和失败记录见 [主题报告](../test/content_model/parcel/REPORT.md)。

- 自动化命令 `flock /tmp/claude-gpu.lock /home/ubuntu/Dev/Envs/mjwarp/bin/python test/content_model/parcel/test_physics.py`，7 项通过。
- `validate.py` 的 fixtures、benchmark、stress 入口完成。N=1/64/512/2048/4096 均运行，CUDA Graph 开启。两拓扑各512环境30秒的 NaN/Inf 失败率均为0。
- `material_checks.py` 检查悬臂单调性与参数极值。`slip_fixture.py` 对照低摩擦8/8滑脱、高摩擦0/8脱落。半步长平均位移差最大0.141 mm。
- `yam_suite.py` 复用共享 YAM v2.2，六原型各一次，三松装成功、三满装因顶面材料不足而捕获失败。最大穿透3.604 mm。
- 实测曲线和 CPU 渲染的 YAM 视频已生成。静态代码检查通过记录随交付保留。

测试局限包括完整 YAM 的 GPU 批次、全 A–G 矩阵、全参数能量审计及实物标定尚未完成。静置性能数字不代表动态抓取性能。首轮仓库检查后未实现的记录已由本轮实际实现与实测结果更新，早期失败与修正保留在主题报告。

源码检查命令为 `python -m compileall -q contents/parcel test/content_model/parcel experience/yam_platform/protocol.py experience/yam_platform/render_video.py` 及对上述源码/任务文档的 `git diff --cached --check`，均通过。YAM 源码与详细 trace 已无损归档并检查，生成数据的原始 XML/SVG 格式保留。

## 2026-11-14 45°视频补充

执行共享YAM入口，`--backend parcel.yam_adapter --model-options '{"prototype":3,"seed":7}' --yaw 45 --tilt 0 --dt .0005`，正式运行持GPU锁。松装易弯原型水平45°完整脚本判定success，最大穿透1.447 mm，跟踪误差峰值42.348 mm。视频使用共享render_video入口和Mesa CPU渲染，输出在 [本轮目录](../test/runs/2026-11-14/bag/parcel-yam-yaw45-seed7-r1/)。单次演示未扩大其他原型或多seed验证。

视频检查完成，ffprobe时长17.10秒，已查看12.5秒帧确认显示yaw45°与提起状态，源快照归档完整性和逐文件字节比对通过。

## 2026-11-14 侧面抓取补充

新增适配器参数grasp_face=side和side_pitch_deg，通过共享approach接口接入，默认top保留。原型3、seed7、region1，dt0.5ms，ubun，正式运行持flock。纯水平侧向r1为setup_fail，初始位置误差4.006 mm。保留侧面抓取点，30°向下接近的r2执行4.0565秒，close阶段grasp_fail，未聚拢材料，闭合时无双指有效接触。最大穿透0.209 mm。

已生成并检查4.10秒的[侧向尝试视频](../test/runs/2026-11-14/bag/parcel-yam-side-pitch30-seed7-r2/yam.mp4)，原失败结果、配置、源码快照和轨迹保留。Config默认值、非法face/region/pitch/NaN边界检查以及compileall通过。该次只验证新增入口和实际失败过程，尚无侧面抓取成功结论。

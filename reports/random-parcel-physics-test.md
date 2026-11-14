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

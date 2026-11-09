# 分段 v1.3 接入统一平台：检查报告

最近检查：2026-11-09，机器 `ubun`。

结论：冻结分段内容物已通过适配器接入统一YAM平台；14次原任务结局为4 success、5 grasp_fail、5 setup_fail。四次任务完成均依赖顶面捏取效果，不能等同于原生摩擦抓取或完整物理验收。

- 范围：`test/flex3d/segmented.py`、集成检查及公共轨迹渲染。冻结模型、策略源码和公共任务判定未改；保留模型发布版0.4ms步长。
- 命令：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /tmp/claude-gpu.lock .venv/bin/python -B -m unittest discover -s test/flex3d -p test_segmented.py -v`：3项通过。初次连接测试误用了另一类实例，产生TypeError；改正夹具后通过，未放宽断言。
- 20ms短入口通过。正式矩阵命令与计划见[主题报告](../test/flex3d/REPORT.md#分段内容物-v13-接入本轮计划2026-11-09ubun)；[原始汇总](../test/runs/2026-11-09/contents/segmented-v13-yam-r1/matrix/summary.json)。9次动力学均无MuJoCo警告或模型失效。
- 限制：5次IK初始化失败；任务success跟踪峰值42–43mm，尚未核查全部姿态/接触质量和重复性。未做正式测速、GPU接入、κ覆盖、实物标定或训练。
- 视频：只回放保存状态，CPU Mesa llvmpipe渲染；首轮易压显示因原生tendon几何过多触及可视化容量，停止初版渲染并关闭tendon显示后重导，不改变轨迹/物理。完整视频与检查结果填回主题报告。
- 导出检查：两类45°视频均1280×480/20fps，318/284帧，完整ffmpeg解码通过；人工查看首帧和易压中段。[视频检查](../test/runs/2026-11-09/contents/segmented-v13-yam-r1/video-checks.json)。

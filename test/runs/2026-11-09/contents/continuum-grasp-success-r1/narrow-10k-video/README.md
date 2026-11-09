# 窄连续体失败与翻转回放

原始试次：[narrow-10k](../narrow-10k/)，原模型和原始轨迹不修改。本目录仅为回放派生资料，不推进动力学。实际X与原结果保留在源试次config/result；120×50×50 mm、density=150 kg/m³、实际质量0.045 kg、young=10000 Pa、poisson=0.1、damping=0.02、yield_strain=0.2、friction=0.8、box、yaw/tilt=0°/0°，辅助捏取=0。

源结果为 `sim_fail`、第7331步四面体翻转，GPU原始failure_bits=4，尚未进入夹持。源result的sim_time_s=3.664825678是最后安全CPU导出时刻。完整 `final_integration_state` 属于第一失败GPU步，CPU `mj_setState` 后提取qpos/time，实际时刻3.665325642 s；未执行CPU动力学。该失败状态只有单元ID43有负有符号体积比，J=-0.000772391803。数据与源SHA256见 [replay-metadata.json](replay-metadata.json)。

`render-only/` 复制源model.xml/config.json/result元数据，复制轨迹保存样本后追加最后安全qpos与真实第一失败qpos，final_qpos设为失败qpos；结果保留sim_fail并附上source元数据。它只是可回放派生记录，不能当成重新执行过的试次。

统一入口命令（仓库根目录，环境设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2`）：

```sh
.venv-continuum/bin/python -B test/flex3d/render_video.py test/runs/2026-11-09/contents/continuum-grasp-success-r1/narrow-10k-video/render-only test/runs/2026-11-09/contents/continuum-grasp-success-r1/narrow-10k-video/failure-raw.mp4 --fps 20
.venv-continuum/bin/python -B test/flex3d/render_video.py test/runs/2026-11-09/contents/continuum-grasp-success-r1/narrow-10k-video/render-only test/runs/2026-11-09/contents/continuum-grasp-success-r1/narrow-10k-video/inversion-raw.mp4 --fps 20 --tetra --focus
.venv/bin/python -B test/runs/2026-11-09/contents/continuum-grasp-success-r1/narrow-10k-video/label_failure.py
```

输出：观看目录 `contents-model-videos-r1/continuum-failed-narrow.mp4` 与 `continuum-tet-inversion.mp4`；同时复制 `docs/assets/` 同名文件。失败全片原速，诊断末0.8 s按10倍慢放并停末帧3 s；红色标负J单元和顶点。插值只用于显示，不是新物理样本。

四面体的有符号体积比J表示局部朝向和体积变化。J>0保留原朝向；J=0说明四点压到同一平面；J<0说明节点穿过对面、单元局部朝向反转。它不是整物旋转，当前有效性门槛在首次翻转时停止GPU。诊断视频不能用来声称失败后仍可继续物理仿真。

渲染/编码进程正常结束。用户要求“不用检查”，因此本轮没有额外ffprobe、全解码或关键帧QA。

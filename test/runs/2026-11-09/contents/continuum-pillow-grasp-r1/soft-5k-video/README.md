# 大软枕头 E5 kPa 失败录像

## 四面体版（外表面隐藏）

新增 [continuum-pillow-soft5k-tetra.mp4](../../contents-model-videos-r1/continuum-pillow-soft5k-tetra.mp4)。复用同一 `safe-render-only/` 记录，用统一入口 `--tetra --fps 20`；全景与局部均隐藏FLEXFACE/FLEXSKIN，不显示外表面，只画完整576四面体线框，保留机械臂；常规近景不截半。保持原速，没有慢放或停帧补时。失败与最后安全时间仍按原记录标注。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2 .venv-continuum/bin/python -B test/flex3d/render_video.py test/runs/2026-11-09/contents/continuum-pillow-grasp-r1/soft-5k-video/safe-render-only test/runs/2026-11-09/contents/continuum-pillow-grasp-r1/soft-5k-video/tetra-raw.mp4 --tetra --fps 20
.venv/bin/python -B test/runs/2026-11-09/contents/continuum-pillow-grasp-r1/soft-5k-video/label_tetra.py
```

该层只是显示外表面，并非给物体增加过独立袋膜；隐藏它不改变本体物理。按用户要求不额外QA，不修改任何网页或复制docs。

源试次：[soft-5k](../soft-5k/)，实际结果sim_fail。标称包络240×160×50 mm、实际质量0.29 kg、pillow，E=5000 Pa、ν=0.3、yield_strain=0.2、damping=0.02、density=251.3860204553869 kg/m³、friction=0.8、roundness=0.8、bulge=0.6、taper=0，yaw/tilt=0°/0°。189节点、576四面体、原平台GPU Newton、辅助捏取=0；原始轨迹和模型不修改。

第8743步、4.371451378 s，close阶段单元ID247翻转，J=-0.000252524410；最后安全CPU导出4.370951176 s。夹持峰0.626432 N、下压峰3.450896 N，未完成夹取。原 [failure-analysis.json](../soft-5k/failure-analysis.json) 保留失效状态说明与指标。

`safe-render-only/` 复制源model/config/result和安全轨迹，追加源final_qpos到其最后安全时间，避免丢失最后安全几何；不取失败GPU几何，不推进任何动力学。统一CPU Mesa回放，全景+局部，不改变相机或裁切放大。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2 .venv-continuum/bin/python -B test/flex3d/render_video.py test/runs/2026-11-09/contents/continuum-pillow-grasp-r1/soft-5k-video/safe-render-only test/runs/2026-11-09/contents/continuum-pillow-grasp-r1/soft-5k-video/safe-raw.mp4 --fps 20
.venv/bin/python -B test/runs/2026-11-09/contents/continuum-pillow-grasp-r1/soft-5k-video/label_video.py
```

最终：[continuum-pillow-soft5k-failed.mp4](../../contents-model-videos-r1/continuum-pillow-soft5k-failed.mp4)，元数据：[continuum-pillow-soft5k-failed.json](../../contents-model-videos-r1/continuum-pillow-soft5k-failed.json)。片内醒目标尺寸、实际质量、E和失败原因，只播放安全几何到终止。渲染/编码进程正常结束；按用户要求不额外执行ffprobe、解码或目检QA。未复制docs、未修改网页或旧片。

# 连续体原生接触成功录像

源试次：[narrow-100k](../narrow-100k/)。本目录仅登记实际success的视频交付，没有新运行、没有改源模型/轨迹/公共动作/阈值。

完整公共任务15.025556564 s，原生接触、辅助捏取=0，无辅助绑定。尺寸120×45×50 mm，质量40.5 g；E=100 kPa、ν=0.1、屈服应变0.2、阻尼0.02、密度150 kg/m³、摩擦0.8、roundness/bulge/taper均0，yaw/tilt均0°。60顶点/144四面体，MuJoCo/MJWarp4 GPU Newton，dt=0.5 ms。

事件：5.346846 s开始闭合；8.524630 s提起；11.226145 s搬运；13.300005 s放下；14.524832 s松开；15.025557 s任务success。无GPU失效/overflow，Jmin=0.759582，Newton最多7次，残差峰9.999877e-6；穿透峰0.882846 mm，机械臂跟踪误差峰40.5994 mm。塑性范数始终0，因此是弹塑性框架中的弹性响应演示，不是已经证明软塑性大物体抓取可靠。

命令（仓库根目录）：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2 .venv-continuum/bin/python -B test/flex3d/render_video.py test/runs/2026-11-09/contents/continuum-grasp-success-r1/narrow-100k test/runs/2026-11-09/contents/continuum-grasp-success-r1/continuum-native-success-raw.mp4 --fps 20
.venv/bin/python -B test/runs/2026-11-09/contents/continuum-grasp-success-r1/label_success.py
```

统一CPU Mesa回放，全景+近景，20 fps；右侧中心440×330裁切放大至640×480，让小尺寸目标更清楚，不改变物理状态。片内记录实际X、结果与局限。最终 [continuum-native-success.mp4](../../contents-model-videos-r1/continuum-native-success.mp4) 同时复制至 `docs/assets/continuum-native-success.mp4`，元数据 [continuum-native-success.json](../../contents-model-videos-r1/continuum-native-success.json)，原观看index已置首播放；失败片全部保留。

渲染/编码正常结束；按用户“不用检查”，不额外运行ffprobe、完整解码或关键帧QA。

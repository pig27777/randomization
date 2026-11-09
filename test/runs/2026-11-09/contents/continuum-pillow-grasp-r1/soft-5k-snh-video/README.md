# 实验性SNH大软枕头四面体录像

源试次：[soft-5k-snh](../soft-5k-snh/)。统一回放直接读取本目录权威 `model.mjb`，保留实验性SNH本构；不重新从XML恢复成StVK。没有新动力学试验或GPU占用。

240×160×50 mm标称包络，实际0.29 kg；E=5000 Pa、ν=0.3、yield_strain=0.2、damping=0.02、density=251.3860204553869 kg/m³、friction=0.8、roundness=0.8、bulge=0.6、taper=0，yaw/tilt=0°/0°。189节点、576四面体、实验性SNH、GPU Newton、dt0.5 ms、纯接触/辅助捏取=0。

实际完整轨迹7.938893318 s，提起阶段双指接触丢失超过50 ms，原结果grasp_fail，没有改标成功。源GPU无翻转/overflow/warnings，Jmin=0.432760776、接触穿透峰0.353942916 mm、末塑性范数0.00261082782、塑性范数采样峰0.00262527538、残差峰9.999868481e-6、Newton最多9次。单次无模型失效不表示已经可靠抓取或实物标定。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2 .venv-continuum/bin/python -B test/flex3d/render_video.py test/runs/2026-11-09/contents/continuum-pillow-grasp-r1/soft-5k-snh test/runs/2026-11-09/contents/continuum-pillow-grasp-r1/soft-5k-snh-video/tetra-raw.mp4 --tetra --fps 20
.venv/bin/python -B test/runs/2026-11-09/contents/continuum-pillow-grasp-r1/soft-5k-snh-video/label_video.py
```

全程原速、无focus/慢放/截半，外表面隐藏，只显示完整四面体线框，机械臂保留。全景+局部仍为统一相机，只对保存状态做显示插值。最终 [continuum-pillow-snh-tetra.mp4](../../contents-model-videos-r1/continuum-pillow-snh-tetra.mp4)，元数据 [continuum-pillow-snh-tetra.json](../../contents-model-videos-r1/continuum-pillow-snh-tetra.json)。

同时打包 `docs/assets/continuum-pillow-snh-tetra.mp4`、`continuum-pillow-soft5k-tetra.mp4`、`continuum-pillow-soft5k-failed.mp4`。原观看index增加大枕头片并保留旧成功/失败；docs/index正文由负责人维护。按用户要求不额外执行ffprobe、完整解码或目检QA，仅要求渲染与合成正常结束。

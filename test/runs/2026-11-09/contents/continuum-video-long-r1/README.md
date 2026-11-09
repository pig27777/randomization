# 连续体真实加长观察录像

本轮计划及结果登记在 [视频检查报告](../../../../../reports/contents-model-videos-test.md)。按用户“连续体的视频多录几秒”，从原用户软参数任务终止态继续真实 GPU Newton 动力学，保持所有原始参数与终止 ctrl；不是慢放或重复帧凑时长，不继续原抓取状态机，也不改判成功。

## 实际结果

源：[continuum-gpu-r1/ui/203244-ddf55a34/trial](../continuum-gpu-r1/ui/203244-ddf55a34/trial/)。240×160×50 mm 标称包络、0.190346 kg、rounded、yaw/tilt=40°/25°，用户自调软参数，未经材料标定。完整实际 X 见源 [settings.json](../continuum-gpu-r1/ui/203244-ddf55a34/settings.json) 及本轮 [soft/result.json](soft/result.json)。

原任务在5.413872719 s因双指接触丢失超过50 ms而 grasp_fail。完整 `mjSTATE_INTEGRATION`（含 PLASTIC、ctrl、warmstart 等）恢复最大绝对差为0；原塑性范数0.029177714055。保持保存的终止ctrl，追加9173步，到10.000318527 s（末步落在10秒后的一个0.5 ms积分格以内），真实新增4.586445808 s。全程GPU Newton，没有CPU动力学/CG回退，没有新抓取执行器。

追加观察段运行时：failure_bits=0，最小J=0.219504932，显式相对残差峰值9.998210042e-6，overflow=0，Newton最多9次迭代，末塑性范数0.029504901。约20 ms保存安全几何与完整状态，末步也已保存。没有触发失败，因此无需另补hard/box片。原始旧轨迹和视频保持不变。

新增单片：[`continuum-user-rounded-10s-observe.mp4`](../contents-model-videos-r1/continuum-user-rounded-10s-observe.mp4)，加在既有 [观看页](../contents-model-videos-r1/index.html) 的加长片区域。前5.414 s保留原任务，之后明确标注“失夹后保持姿态观察”，原grasp_fail不变。全部播放时间来源于真实状态推进，只在保存帧之间按统一入口插值。

## 命令与保存

机器ubun；环境`.venv-continuum`、MuJoCo/MJWarp4.0.0、Warp1.19.0。CPU只编译/恢复状态与几何回放。GPU运行持共享锁：

```sh
flock /tmp/claude-gpu.lock .venv-continuum/bin/python -B test/content_model/extend_continuum_recording.py test/runs/2026-11-09/contents/continuum-gpu-r1/ui/203244-ddf55a34/trial test/runs/2026-11-09/contents/continuum-video-long-r1/soft --until 10
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2 .venv-continuum/bin/python -B test/flex3d/render_video.py test/runs/2026-11-09/contents/continuum-video-long-r1/soft test/runs/2026-11-09/contents/continuum-video-long-r1/continuum-soft-10s-raw.mp4 --fps 20
```

生成的 [soft/trajectory.npz](soft/trajectory.npz) 合并原始轨迹、原终止完整状态和追加观察段；[soft/observation.json](soft/observation.json) 保存ctrl、源SHA256、恢复差、时长和健康数据，[soft/observation_trace.json](soft/observation_trace.json) 保存采样诊断，[soft/extension_source.py](soft/extension_source.py) 是实际执行源码快照。片内中文标签合成命令见 [label_video.py](label_video.py)。

用户随后要求“不用检查”，本轮没有额外执行ffprobe、完整解码或关键帧检查；仅保留仿真推进必需的逐步有效性判定与渲染进程的正常退出要求。运行数据不代表该材料或抓取可靠性验收通过。

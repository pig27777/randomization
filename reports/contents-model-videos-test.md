# 内容物模型视频检查

最近检查：2026-11-09 21:22（Asia/Shanghai），机器 `ubun`。结论：5 条最终视频与原生 HTML 播放入口通过检查；仅验证回放交付，没有新增动力学或物理验收。

## 范围与方法

复用统一平台保存的四条真实 YAM 轨迹，通过 `test/flex3d/render_video.py` CPU llvmpipe 回放，添加明确模型、参数、辅助捏取及失夹标签，制作原生接触近景并排。没有改模型、公共动作/判据或 UI，没有启动 GPU 任务。

```sh
.venv/bin/python -B test/runs/2026-11-09/contents/contents-model-videos-r1/render_and_check.py
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=codec_name,width,height,pix_fmt,nb_read_frames,r_frame_rate:format=duration -of json VIDEO.mp4
ffmpeg -v error -threads 1 -i VIDEO.mp4 -f null -
```

脚本对每条最终 MP4 执行 ffprobe 和完整解码，并抽取关键帧。另用 NumPy 检查四条源采样 qpos 与 final_qpos 均有限；用 HTMLParser 检查 `index.html` 的 10 个本地引用存在且没有外部 URL。核对源平台快照差异仅涉及后端刷新/状态记录，公共动作与失夹阈值未变。

## 结果

全部输出 H.264 / yuv420p / 1280×480 / 20 fps；渲染器均为 `llvmpipe (LLVM 20.1.2, 256 bits)`。

| 文件 | 时长 | 帧数 | ffprobe / 全解码 |
| --- | --- | --- | --- |
| segmented-pinch.mp4 | 16.00 s | 320 | 通过 |
| segmented-native.mp4 | 9.65 s | 193 | 通过 |
| continuum-hard-native.mp4 | 8.30 s | 166 | 通过 |
| continuum-user-rounded.mp4 | 5.45 s | 109 | 通过 |
| native-contact-comparison.mp4 | 9.65 s | 193 | 通过 |

关键帧已目检：分段 12.8 s 搬运、连续体用户 rounded 5.3 s、分段原生接触 9.5 s、并排 9.5 s。模型/条件标签可读，辅助捏取=1 用黄色醒目标注，并排短片尾部显示黄色“已停止”。分段辅助捏取片完成原始任务，其余三条在实际提起阶段失夹，不补造后续成功动作。

来源、配置、源码与数据 SHA256、完整 X、命令及提前终止说明见[本轮 README](../test/runs/2026-11-09/contents/contents-model-videos-r1/README.md)，自动文件检查数据见 [video-checks.json](../test/runs/2026-11-09/contents/contents-model-videos-r1/video-checks.json)，观看入口为 [index.html](../test/runs/2026-11-09/contents/contents-model-videos-r1/index.html)。

## 限制

原生并排保持标称尺寸、质量、角度与接触方式；材料未匹配，源动力学分别 CPU / GPU，步长 0.4 / 0.5 ms，不能作等材料或速度结论。用户 rounded 的 `compressible` 是未标定参数标签，质量 0.190346 kg、yaw/tilt=40°/25°，与原生并排配置不同。采样几何约每 20 ms 保存，末样本落后结果终点约 2.5–13 ms；视频显示在样本间插值，视频帧格不代表额外物理步进。并排连续体停止后保持末画面 1.35 s，片内明确标记。

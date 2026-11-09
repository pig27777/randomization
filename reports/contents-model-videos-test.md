# 内容物模型视频检查

## 追加计划：实验性SNH大软枕头全程四面体片

复用 `continuum-pillow-grasp-r1/soft-5k-snh` 的实际grasp_fail记录，统一CPU Mesa回放优先读取该试次权威 `model.mjb`。`--tetra --fps 20`，不focus、不慢放、不截半，隐藏外表面，保留机械臂。实际240×160×50 mm、0.29 kg、E5 kPa、ν0.3、屈服应变0.2，576四面体，GPU Newton源动力学、纯接触/辅助捏取=0；7.938893 s提起失夹，不改标成功。新片置原观看页首段，并将本片、首例四面体片与首例有表面失败片复制docs/assets；不改docs/index正文。按用户要求不额外QA。

结果：`continuum-pillow-snh-tetra.mp4` 已生成并置原观看页首段，渲染与编码正常结束。已复制docs/assets同名SNH片，以及 `continuum-pillow-soft5k-tetra.mp4`、`continuum-pillow-soft5k-failed.mp4`；未改docs/index正文。原成功与失败全部保留。本例GPU源记录无翻转/overflow/warnings、Jmin0.432761、穿透峰0.353943 mm、末塑性范数约0.00261083，结果仍为7.938893 s提起失夹。没有额外ffprobe、解码或目检QA。命令/来源见 [soft-5k-snh-video/README.md](../test/runs/2026-11-09/contents/continuum-pillow-grasp-r1/soft-5k-snh-video/README.md)。

## 追加计划：隐藏外表面的完整四面体版本

按用户要求，复用soft-5k的完整安全轨迹，使用负责人更新后的统一 `render_video.py --tetra --fps 20`：全景/局部均关闭FLEXFACE/FLEXSKIN，完整四面体网格不截半，保留机械臂。原速、无慢放、只到最后安全几何。新文件 `continuum-pillow-soft5k-tetra.mp4` 放原观看目录，保留大枕头240×160×50 mm、0.29 kg、E5 kPa、576四面体和4.37 s翻转失败标签。不改模型/网页，不占GPU；按用户要求不额外QA。

结果：四面体版已生成，统一渲染与标签编码进程正常结束。原8768观看目录可直接访问 `continuum-pillow-soft5k-tetra.mp4`，未改任何网页或复制docs。没有额外ffprobe、解码或目检QA；命令见 [soft-5k-video/README.md](../test/runs/2026-11-09/contents/continuum-pillow-grasp-r1/soft-5k-video/README.md)。

## 追加计划：240×160×50 mm、0.29 kg软枕头失败片

复用负责人已完成的 `continuum-pillow-grasp-r1/soft-5k`：E=5 kPa、ν=0.3、yield_strain=0.2，189节点/576四面体，纯接触/辅助捏取=0，close阶段单元247翻转。仅建立回放复制记录并追加源最后安全qpos（不取失败后几何、不推进动力学），用统一CPU Mesa全景+局部回放并标真实失败。新视频为 `contents-model-videos-r1/continuum-pillow-soft5k-failed.mp4`，不覆盖旧片、不改网页或复制docs；按用户要求不额外执行视频QA。

结果：新片已生成，仅显示至4.370951 s的安全几何，并明确标记4.371451 s、单元ID247翻转的原sim_fail结果。渲染/编码进程正常结束，无额外ffprobe、解码、目检QA；未复制docs或改网页。来源/命令见 [soft-5k-video/README.md](../test/runs/2026-11-09/contents/continuum-pillow-grasp-r1/soft-5k-video/README.md)。

## 追加结果：真实原生接触成功录像

复用负责人已完成的 `continuum-grasp-success-r1/narrow-100k` 实际success，按用户要求立即CPU Mesa回放，不新跑动力学。120×45×50 mm、40.5 g、E100 kPa、ν0.1、屈服应变0.2、damping0.02、friction0.8、yaw/tilt0°/0°，辅助捏取=0、无辅助绑定。原公共完整任务15.025556564 s完成提起、搬运、放下与松开。

已生成观看目录 `continuum-native-success.mp4` 并复制 `docs/assets/continuum-native-success.mp4`，原观看index已置首播放。右侧近景仅将中心440×330图像显示放大为640×480。片内标明该例plastic_norm=0，是弹塑性框架中的弹性响应演示；单个小尺寸成功不外推全域可靠。机械臂跟踪峰40.5994 mm单列；源GPU无失败/overflow、Jmin0.759582、残差峰9.999877e-6、Newton最多7次、穿透峰0.882846 mm。

渲染与标签编码进程均正常结束。按用户“不用检查”，没有额外ffprobe、解码或关键帧QA。来源/命令/事件见 [narrow-100k-video/README.md](../test/runs/2026-11-09/contents/continuum-grasp-success-r1/narrow-100k-video/README.md)。

## 追加计划：保留窄实体失败与翻转解释

按用户“失败的也要”与“四面体翻转给我解释是什么”，复用 `continuum-grasp-success-r1/narrow-10k`。仅CPU恢复第一失败步的完整integration state并提取实际qpos，不推进动力学；建立独立render-only复制记录，保留源最后安全几何并追加实际失败几何。用统一入口录制完整失败片及末0.8 s/10倍慢放/末帧保持3 s的四面体诊断片，标红负有符号体积单元。原试次不修改，状态仍为sim_fail。按用户“不用检查”不额外执行ffprobe、解码或关键帧QA。输出放观看目录并复制docs/assets，页面明确区别实际采样状态、显示插值与慢放。

结果：两片已编码并复制 `docs/assets/continuum-failed-narrow.mp4`、`docs/assets/continuum-tet-inversion.mp4`，原观看index已追加失败与解释。第一失败GPU时刻3.665325642 s，源结果最后安全时刻3.664825678 s；单元ID43的J=-0.000772392，与原GPU翻转失效一致。没有改原试次，没有CPU动力学，未额外做视频QA。回放来源、命令及metadata见 [narrow-10k-video/README.md](../test/runs/2026-11-09/contents/continuum-grasp-success-r1/narrow-10k-video/README.md)。

## 追加轮计划：真实延长连续体观察段

2026-11-09，复用用户 rounded 软参数试次 `continuum-gpu-r1/ui/203244-ddf55a34/trial`，从含 PLASTIC 与已保存 ctrl 的完整 `final_integration_state` 恢复，保留 5.413873 s 的 grasp_fail。保持终止控制目标，以现有 GPU Newton 动力学继续至总仿真时间 10 s；每约 20 ms 保存安全几何与完整状态。所有 GPU 步进持 `/tmp/claude-gpu.lock`，不修改平台、模型、判据、参数或 UI。若首次有效性失败，立即保留最早失败状态与最后安全几何，不转 CPU/CG。若软参数不能达到10 s，仅对 hard/box 的8.244 s完整终止态执行一次同类观察至12 s，不再挑参数。

通过标准：恢复完整 integration state，保留原失夹结果；新增物理时间真实推进，观察段明确标注“失夹后保持姿态观察”；采用统一 CPU Mesa 回放。用户随后要求“不用检查”，本轮跳过额外 ffprobe、全解码和关键帧检查，只保留推进必需的运行时有效性停止规则。结果写入 `continuum-video-long-r1/`，旧轨迹/视频不覆盖；观看入口增加新文件。状态：已完成真实10秒软参数片。

追加结果：完整integration state含PLASTIC恢复差0，从5.413872719 s推进9173步至10.000318527 s，真实增加4.586445808 s。保持原ctrl，未继续抓取状态机，原grasp_fail结果保留。追加段failure_bits=0、minJ=0.219504932、残差峰9.998210042e-6、overflow=0，未触发失效，不需要hard/box补录。渲染与标签编码进程正常结束；按用户要求未做额外视频检查。

新增 [真实10秒片](../test/runs/2026-11-09/contents/contents-model-videos-r1/continuum-user-rounded-10s-observe.mp4) 已加入原 [观看页](../test/runs/2026-11-09/contents/contents-model-videos-r1/index.html) 首段。数据、完整恢复/保持控制说明、命令和源码快照见 [continuum-video-long-r1/README.md](../test/runs/2026-11-09/contents/continuum-video-long-r1/README.md)。原轨迹、旧片、UI与共享报告索引均未修改。

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

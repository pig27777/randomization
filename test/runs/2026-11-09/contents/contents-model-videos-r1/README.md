# 分段与连续体内容物视频

## 本轮计划

问题：将已有完整 YAM 机械臂状态回放为可观看视频，展示分段与连续体内容物的真实运动、变形和停止事件。每个配置复用一次既有轨迹，不重新运行动力学、不占用 GPU、不改参数或判据。

选择：分段易弯 v1.3 的 0°/0° 默认捏取与纯接触各一条；连续体 hard/box 的 0°/0° 纯接触一条；用户已有易压 rounded 的 40°/25° 纯接触一条。已有结果失败时立即结束视频，不补造搬运。相同尺寸/质量/抓取角度的纯接触两段可制作并排，明确材料参数与步长不同，不能作为等材料或速度比较。

通过标准：统一平台 `test/flex3d/render_video.py` 只回放保存状态；20 fps、单段 1280×480（全景+近景）、同相机与编码；片内标签明确模型、条件和结果；所有 MP4 经 ffprobe 和完整解码验证，抽取并查看关键帧。若渲染或验证失败，保留错误并修复回放链路；不改模型或重挑成功参数。

机器：ubun。开始：2026-11-09 21:15，完成：2026-11-09 21:22（Asia/Shanghai）。状态：回放制作与文件检查通过；没有新增动力学试验。

## 输出与来源

直接观看入口：[index.html](index.html)。每段单片左为机械臂全景、右为跟随内容物的近景。主展示为分段辅助捏取完整任务、连续体用户软参数任务和原生接触并排。

| 视频 | 来源 | 真实事件 | 仿真终点 / 视频时长 |
| --- | --- | --- | --- |
| [分段易弯，辅助捏取=1](segmented-pinch.mp4) | [segmented-v13-yam-r1/matrix/trial_000](../segmented-v13-yam-r1/matrix/trial_000/) | success；完成提起、搬运、放下、松开 | 15.9448 / 16.00 s |
| [连续体 rounded，用户软参数](continuum-user-rounded.mp4) | [ui/203244-ddf55a34/trial](../continuum-gpu-r1/ui/203244-ddf55a34/trial/) | grasp_fail；5.413873 s 提起阶段失夹 | 5.413873 / 5.45 s |
| [原生接触并排](native-contact-comparison.mp4) | 下两条原生接触轨迹 | 左 9.586 s、右 8.244284 s 提起阶段失夹 | 9.65 s；右侧停止后保持末画面 1.35 s |
| [分段易弯，原生接触](segmented-native.mp4) | [segmented-v13-yam-r1/matrix/trial_012](../segmented-v13-yam-r1/matrix/trial_012/) | grasp_fail；双指接触丢失超过 50 ms | 9.586 / 9.65 s |
| [连续体 hard/box，原生接触](continuum-hard-native.mp4) | [grasp-behavior-refined/trial_000](../continuum-gpu-r1/grasp-behavior-refined/trial_000/) | grasp_fail；双指接触丢失超过 50 ms | 8.244284 / 8.30 s |

分段辅助捏取片的关键阶段：5.2664 s 接近结束；9.7188 s 开始提起；12.4196 s 开始搬运；14.4932 s 放下；15.4444 s 松开；15.9448 s 结束。9.7184 s 创建捏取连接，15.4444 s 解除连接。片头用黄色醒目标注“辅助捏取=1”，不能据此声称纯接触成功。

分段纯接触：9.1504 s 开始提起，9.586 s 失夹；连续体 hard/box：7.814343 s 开始提起，8.244284 s 失夹；连续体用户 rounded：5.363853 s 开始提起，5.413873 s 失夹。这里的“提起阶段”是公共状态机阶段，不代表内容物已经成功离台。

所有单片保留源记录从初始化至实际停止的全过程。视频时长与终点相差不足 1.3 个 20 fps 显示帧，来自末帧和编码帧格；没有补造后续成功动作。并排仅将较短的连续体片保持末画面，并明确显示黄色“已停止：此后保持末个保存画面”。已有分段成功记录长于 10 s；这是既有轨迹回放，没有启动新任务，也没有改变当前 UI 的 10 s 上限。

## 配置与版本

两条原生接触并排保持相同**标称包络 240×160×50 mm、质量 0.29 kg、yaw/tilt=0°/0°、pinch_coeff=0**、YAM 安装与公共动作、判据和相机。材料未匹配，积分步长与动力学实现不同：分段为 MuJoCo 3.17.0 CPU / 0.4 ms，连续体为 MuJoCo/MJWarp 4.0.0 GPU Newton / 0.5 ms。该片只展示各自真实动作，不支持等材料物理优劣或速度结论。

分段使用冻结 v1.3 易弯锚点，归档 SHA256 为 `940efb2f424e8920940687f1a313f93a09cbc64a5c81ff6df45ffc93edad415e`。完整 κ 见 [segmented-kappa.json](segmented-kappa.json)：6×3×2 段，横向/厚向压缩刚度 0.5/0.3 N/mm，压缩限位 0.2/0.2，droop_ratio=0.32，damping_ratio=0.5，surface_friction=0.8，无移动质量。两片只分别用原试次的 pinch_coeff=1 与 0。

连续体为 full-3D StVK/J2 弹塑性候选，60 顶点、144 四面体、864 个塑性状态量，Warp 1.19.0，Newton 残差门槛 1e-5、refinement_steps=2。下表是实际 X，不将类别名称视作材料标定结论：

| 实际 X | hard / box | 用户 rounded |
| --- | --- | --- |
| size (m) | [0.24, 0.16, 0.05] | [0.24, 0.16, 0.05] |
| density (kg/m³) | 151.04166666666666 | 151 |
| young (Pa) | 100000 | 3006.660569760869 |
| poisson | 0.25 | 0.05 |
| damping | 0.02 | 0.02 |
| yield_strain | 0.05 | 0.01995638838946398 |
| friction | 0.8 | 0.8 |
| roundness | 0 | 0.6 |
| bulge | 0 | 0.49 |
| taper | 0 | 0.1 |
| actual mass (kg) | 0.29000000000000004 | 0.1903462031330585 |
| yaw / tilt (deg) | 0 / 0 | 40 / 25 |
| pinch_coeff | 0 | 0 |

用户 rounded 的原始 [settings.json](../continuum-gpu-r1/ui/203244-ddf55a34/settings.json) 和 [result.json](../continuum-gpu-r1/ui/203244-ddf55a34/trial/result.json) 保留实际参数与结果。`compressible` 只是此次用户选择的参数标签，未经实物材料标定。圆角形状改变了实际填充体积，因此同包络、近似相同密度不等于相同质量。该片与 0° 分段片的角度、形状、质量和材料不同，不作并排等条件比较。

各源试次的完整 config、模型与平台源码指纹、原始文件 SHA256、渲染信息保存在 [manifest.json](manifest.json)。原始 `platform_source.py` 核对表明分段/连续体两版的差异是后端刷新及完整状态记录接入，公共阶段动作和失夹判据未改变。本轮没有修改源试次、模型、本构、阈值、动作或用户 UI。

## 回放与复现

统一入口从 `model.xml`、`trajectory.npz` 和 `result.json` 回放 qpos，只在保存状态之间插值，不调用仿真步进。渲染均为 Mesa `llvmpipe (LLVM 20.1.2, 256 bits)`，未占用 GPU。全景相机 lookat=[0,-0.02,0.20]，distance=1.35，azimuth=-45°，elevation=-25°；近景 distance=0.67，azimuth=-45°，elevation=-30°，跟随目标中心。并排取两条原生接触单片的右侧近景，保持同一相机尺度。

源状态约每 20 ms 采样。四条源 qpos 样本数依次为 798、271、480、413；末个保存几何状态分别落后结果终点约 4.4、13.0、5.6、2.5 ms。视频沿用统一入口的末个采样几何，不渲染不存在的失效后状态。视频标签时间在终点截断。`final_qpos` 和所有采样 qpos 均已检查为有限。

在仓库根目录执行以下命令；已存在的原始/最终视频会复用，复现至全新目录时需保留相对层级与本目录脚本。脚本没有动力学运行入口。

```sh
.venv/bin/python -B test/runs/2026-11-09/contents/contents-model-videos-r1/render_and_check.py
```

底层实际回放命令（`OUT=test/runs/2026-11-09/contents/contents-model-videos-r1`，环境统一设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2`）：

```sh
OUT=test/runs/2026-11-09/contents/contents-model-videos-r1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2
.venv/bin/python -B test/flex3d/render_video.py test/runs/2026-11-09/contents/segmented-v13-yam-r1/matrix/trial_000 "$OUT/raw/segmented-pinch.mp4" --fps 20
.venv/bin/python -B test/flex3d/render_video.py test/runs/2026-11-09/contents/segmented-v13-yam-r1/matrix/trial_012 "$OUT/raw/segmented-native.mp4" --fps 20
.venv-continuum/bin/python -B test/flex3d/render_video.py test/runs/2026-11-09/contents/continuum-gpu-r1/grasp-behavior-refined/trial_000 "$OUT/raw/continuum-hard-native.mp4" --fps 20
.venv-continuum/bin/python -B test/flex3d/render_video.py test/runs/2026-11-09/contents/continuum-gpu-r1/ui/203244-ddf55a34/trial "$OUT/raw/continuum-user-rounded.mp4" --fps 20
```

FFmpeg 添加中文条件与结果标签，修正统一入口对连续体继承的通用“3D flex”标题。正式视频均 H.264 / yuv420p / 1280×480 / 20 fps，编码 CRF=20；原始平台回放 CRF=22。NotoSansCJK 字体来自本机，不依赖外部资源。具体命令保存在 [render_and_check.py](render_and_check.py)，统一渲染源码快照为 [raw/render_video_source.py](raw/render_video_source.py)。`raw/` 仅为中间回放、日志与源码快照，请观看根目录中文标签视频。

## 验证与限制

5 条最终 MP4 全部通过 ffprobe、逐帧完整解码与帧率/尺寸检查；[video-checks.json](video-checks.json) 记录时长、帧数、SHA256 与关键帧。已目检 [分段搬运关键帧](keyframes/segmented-pinch.png)、[用户软形状关键帧](keyframes/continuum-user-rounded.png)、[原生接触并排末段](keyframes/native-contact-comparison.png) 和 [分段纯接触关键帧](keyframes/segmented-native.png)，标签可读、视角一致、停止标记可见。原生 HTML 播放入口的 10 个本地媒体/文档引用全部存在，无外部资源。

检查报告：[reports/contents-model-videos-test.md](../../../../../reports/contents-model-videos-test.md)。本轮是已有单次结果的视频交付；未新增重复性、材料标定、物理可靠性或吞吐验收结论。

# 仿真视频

YAM 快递堆视频为归档试次按原配置重放后的连续积分状态，30 Hz 采样、0.5 倍速展示；每段同时包含整机视角和抓取近景。停止后保留末帧 2 秒，并排视频较短的一侧继续暂停至另一侧结束。

建议先看三段对比：

| 视频 | 内容 |
| --- | --- |
| [书本 / 衣物](book_clothes_comparison.mp4) | 同外袋、同堆形、同抓点；书本单次成功，衣物抬升不足；结果尚无步长稳定性保证 |
| [书本步长对照](book_timestep_comparison.mp4) | dt=0.2 ms 成功与 dt=0.1 ms 穿透超限并列，保留反例 |
| [薄信封试抓](thin_envelope_comparison.mp4) | 装纸信封低堆顶上方抓取与装薄硬片信封高堆空隙侧夹；支撑与内容物均不同，不构成配对策略比较 |

独立视频：[书本](book.mp4)、[衣物](clothes.mp4)、[书本小步长](book_small_step.mp4)、[装纸信封](paper_envelope.mp4)、[薄硬片信封侧夹](card_envelope_side.mp4)。

模型尚未实物标定，薄壳接触存在步长敏感性。视频不代表真实机械臂效果；结果解释见[实验报告](../test/policy/HIDDEN_CONTENTS_REPORT.md)。密集状态、重放一致性、字幕及视频检查记录位于[来源档案](../policy/results/parcel_heap_20261109/continuation/videos/)。

## 冻结分段内容物：统一YAM平台

[易弯内容物45°抓取](../test/runs/2026-11-09/contents/segmented-v13-yam-r1/bendable-yaw45-clean.mp4) · [易压内容物45°抓取](../test/runs/2026-11-09/contents/segmented-v13-yam-r1/compressible-yaw45-clean.mp4)：冻结v1.3、原平台动作、pinch_coeff=1，使用动态掌部捏取连接。各为一次任务success的原轨迹回放，左右是机械臂和内容物近景，20fps/1倍速；不是原生摩擦承重证明。

14次矩阵为4任务success、5抓取失败、5初始化失败；成功样本跟踪峰值约42–43mm，未宣称完整物理验收。[实验报告](../test/flex3d/REPORT.md#分段内容物-v13-接入本轮计划2026-11-09ubun) · [帧数、CPU渲染与完整解码检查](../test/runs/2026-11-09/contents/segmented-v13-yam-r1/video-checks.json)。

### 改变大小与形状比例

- [小 / 中 / 大并排](../test/runs/2026-11-09/contents/segmented-size-shape-r1/videos/size-comparison.mp4)：180×120×37.5、240×160×50、288×192×60mm，同密度，质量随体积变化。
- [长条 / 扁平 / 厚块并排](../test/runs/2026-11-09/contents/segmented-size-shape-r1/videos/shape-comparison.mp4)：320×120×50、200×240×40、160×150×80mm，同体积、同质量约0.29kg。

实际重建几何并运行同一45°抓取，镜头距离/角度固定，12fps/1倍速；短的一侧结束后明确标注保持末帧。扁平包在合拢结束未满足捏取资格，其余五次为任务success；跟踪仍偏大，大包穿透峰1.065mm，未宣称物理验收通过。形状仅指长方体比例，未实现圆柱/圆枕。[实验/命令](../test/flex3d/REPORT.md#分段内容物尺寸与形状比例视频本轮计划2026-11-09ubun) · [八个MP4检查](../test/runs/2026-11-09/contents/segmented-size-shape-r1/videos/video-checks.json)。

## 薄膜与自由颗粒

[GPU 连续 5 秒落袋与重排](flex_granular_gpu5s.mp4)：64 个直径 16 mm 的自由球粒、98 个 2D flex 膜顶点，单环境 GPU 连续 10000 步，100 Hz 记录、25 fps/1 倍速回放。左为半透明袋膜，右为线框内视；无机械臂动作。

本轮无原始求解标志、未检测到颗粒出袋，接触峰值穿透 0.8654 mm。预设矩形鼓壳与理想拉伸膜未实物标定；视频只展示该模型的落袋与重排。[实验及来源](../test/flex_granular/REPORT.md) · [渲染记录](flex_granular_gpu5s.render.json) · [检查报告](../reports/flex-granular-video-test.md)。

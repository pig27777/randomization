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

## 薄膜与自由颗粒

[GPU 连续 5 秒落袋与重排](flex_granular_gpu5s.mp4)：64 个直径 16 mm 的自由球粒、98 个 2D flex 膜顶点，单环境 GPU 连续 10000 步，100 Hz 记录、25 fps/1 倍速回放。左为半透明袋膜，右为线框内视；无机械臂动作。

本轮无原始求解标志、未检测到颗粒出袋，接触峰值穿透 0.8654 mm。预设矩形鼓壳与理想拉伸膜未实物标定；视频只展示该模型的落袋与重排。[实验及来源](../test/flex_granular/REPORT.md) · [渲染记录](flex_granular_gpu5s.render.json) · [检查报告](../reports/flex-granular-video-test.md)。

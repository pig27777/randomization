# 完整YAM随机袋 GPU 运行汇总

机器：ubun。各GPU档3次同步计时，100步≈10ms。

| 袋型 / 内容物 | 阶段 | 夹持接触/承重 | 环境数 | 环境步/s | 整批步/ms | 单环境实时率 | 显存/MiB | 数值/物理有效 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |

| 混合组数 | 环境数 | 环境步/s | 整批步/ms | 单环境实时率 | 显存/MiB | 数值/物理有效 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |

数值有效要求有限状态、零overflow及求解未触上限；物理有效另要求几何深度≤1mm、无表面采样点外露、原生穿透≤1mm、求解未到上限。阶段测速固定真实采样状态和驱动指令，不是完整闭环任务吞吐。

| 试验 | 袋型 / 内容物 | 引擎 | 结果 | 仿真/s | 墙钟/s | 抬升/mm |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 0 | kraft / rigid-box | CPU shared full protocol | grasp_fail | 11.069 | 296.87 | 10.196 |
| 1 | pe / rigid-capsule | CPU shared full protocol | sim_fail | 0.040 | 0.93 | -0.000 |
| 2 | bubble / compressible-box | GPU physics + CPU shared controller | compute_limit | 14.492 | 1800.47 | 131.282 |
| 3 | kraft / rigid-sphere | GPU physics + CPU shared controller | grasp_fail | 12.480 | 2182.95 | 3.675 |
| 4 | pe / bendable-box | CPU shared full protocol | grasp_fail | 12.765 | 255.28 | 4.332 |
| 5 | bubble / rigid-cylinder | CPU shared full protocol | grasp_fail | 12.883 | 304.06 | 3.803 |

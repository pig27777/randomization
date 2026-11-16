# 完整YAM随机袋 GPU 运行汇总

机器：ubun。各GPU档3次同步计时，100步≈10ms。

| 袋型 / 内容物 | 阶段 | 夹持接触/承重 | 环境数 | 环境步/s | 整批步/ms | 单环境实时率 | 显存/MiB | 数值/物理有效 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| pe / compressible-box | close | True/False | 1 | 25.6 | 38.997 | 0.0026 | 1592 | True/False |
| pe / compressible-box | close | True/False | 1024 | 7100.3 | 144.218 | 0.0007 | 5210 | True/False |
| pe / compressible-box | close | True/False | 128 | 1934.0 | 66.185 | 0.0015 | 2042 | True/False |
| pe / compressible-box | close | True/False | 2048 | 7935.7 | 258.074 | 0.0004 | 8826 | True/False |
| pe / compressible-box | close | True/False | 32 | 552.8 | 57.891 | 0.0017 | 1690 | True/False |
| pe / compressible-box | close | True/False | 512 | 5093.0 | 100.530 | 0.0010 | 3386 | True/False |
| pe / compressible-box | close | True/False | 8 | 155.1 | 51.584 | 0.0019 | 1626 | True/False |
| kraft / rigid-capsule | close | True/False | 1 | 92.7 | 10.790 | 0.0093 | 1590 | True/True |
| kraft / rigid-capsule | close | True/False | 1024 | 25117.7 | 40.768 | 0.0025 | 5176 | True/True |
| kraft / rigid-capsule | close | True/False | 128 | 7843.4 | 16.319 | 0.0061 | 2040 | True/True |
| kraft / rigid-capsule | close | True/False | 2048 | 27669.0 | 74.018 | 0.0014 | 8792 | True/True |
| kraft / rigid-capsule | close | True/False | 32 | 2150.1 | 14.883 | 0.0067 | 1688 | True/True |
| kraft / rigid-capsule | close | True/False | 4096 | 29903.6 | 136.974 | 0.0007 | 16024 | True/True |
| kraft / rigid-capsule | close | True/False | 512 | 19068.2 | 26.851 | 0.0037 | 3384 | True/True |
| kraft / rigid-capsule | close | True/False | 8 | 607.6 | 13.166 | 0.0076 | 1592 | True/True |
| kraft / rigid-capsule | close | True/False | 8192 | 32716.5 | 250.394 | 0.0004 | 30520 | True/True |
| pe / bendable-box | approach | True/False | 8 | 147.1 | 54.391 | 0.0018 | 1622 | True/True |
| pe / compressible-box | approach | True/False | 8 | 152.6 | 52.436 | 0.0019 | 1626 | True/False |
| pe / compressible-box | close | True/False | 8 | 158.1 | 50.606 | 0.0020 | 1626 | True/False |
| pe / compressible-box | lift | True/False | 8 | 157.0 | 50.941 | 0.0020 | 1626 | True/False |
| bubble / rigid-sphere | approach | True/False | 8 | 723.9 | 11.051 | 0.0090 | 1592 | True/True |
| bubble / rigid-sphere | close | True/False | 8 | 941.7 | 8.495 | 0.0118 | 1592 | True/True |
| bubble / rigid-sphere | lift | False/False | 8 | 926.1 | 8.639 | 0.0116 | 1592 | True/True |
| bubble / rigid-cylinder | approach | True/False | 8 | 674.5 | 11.860 | 0.0084 | 1592 | True/True |
| bubble / compressible-box | approach | True/False | 8 | 461.2 | 17.346 | 0.0058 | 1626 | True/False |
| kraft / rigid-capsule | approach | True/False | 8 | 672.5 | 11.896 | 0.0084 | 1592 | True/True |
| kraft / rigid-capsule | close | False/False | 8 | 941.9 | 8.493 | 0.0118 | 1592 | True/True |
| kraft / rigid-capsule | lift | False/False | 8 | 616.6 | 12.974 | 0.0077 | 1592 | True/True |
| pe / rigid-box | approach | True/False | 8 | 167.7 | 47.704 | 0.0021 | 1592 | True/False |
| kraft / rigid-capsule | approach | True/False | 8 | 635.5 | 12.588 | 0.0079 | 1592 | True/True |
| kraft / rigid-capsule | close | True/False | 8 | 556.0 | 14.388 | 0.0070 | 1592 | True/True |
| kraft / rigid-capsule | lift | True/False | 8 | 641.6 | 12.468 | 0.0080 | 1592 | True/True |
| kraft / rigid-capsule | approach | True/False | 8 | 783.9 | 10.206 | 0.0098 | 1592 | True/True |
| kraft / rigid-capsule | close | True/False | 8 | 603.5 | 13.255 | 0.0075 | 1592 | True/True |
| kraft / rigid-capsule | lift | True/False | 8 | 547.9 | 14.602 | 0.0068 | 1592 | True/True |

| 混合组数 | 环境数 | 环境步/s | 整批步/ms | 单环境实时率 | 显存/MiB | 数值/物理有效 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 9 | 9 | 50.8 | 177.141 | 0.0006 | 1714 | True/False |
| 9 | 9216 | 13585.5 | 678.371 | 0.0001 | 12156 | True/False |
| 9 | 1152 | 3892.2 | 295.978 | 0.0003 | 3004 | True/False |
| 9 | 18432 | 14753.1 | 1249.361 | 0.0001 | 22620 | True/False |
| 9 | 288 | 1095.5 | 262.885 | 0.0004 | 2008 | True/False |
| 9 | 4608 | 9890.7 | 465.891 | 0.0002 | 6908 | True/False |
| 9 | 72 | 309.5 | 232.661 | 0.0004 | 1784 | True/False |
| 9 | 23040 | 15821.6 | 1456.238 | 0.0001 | 27868 | True/False |

数值有效要求有限状态、零overflow及求解未触上限；物理有效另要求几何深度≤1mm、无表面采样点外露、原生穿透≤1mm、求解未到上限。阶段测速固定真实采样状态和驱动指令，不是完整闭环任务吞吐。

| 试验 | 袋型 / 内容物 | 引擎 | 结果 | 仿真/s | 墙钟/s | 抬升/mm |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 0 | kraft / rigid-capsule | GPU physics + CPU shared controller | grasp_fail | 5.614 | 726.13 | 0.816 |
| 1 | pe / rigid-box | GPU physics + CPU shared controller | sim_fail | 2.260 | 1173.79 | 0.001 |
| 2 | bubble / rigid-sphere | GPU physics + CPU shared controller | grasp_fail | 5.985 | 662.44 | 0.075 |
| 3 | kraft / rigid-capsule | CPU shared full protocol | grasp_fail | 6.460 | 118.79 | 6.230 |
| 4 | pe / bendable-box | CPU shared full protocol | grasp_fail | 8.300 | 311.41 | 0.000 |
| 5 | bubble / rigid-cylinder | CPU shared full protocol | grasp_fail | 8.300 | 232.87 | 0.006 |
| 6 | kraft / rigid-capsule | CPU shared full protocol | grasp_fail | 6.165 | 115.02 | 2.624 |
| 7 | pe / compressible-box | CPU shared full protocol | grasp_fail | 7.698 | 310.04 | 0.345 |
| 8 | bubble / compressible-box | CPU shared full protocol | grasp_fail | 8.300 | 218.65 | 0.028 |

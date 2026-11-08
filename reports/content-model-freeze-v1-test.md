# 内容物冻结 v1 任务检查

最近检查：2026-11-05 17:56:38 UTC+8（Asia/Taipei）。

**结论：四项达到预设要求，CPU 速度未通过；未冻结。** 原始数据在 [freeze_v1/](../contents/segment/results/freeze_v1/)，完整论证在 [内容物报告“冻结 v1 验收”](../test/content_model/REPORT.md)。

| 范围 / 命令或方法（工作目录 contents/segment） | 结果 |
|---|---|
| `freeze_v1.py calibrate --run-id mapping_pass1 --all-segments --workers 8` | 420 点、363 可用；6 次确认越限全部排除；原始失败保留 |
| `freeze_v1.py m3 --run-id m3_pass1 --table results/freeze_v1/mapping_pass1/table.json --workers 8` | **12/18，未通过**，触发原计划内的局部校正 |
| `freeze_v1.py calibrate --run-id mapping_pass2 --table results/freeze_v1/mapping_pass1/table.json --categories compressible --mults .4 .6 .8 1 1.4 --workers 8` | 易压物 3 配置各 15 点，更新 3 条、保留其余 25 条 |
| `freeze_v1.py m3 --run-id m3_pass2 --table results/freeze_v1/mapping_pass2/table.json --workers 8` | **14/18，通过最低比例**；硬物通过，仍有 4 项超差/表外 |
| `python -B -m unittest discover -s tests -v` | **19/19，通过，7.712 s**：M1 10 项、原采样器 4 项、新接口 5 项；包括 10,000 组混合 κ、类别、RJ 几何/质量/惯量配对、表外拒绝、速度诊断分类 |
| `freeze_v1.py random --run-id exploration_random3 --count 3 --seed 20261104 --workers 3` | 小规模入口确认，0 失败/0 执行错误；不计入正式分母 |
| `freeze_v1.py random --run-id random200_seed20261105 --count 200 --seed 20261105 --workers 8` | **1/200 =0.5%，通过；执行错误 0**。65/71/64 个硬物/易弯物/易压物；速度诊断 3 例未单独判失败 |
| `freeze_v1.py evaluate --run-id evaluate10 --count 10` | 易弯物 20 N 中心横夹 **10/10 完整成功**，提起 99.34 mm，外排绕 x ≤0.1305°；I-22 通过 |
| 同一 evaluate 串行计时，每类 10 次全部计入 | 硬物 **1.286 s**、易弯物 **7.505 s**、易压物 **9.215 s**；**速度未通过 ≤8 s**。易压物 10 N 配置 10/10 `grasp_fail`，计时不代表完整成功搬运 |
| 每步能量统计开/关性能探查，物理输出逐项对比，cProfile | 物理输出相同；共享 CPU 负载变化，墙钟 16.27/22.95 s 未显示可采纳提速。**未修改默认设置**，不替代正式 30 次计时 |
| `calibrate.py --help`、全部 Python 源文件 `compile()` | 通过；旧标定 CLI 要求新 `--output-dir`，避免覆盖候选/历史表 |
| 最终运行时源码指纹、文档与产物链接核对 | 见下方交付静态检查 |

数值运行使用 `/home/ubuntu/Dev/Envs/mjwarp/bin/python -B`；正式实验均在 `flock /tmp/claude-gpu.lock` 下，后半段为同一持锁 bash 顺序执行，已退出释放锁。无包安装/升级、共享库修改或其他线代码变更。未做 MJWarp。

失败和修复记录：探索接口测试曾有 1 项失败，原因是指定类别校验选择的几何表与 `build()` 不同，已修复；原始 [失败日志](../contents/segment/results/freeze_v1/logs/exploration_interfaces.log) 和后续通过日志保留。最终正式单元检查没有删断言、跳过项目或放宽阈值。[正式日志目录](../contents/segment/results/freeze_v1/logs/)。

唯一正式随机失败 [case_086/result.json](../contents/segment/results/freeze_v1/random200_seed20261105/case_086/result.json) 包含完整 κ、阶段、qpos/qvel 和警告；t=6.430 s、R3 hold，z 压缩 **1.20550 倍限位**。[保存状态重建](../contents/segment/results/freeze_v1/random200_seed20261105/case_086/failure_analysis.json) 无轨迹重跑，最大速度 5.86 mm/s、警告 0。仍按失败计入，未排除后重算为 0%。

候选 [candidate.json](../contents/segment/results/freeze_v1/candidate.json) 记录 `frozen=false`、全源码 SHA256 与逐项状态；表版本 `mapping_pass2`，SHA256 `34f2d4f6e81a1f83873129b0932baa7d30001b0b5eae8699d3caaa473ee0dce8`。7 个运行时模块在最终 M3、随机 200、evaluate 三批与交付源码完全匹配。最后的 `calibrate.py` 改动仅保护 CLI 输出，采样导入的辅助函数未改；已做语法及 CLI 静态检查。

遗留：速度门槛未过；M3 仍有 4 项未命中/表外；1 个随机压缩失败；I-26 未关闭；易压物 10 N 中心夹取失败没有作因果归因。RJ 动态稳定、任意随机几何的响应精度、实物标定及 GPU 后端未由本轮验证。保留全部历史结果；不创建冻结标签或通知策略线已可开始 v1 训练。

交付静态检查：通过。候选全源码指纹、表 SHA256 与正式运行时模块核对一致；历史表 SHA256 未变；新增报告/进度的 22 处本地链接存在，200 份逐例结果和 30 份 evaluate 结果齐全，冻结协议源码与实际运行版本一致。记录见 [delivery_check.json](../contents/segment/results/freeze_v1/delivery_check.json)。

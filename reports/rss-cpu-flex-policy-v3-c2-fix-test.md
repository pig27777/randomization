# CPU Flex Policy V3 C2 r2 检查报告

最近检查为 2026-11-26 21:39 CST，主机 `ubun`。使用指定 PE Python、MuJoCo 4.0.1 与 PyTorch 2.15.0+cpu。

| 检查 | 执行方法 | 结果 |
| --- | --- | --- |
| C1 归档和划分 | r2 `run-r2.sh train`，对比归档回合标识与既有 `valid_bce_ids` | 450 条归档全部匹配，265／56／55 条有效，116 正样本 |
| 离线断言 | 重载三组权重，比对匹配 RGB；分别构造低于阈值与同候选三峰的输入检查拒绝 | 三态像素一致。Informed 分差 0.275187，峰值 g12／g11／g12，两个模拟失败条件均被拒绝 |
| 闭环及归档复放 | r2 `evaluate`；从三个归档重新解包后再次汇总并比较 JSON SHA-256 | 45 个状态、90 个旧闭环分支和 52 个新增闭环分支。冷缓存复放缺失 0，三份 JSON 字节一致 |
| Policy V3 原单测 | `PYTHONPATH=src:experience:. .venv-physical-parcel/bin/python -B -m unittest discover -s tests/policy_v3 -p test_v3.py -v` | 12/12 通过 |
| 冻结文件 | r1 `check-frozen.sh` 和 r1 归档路径的 `git diff --exit-code` | 四个基线与 r1 文件的路径级检查均退出 0 |
| 图件 | `pdfinfo`、`pdfimages -list`、PNG 像素与 DPI 读回、PDF 渲染预览 | PDF 单页、无栅格图对象，PNG 3217 × 1867、450 dpi |

首轮 C2 的 0.075136 跨态分差达到下限，三态同选 g9，门禁阻止物理启动。最终匹配门禁通过后执行 52 个原生物理分支。Center 的 Informed 为 4/6，Blind 为 0/0；28 个共同有效配对的差为 −3.6 个百分点，95% CI [−11.1, 0.0]。全样本未知界限跨零。

[实验报告](../test/runs/2026-11-26/cpu-flex-policy-v3-c2-fix-r2/c2_fix_report.md) · [核验日志](../test/runs/2026-11-26/cpu-flex-policy-v3-c2-fix-r2/verification.log) · [冻结检查](../test/runs/2026-11-26/cpu-flex-policy-v3-c2-fix-r2/frozen-check.log) · [Policy V3 单测](../test/runs/2026-11-26/cpu-flex-policy-v3-c2-fix-r2/policy-v3-regression.log)

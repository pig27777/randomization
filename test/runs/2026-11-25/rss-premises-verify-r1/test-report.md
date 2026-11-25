# RSS 前提验证测试报告

2026-11-25，中国标准时间。机器 `ubun`，RTX 5090，指定 IsaacLab v3.0.0 原生环境。运行入口均通过绝对路径 `python.sh -B` 并持 `/tmp/claude-gpu.lock`。

## 命令与证据

从仓库根目录执行。以下命令用于新目录复现，归档输出目录已存在。

```bash
D=test/runs/2026-11-25/rss-premises-verify-r1
"$D/run.sh" --cases 1A --output "$PWD/$D/entry-repro"
"$D/run.sh" --cases 1B 1C-empty 1C-500 1C-1000 --output "$PWD/$D/loaded-preflight-repro"
"$D/run.sh" --cases 1A 1B --grasp --seconds 4.20 --output "$PWD/$D/grasp-reset-repro"
"$D/run.sh" --cases 1A --grasp --film-mass-g 10 --seconds 4.20 --output "$PWD/$D/film-10g-repro"
"$D/run.sh" --cases 1B --grasp --height-retry --through-approach --film-mass-g 10 --seconds 4.88 --output "$PWD/$D/through-10g-repro"
"$D/run.sh" --cases 1C-empty 1C-500 1C-1000 --grasp --film-mass-g 10 --seconds 4.60 --output "$PWD/$D/joint-10g-repro"
MPLBACKEND=Agg /home/ubuntu/workspace/czx/system0/sim/python.sh -B "$D/summarize.py"
bash -n "$D/run.sh" "$D/check-frozen.sh"
/home/ubuntu/workspace/czx/system0/sim/python.sh -B "$D/verify.py"
"$D/check-frozen.sh"
```

旧质量工况的五组 120 ms 静置预检通过。首轮抓取被启动时的 Newton 关节复位污染。保留原始日志以复现该问题，修正后断言低位抓取初态与计划一致。旧 1C 的首个空袋已完成，后续入口因运行中编辑脚本导致 Isaac Lab 的 lambda 源码自省失败，改为具名碰撞管线后完成两档带载试次。几何图首次生成发现坐标布置的数组解包错误，修正后可导出 PNG/PDF。[旧质量与 10 g 试次索引](stage1_audit.json)

## 最终读回

最近检查时间为 2026-11-25 20:38 CST。20 对 JSON/NPZ 原始数据互相匹配，均通过严格 JSON 读回、模拟帧有限性、网格形状及索引校验。五个 10 g 工况验证了求解器节点质量、与旧工况完全相同的参考顶点和控制目标。1C 的三组 `q_des` 同为 `116 × 8`，SHA-256 为 `c572e4b5bd73c5bce1a9890f52ddc3e8c38555ed7bd9b3780eabeb50ed3057c9`。

| 检查范围 | 结果 |
| --- | --- |
| 10 g 物理复测 | 1A 未离台。1B 在 0.9720 s 几何侵入。1C 三档均无有效悬空窗口 |
| 图表读回 | 两幅单页 PDF 与两张 2520 px 宽的 PNG 通过 |
| 静态检查 | Python AST 和 Bash 语法通过 |
| 冻结源与配置 | `2ef4a394` 的接触与载荷源、`736db9b2` 的 YAM 力端口及本轮前后的 `configs/` 均通过路径级 `git diff --exit-code` |
| 下游限制 | Exp A/B 按 Stage 1 门禁未启用。未启动任何策略训练 |

[主报告](report.md) · [检查输出](validation.log) · [原始数据索引](stage1_audit.json)

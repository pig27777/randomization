# Policy V3 C1/C2 · CPU Flex 任务检查

主机 `ubun`，50 µs CPU Flex，PE 参考夹具。使用指定 PE Python，正式采样每进程 OpenMP、OpenBLAS、MKL 各 1 线程。有关试探、候选、门槛及恢复计划见 [实验报告](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/c1c2_report.md)。

| 检查 | 方法 | 当前结果 |
| --- | --- | --- |
| Policy V3 原单测 | `PYTHONPATH=src:experience:. .venv-physical-parcel/bin/python -B -m unittest discover -s tests/policy_v3 -v` | 12/12 通过 |
| 信息边界与回放 | 实测同扰动的 320 × 240 RGB 哈希、试探归档转 V3，检查 V0_ONLY 与 BCE 有效行 | RGB 三状态逐位相同，归档及无效剔除通过 |
| 物理检查点 | `mjSTATE_INTEGRATION` 恢复后按字节比较 | 通过 |
| 首轮连续门槛 | 30 回合，22 CPU 进程，2 N 探头限幅及 4 N 抓取 | 11/30 物理有效，0/11 成功。未通过 |
| 冻结路径与执行器版本 | `bash test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/check-frozen.sh` | 四引用 exit 0，Exp A/B 已提交执行器快照一致 |

[回归日志](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/policy-v3-regression.log) · [接口核对](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/interface-check.log) · [冻结日志](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/frozen-check.log)

首轮保存的 post-Try 状态复放原 55 mm 悬停，在抓取阶段 1.685 s 检出停放探头 0.423 N 接触反力。确认扫掠干涉后，将中立位移至 230 mm 并用独立种子复验。一次未固定适配器的尝试在产生数据前停止，复放改用 `94d44457` 源码快照。[原工装干涉复现](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/hover-interference-repro.log)

第二轮完成 30 回合，17 个物理有效，3 次成功。有效率 56.7% 未过门槛，正样本率 17.6%。五次松爪附近的停放探头接触保持无效，继续采用相同安全标准复验侧向停放。[第二轮摘要](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/pilot-2/summary.json)

第三轮有 22/30 物理有效及 8 次成功，三类均有成功，停放探头全过程零反力。边长比超限 5、指垫侵入 3，物理有效率 73.3% 未过标准。用未通过摘要尝试正式入口时命令退出 2，未创建全量数据目录。[第三轮摘要](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/pilot-3/summary.json) · [门禁拒绝](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/gate-rejection.log)

第四轮 21/30 物理有效与 4 次成功。三条右位长扫触发硬上限，另六条抓取的膜边长比超限。用三条原种子复放较浅的长扫和软回退，全部安全撤回，峰值 1.983–2.422 N。正式候选配额验证为每状态 150 个新种子，三状态相同配额，12 候选均有分配。[第四轮摘要](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/pilot-4/summary.json) · [力护栏回放](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/probe-guard-replay.log) · [配额](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/sampling-balance-check.log)

第五轮在原门槛、侧向停放、长扫 +3 mm、2 N 软回退下通过。27/30 物理有效（90.0%），10/27 成功（37.0%），3 个无效均为膜边长比。150 条入口归档及 V3 转换逐条读回通过，正式 450 条已启动。[第五轮摘要](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/pilot-5/summary.json) · [五轮验证](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/verify-pilots.log)

正式 C1 采齐 450 回合，376 个物理有效、116 次成功。V3 GroupSplit 拆分 315/68/67，有效 BCE 行 265/56/55。第一次 checkpoint 命名因去力标签含斜线中断。修复路径后逐条比较既有归档与原始包，安全重启固定训练并得到三组权重。原回合数据沿用。[正式日志](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/formal.log)

## 最终闭环与图件核验

最近核验为 2026-11-26 07:16 CST。读回 450 条正式数据、三组 CPU 权重和互斥划分。评分示例的三个 RGB 数组逐位相同，三条实测试探历史有物理反力差异。267 个实际分支齐全，含独立 45 个状态的 135 个抓取执行与 C2 排序的 132 个补采执行。检查点指纹、成功判据、空无效标签和配对统计全部通过。21/45 个双策略物理有效对用于条件净差，固定与 Blind 在中央状态无有效回合，状态等权基线保留 NA。四条冻结路径核验退出 0。主图经视检，PDF 为单页矢量，PNG 为 5002 × 2369、450 dpi。[最终核验日志](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/verify-final.log) · [冻结记录](../test/runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/frozen-check.log)

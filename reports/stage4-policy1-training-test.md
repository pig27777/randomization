# Stage 4 Step 2 测试报告

机器 `ubun`，RTX 5090，PyTorch 2.11.0+cu130，float32，2 ms。训练分支 `stage4/fix4-training`。Step 1 交付 `2fcc183d` 的环境、PPO 和单测在 `046384ab` 上重放为 `d7693078`。正式训练入口提交为 `5d42a43b`。

| 检查 | 结果 |
| --- | --- |
| 冻结基线 | 物理、有限指垫、YAM、RGB、DR 及全部配置与 `046384ab` 的 Git 差异为空。训练分支祖先排除 `554f335d`，固定微提拉 25 mm。 |
| CUDA Graph | 8 路固定动作与同种子逐步执行的观察、结果和终态按 1e-6 容差一致。入口耗时分别 14.90 s 和 65.98 s。 |
| 入口 | 32 路训练、PPO 更新、检查点、16 路验证和两组 16 路留出评估完成。首次工作树依赖链接缺少虚拟环境配置，补齐原环境的 `pyvenv.cfg` 和 `lib` 后复测通过。 |
| 冻结接口回归 | CPU 定向检查 5 项通过，GPU 专属用例由正式训练验证。C10 保持 6 N／130 ms 协议，双臂估力在 CPU 复测中连续 7 帧超过 3.5 N。 |
| 正式种子 17 | 4,096 并发，30 轮，按 6／9／15 轮执行课程。验证成功率择优后用 4,096 路独立全宽参数与 4,096 路固定角点盲测。 |

以训练分支为工作目录，使用项目原有虚拟环境。正式命令和日志如下。[入口与回归日志](../test/runs/2026-11-22/policy/stage4-policy1-entry-r1/) · [正式运行日志](../test/runs/2026-11-22/policy/stage4-policy1-ours-s17/) · [验收计划](../test/low_order_parcel/STAGE34_REPORT.md#stage-4-step-22026-11-22ubun)

```sh
git diff --exit-code 046384ab -- src/low_order_parcel/lumped_contact_proxy.py \
  src/low_order_parcel/unified_payload.py src/low_order_parcel/macro_parcel_sim.py \
  src/low_order_parcel/dual_arm_parcel_sim.py
export PYTHONPATH="$PWD/.venv-physical-parcel/stage3-cu130:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B \
  scripts/train_stage4_ppo.py --worlds 4096 --updates 30 --seed 17 \
  --output test/runs/2026-11-22/policy/stage4-policy1-ours-s17/training
```

## 正式复验结果

2026-11-22 19:01（北京时间）。**Step 2 完成，Step 3 熔断。** 全宽盲测 8.96%（367/4,096），低于 80% 门槛。硬书路由通过，C1、C6、C7 未达到每角点 80% 的标准。种子 17 共 30 轮，按验证集择优保留第 30 轮权重。

| 全宽盲测指标 | 结果 |
| --- | --- |
| 硬书首选侧抓 | 100.00%（1,447/1,447） |
| 无侧向遮挡的软衣和枕头首选顶抓 | 0.00%（0/989） |
| 有侧向遮挡的软衣和枕头首选顶抓 | 0.00%（0/409） |
| 固定 25 mm 微提拉末帧净空 | 4,096/4,096 为 0 mm，所有分位数均为 0 mm |
| 虚夹持或单侧滑移选择增力／重试 | 0.00%（0/1,838） |
| 虚夹持或单侧滑移最终补救成功 | 0.00%（0/1,838） |
| 掉包、撕裂、平均重试 | 5.98%（245/4,096）、0.00%（0/4,096）、0.00 |

| 角点 | 盲测成功率 | 掉包数 |
| --- | --- | --- |
| C1 | 0.00%（0/512） | 0 |
| C2 | 100.00%（512/512） | 0 |
| C3 | 100.00%（512/512） | 0 |
| C4 | 100.00%（512/512） | 0 |
| C5 | 100.00%（512/512） | 0 |
| C6 | 2.34%（12/512） | 3 |
| C7 | 0.00%（0/512） | 0 |
| C8 | 100.00%（512/512） | 0 |

C6 是 65 g、低台面摩擦与最坏硬件组合。512 路首选侧抓，Phase B 虚夹持或单侧滑移标记为 0，最终仅 12 路稳悬，判定为掉包的有 3 路。全宽样本同时出现侧抓偏置、末帧净空为零和补救动作缺失。各因素的独立影响需通过对照试验区分。

| 轮次 | 阶段 | 训练成功率 | 独立全宽验证成功率 |
| --- | --- | --- | --- |
| 1 | A | 51.73%（2,119/4,096） | — |
| 6 | A | 53.88%（2,207/4,096） | — |
| 7 | B | 35.45%（1,452/4,096） | — |
| 15 | B | 50.88%（2,084/4,096） | — |
| 20 | C | 43.53%（1,783/4,096） | 6.54%（67/1,024） |
| 25 | C | 22.29%（913/4,096） | 7.71%（79/1,024） |
| 30 | C | 7.91%（324/4,096） | 8.98%（92/1,024） |

第 20 轮从实际非空分桶中选中松散填充，其成功率为 1,184/3,069，之后使用 2 倍采样权重。训练与 PPO 更新共 26.37 分钟（0.440 GPU 小时），包含验证和盲测的墙钟约 27.88 分钟。平均每秒 167.56 个宏决策、467,476 个 2 ms 物理环境步。PyTorch 显存峰值分配 1,843 MiB，峰值保留 2,600 MiB。

尺寸自适应微提拉保持为独立试验。后续按附件等候确认。[训练曲线](../test/runs/2026-11-22/policy/stage4-policy1-ours-s17/training/updates.jsonl) · [逐项数据](../test/runs/2026-11-22/policy/stage4-policy1-ours-s17/training/result.json) · [权重](../test/runs/2026-11-22/policy/stage4-policy1-ours-s17/training/best.pt) · [SHA-256 清单](../test/runs/2026-11-22/policy/stage4-policy1-ours-s17/SHA256SUMS)

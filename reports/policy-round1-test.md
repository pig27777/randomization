# 策略线第 1 轮：测试报告

最近测试：2026-11-04。实验结论见 [test/policy/REPORT.md](../test/policy/REPORT.md) 第 1 轮结果；本文件只记录检查范围和结果。

## 范围

`policy/` 新增：`env.py`（evaluate 降阶模型后端、R1/R3 物体采样、观测）、`learn.py`（MLP 集成 + 选动作）、`run.py`（P1、训练、P4）、`content_snapshot/`（内容物代码只读副本）。

## 命令（在 `policy/` 下）

```
flock /tmp/claude-gpu.lock /home/ubuntu/Dev/Envs/mjwarp/bin/python -B run.py p1     # 正式
/home/ubuntu/Dev/Envs/mjwarp/bin/python -B run.py train --dist R1 --tag smoke --n1 112 --n2 56   # 试跑，结果已删
/home/ubuntu/Dev/Envs/mjwarp/bin/python -B run.py train --dist R3 --tag smoke --n1 112 --n2 56   # 试跑，结果已删
```

日志 `policy/results/logs/p1.log`；结果 `policy/results/p1/`（`p1.jsonl` 90 行、`summary.json`）。

## 结果

| 项 | 结果 |
|---|---|
| P1（4 个物体、9 组参数 × 10 种子） | 通过：sim_fail 0/90；硬物 20 N 成功、2 N 滑落；易压物高 F 过度挤压。易弯物中心横夹 20 N 10/10 夹取失败，沿长边 10 N 10/10 成功 |
| 训练流程试跑 | R1、R3 各 168 次能跑通，拟合与选动作正常；R1 第 2 批成功率 29% → 54% |
| P2–P4 | 未运行（用户决定当日不训练） |
| P5 | 未运行（需要代理测试集结果） |

## 遗留问题

- R3 分布上随机动作的 sim_fail 12.5%，等内容物线第 2 轮 S1。
- 夹爪“达到 F 后”的解释、R1 阈值（I-13）、与 `proxy_eval` 统一动作（I-14）待决定。
- 刚体的 `max_grip_force_N` 含接触尖峰（目标 20 N 时中位约 40 N），不影响判定。
- 未写单元测试；`learn.py` 只通过试跑检查。

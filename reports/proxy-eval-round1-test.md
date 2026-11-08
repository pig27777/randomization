# 代理真实测试集第 1 轮：测试报告

最近测试：2026-11-04。实验结论见 [test/proxy_eval/REPORT.md](../test/proxy_eval/REPORT.md) 第 1 轮结果；本文件只记录检查范围和结果。

## 范围

新增 `proxy_eval/`：`task.py`（任务流程）、`objects.py`（X1–X12）、`evaluate.py`（Newton 后端）、`run_e2.py`、`run_round1.sh`。另在 `bag_sim/README.md` 已知问题 6、7 后补了一句指向本轮报告；`log/ideas.md` 新增 I-23。

## 命令（在 `proxy_eval/` 下，Newton 1.6.1）

```
bash run_round1.sh X1 X3 X5 X6 X2 X4             # 正式运行，每个物体单独 flock，各 50 组
LIMIT=3 bash run_round1.sh X8 X7 X12 X11          # 正式运行，袋子物体各 3 组
LIMIT=1 bash run_round1.sh X10; LIMIT=1 bash run_round1.sh X9
python -B run_e2.py summary                       # 汇总，写 results/e2_summary.json
python -B evaluate.py X1 --params '{...}' --trace # 试跑（不排队，只用于调试和选参数）
```

结果：`results/e2_X*.jsonl`（共 314 行），`results/e2_summary.json`；日志：`results/logs/e2_X*.log`。

## 结果

| 项 | 结果 |
|---|---|
| E1 接口 | `evaluate(object_spec, grasp_params, seed)` 返回 README 规定的字段，外加诊断字段；12 个物体都能建模并跑通 |
| E2 稳定性（≤ 2%） | 不通过：X1–X5 为 28–46%，X6 为 4%；袋子物体在 1–3 组中多数是 `sim_fail` |
| E3 区分度 | 每个物体至少两种结果；全体四种结果都出现（掉落只有 1 次） |
| E4 耗时（中位 ≤ 120 s） | 四面体物体通过（中位 16.6 s）；袋子物体不通过（223–1091 s） |

运行中修复的问题：
- 松开时张开方向写反了；
- 合拢超时被误判为夹取失败；
- 一侧指垫没有接触时，记录接触行程会报错，X6 第 12 组后因此中断，修复后续跑完；
- 10 mm 海绵网格的第一次 X1 正式运行（3 组，全部 `sim_fail`）已丢弃，改用 20 mm 网格后重跑。

## 遗留问题

- 袋子物体没有跑满 50 组；被弹飞的判法与 `test/content_model/README.md` 有差异，待确认；任务流程与 `policy/env.py` 不一致（I-14）。
- 未做步长减半等可信度检查。

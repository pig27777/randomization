# Policy V3-MVP · 2026-11-25 · ubun

环境为 `.venv-physical-parcel`。MuJoCo/MJWarp 4.0.1、Warp 1.19.0、PyTorch 2.15.0+cpu，GPU 为 RTX 5090。入口参数和完整复现命令见[主题报告](../../../../policy_v3/REPORT.md)和[实现说明](../../../../../src/policy_v3/README.md)。[源码和资产指纹](provenance.json)

| 子运行 | 输入与输出 |
| --- | --- |
| [smoke-verified](smoke-verified/smoke.json) | 两 Worker，100 回合，种子 7，B=1，K=8，12 个 BCE epoch，16 回合 PPO。 |
| [c1-final-1000](c1-final-1000/c1/collection.json) | 八个 CPU Worker，1,000 回合，种子 117，B=1，K=8。原始 JSONL.gz 在各 Worker 目录中。 |
| [c2-final-1000](c2-final-1000/bce.json) | 上一批 Replay，20 个 BCE epoch，保存固定分组的检查点。 |
| [c4-final](c4-final/round-0/update.json) | 两 Worker，12 回合，PPO 后接累计 Replay 的单轮 BCE。 |
| [rigid-gpu](rigid-gpu/scripted.json) | 单卡持锁，外表面 mask 和深度，脚本抓放。 |
| [flex-probe](flex-probe/scripted.json) | 持锁运行 PE Flex 深试探。22 帧后触发穿透保护，保留停机前轨迹。 |
| [冻结检查](frozen.log) · [单测](unit-final.log) · [数据审计](archive-audit.log) | 路径级零差异、12 项单测和历史、时间戳、配对状态与分组检查。 |

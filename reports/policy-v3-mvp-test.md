# Policy V3-MVP 测试报告

2026-11-25，北京时间，主机 `ubun`。

- `.venv-physical-parcel/bin/python -B -m unittest discover -s tests/policy_v3 -v` 12 项单测全部通过，覆盖视觉信息边界、训练候选来源、GroupSplit、物理无效分流、边界 pre-tanh PPO、历史和 mask、原生夹爪力矩标定、PE 接触掩码、急停、评测、C4 屏障和冻结路径。[单测日志](../test/runs/2026-11-25/policy/policy-v3-mvp-r1/unit-final.log)
- 路径级 `git diff --exit-code` 对 `fix4.1-route-b-base`、`2ef4a394`、`736db9b2` 以及任务起始版本 `2229e650` 均退出 0，覆盖冻结核心、原 Route B／Plan B 配置、既有 7D PPO 与共享 Yam_B 平台。
- 原生刚体 C0 通过，C1 采集 100 和 1,000 回合，C2 训练 BCE 分别下降到 0.3388 和 0.3290，C3/C4 完成实际单批更新。详细结果和失效原因见[主题报告](../test/policy_v3/REPORT.md)与[运行数据](../test/runs/2026-11-25/policy/policy-v3-mvp-r1)。
- 两 Worker 测试使用独立 CPU 进程模拟 GPU 编排。本机只有一张 RTX 5090。结果范围是 MVP 软件验证，正式袋体训练、双物理 GPU 验证、科学验证及 Sim-to-Real 尚未执行。

# Yam_B Policy V3 代理预训练测试报告

最近测试于 2026-11-27 00:54（北京时间），主机 `ubun`。C0 传力门禁未通过。运行配置、原始记录与后续阶段状态见[实验报告](../test/runs/2026-11-26/yam-surrogate-v1-rl/v1_rl_report.md)。

|范围|方法|结果|
| --- | --- | --- |
|预检与冻结|`git fetch origin main` 和对规定标签/提交的路径级 `git diff --exit-code`|PASS。预检时本地与远端同步，规定的冻结文件和配置零差异。 [日志](../test/runs/2026-11-26/yam-surrogate-v1-rl/frozen-check.log)|
|Policy V3 回归|`PYTHONPATH=src:experience:. .venv-physical-parcel/bin/python -B -m unittest discover -s tests/policy_v3 -v`|PASS，原 12 项与另外 4 项子步审计共 16/16。 [日志](../test/runs/2026-11-26/yam-surrogate-v1-rl/policy-v3-regression.log)|
|薄板片预筛|原生 MuJoCo 4.0.1、0.4 ms、每种结构/工况一例，按原模型包覆流程静置|8 条诊断有完整输出。用户排除此模型，结果保留作历史。 [数据](../test/runs/2026-11-26/yam-surrogate-v1-rl/excluded-plates/c0-contact-qualification.jsonl)|
|周一版直接承载诊断|外层持锁运行 `audit_monday_rope.py`，逐 2 ms 记录柔索空间力，两条原轨迹各复跑 1 次|轨迹逐元素复现，数值均有限。悬停时柔索直接承担 99.998% / 99.997% 重量。C0 独立接触传力门禁 FAIL。 [JSON](../test/runs/2026-11-26/yam-surrogate-v1-rl/monday-rope-audit.json)|

完整闭环 C0、并发扩容、C1–C4 和留出评估均未执行。原生 `robot_payload_direct_force` 未测，诊断中的直接承载指柔索向四块内容物分配的力。

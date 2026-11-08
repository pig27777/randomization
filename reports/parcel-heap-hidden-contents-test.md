# 快递堆隐藏内容物与薄信封实验检查

最近检查：2026-11-09，机器 `ubun`。本轮已完成并留档，尚无有效完成整段抓取的结果；不能据此计算真实抓取成功率或断言任务无效。

## 范围

完整 YAM linear_4310 + 5–6 个自由刚性邻件 + 闭合柔性目标壳。书/衣物保持相同外壳、尺寸和质量；薄信封含空、纸张代理、薄硬片三种，膜接触厚度固定 0.15 mm。对照堆顶、部分压住、跨空隙；无抓取焊接或夹爪绑定。

## 执行与结果

| 检查 | 结果 |
| --- | --- |
| 入口诊断，共 32 条（含旧失败轮及两个单独试跑） | 修复轮主矩阵 15 条，7 通过、8 未通过；无初态穿插、无 MuJoCo 警告 |
| 中心/侧面接触试抓，14 条 | 3 接触物理超限、4 计算时限、7 初态 IK 失败；没有有效整段成功/失败样本 |
| `python -m unittest discover -s test/policy -p 'test_parcel_heap.py' -v` | 4 项通过：外壳/膜厚/无绑定、初态堆形、原 YAM 路径回归、侧夹方向 |
| `python -m unittest discover -s test/policy -p 'test_probe_feedback.py' -v` | 原有 3 项通过 |
| `python policy/audit_parcel_heap.py` | 15 条主入口齐全；各阶段源码快照校验通过；生成 46 条 CSV 与汇总 |
| 录制状态渲染 | 初始外观及静置/接触末帧留档，无重新运行轨迹或伪造成功帧 |
| 修正方向后的侧夹只读 IK 对照 | 低位未过门槛，250 mm 高位对照位置误差 <0.004 mm；不是高位抓取测试 |

以上 `python` 均使用仓库 `.venv/bin/python`。正式动力学命令均加 `flock /tmp/claude-gpu.lock`；详细命令见实验报告。

初版邻箱 15 mm 重叠已修复，首批“通过”不进入后续资格判断；薄膜接触阻抗调为 .999、薄信封步长改为 0.1 ms，穿透验收线未放宽。侧夹方向错误已修正，历史结果与其源码快照保留，没有覆盖或追认通过。

## 限制与交付

壳、纸张和衣物为未实物标定的代理；当前只是小型混合堆，不是大规模全软包山。接触峰值为每 25 步采样值。初始同外壳不代表静置后完全无法从图像区分。未运行完整抓点矩阵、训练、泛化评估或真实机械臂。

- [实验计划、完整结果和解释](../test/policy/HIDDEN_CONTENTS_REPORT.md)
- [逐次 CSV](../policy/results/parcel_heap_20261109/trials.csv) · [汇总](../policy/results/parcel_heap_20261109/summary.json)
- [初始及录制关键帧](../policy/results/parcel_heap_20261109/contact_fix/frames/)
- [实验入口](../policy/parcel_heap.py)

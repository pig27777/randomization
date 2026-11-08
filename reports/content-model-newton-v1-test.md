# 内容物 v1 Newton 补充验收检查

最近检查：2026-11-05 19:58（UTC+8）。状态：规定门槛全部通过，内容物 v1 已冻结，接续 v1.1。

## 范围与方法

- MuJoCo 3.17.0、CPU bushing、0.4 ms、implicitfast；Newton 上限 100 次、线搜索上限 20、容差 1e-8。
- I-27 场景片段、直接接触开关、外来 tendon；原物理与采样检查。正式命令统一 `flock /tmp/claude-gpu.lock`，源码只改 `contents/segment/`。
- 两软锚点的 CG/Newton 配对 R1y/z、R2、R3；Newton 表重新标定后独立 M3、新种子 200 组、I-22 与三类各 10 次 evaluate。
- 策略 `env.py` 只读；内容物适配器按已经批准的 README 将旧 4 s 合拢超时对齐为 8 s，未更改任务标准，未提前采用 v2。

## 已运行与结果

工作目录为 `contents/segment/`，Python 为 `/home/ubuntu/Dev/Envs/mjwarp/bin/python`。

```bash
segment_python=/home/ubuntu/Dev/Envs/mjwarp/bin/python
flock /tmp/claude-gpu.lock "$segment_python" -B -m unittest discover -s tests -v
flock /tmp/claude-gpu.lock "$segment_python" -B freeze_v1.py compare --output-root results/freeze_v1_newton --run-id solver_compare --table results/freeze_v1/mapping_pass2/table.json --workers 8
flock /tmp/claude-gpu.lock "$segment_python" -B freeze_v1.py calibrate --output-root results/freeze_v1_newton --run-id mapping_pass1 --all-segments --workers 8 --table results/freeze_v1/mapping_pass2/table.json
flock /tmp/claude-gpu.lock "$segment_python" -B freeze_v1.py m3 --output-root results/freeze_v1_newton --run-id m3_final --table results/freeze_v1_newton/mapping_pass1/table.json --workers 8
flock /tmp/claude-gpu.lock "$segment_python" -B freeze_v1.py evaluate --output-root results/freeze_v1_newton --run-id evaluate10 --count 10
flock /tmp/claude-gpu.lock "$segment_python" -B freeze_v1.py random --output-root results/freeze_v1_newton --run-id random200_seed20261107 --count 200 --seed 20261107 --workers 8
```

- 单元检查 **23/23，通过，7.454 s**；包含原 19 项与新增 4 项 I-27 接口检查。[正式日志](../contents/segment/results/freeze_v1_newton/logs/unit_formal.log)。
- 配对对照 **16 次无确认仿真失败**；10 ms 采样的 Newton 实际迭代最多 16 次，CG 多次触及 100 次上限。易弯物 R1y/R1z/R2/R3 相对变化为 −1.314%/−0.017%/+0.184%/+1.222%；易压物为 −0.279%/−0.025%/−0.943%/+15.614%（R3 为接触滑移，绝对值 0.697→0.805 mm）。[完整对照](../contents/segment/results/freeze_v1_newton/solver_compare/summary.json)。
- 探索 evaluate 每类一次：硬物 0.713 s、易弯物 7.029 s、易压物 8.116 s；仅确认入口，不用于正式吞吐结论。探索后省去力计算路径未使用的能量求和与严格零 COM 偏置力矩计算；力学单元检查通过。
- Newton 表 420 次标定、353 有效点；6 次确认失败原样保留并另作故障状态重放，不入表。独立 M3 **15/18，硬物通过**，表外拒绝。
- 本轮随机入口曾漏传原 `OMEGA_MAX`，错误配置 20261106 批 **5/200** 失败；完整保留。修复只恢复标准平台上限，原五个 κ 静置+R3 **0/5** 失败。[配对复核](../contents/segment/results/freeze_v1_newton/corrected_entry_replay5/summary.json)。新增入口回归后 **24/24 单元检查通过，7.414 s**，见 [最终日志](../contents/segment/results/freeze_v1_newton/logs/unit_corrected.log)。
- 修复后独立新种子 20261107：**1/200=0.5%**，执行错误 0，静置无失败；#57 在 R3 close 时 y 压缩越限，保存状态重建比例 1.203528，警告计数 0。[证据](../contents/segment/results/freeze_v1_newton/random200_seed20261107/case_057/failure_analysis_r3.json)。
- I-22 **10/10 成功**，外排最大转角 0.13193°，两侧持续接触；CPU 三类各 10 次中位 **0.695/6.791/7.880 s**，两软物均 ≤8 s。
- [最终汇总](../contents/segment/results/freeze_v1_newton/acceptance.json)记录所有门槛和源码核对；最终 M3、200 组、evaluate 的 7 个运行模块指纹一致。[v1 标签](../contents/segment/results/freeze_v1_newton/release_v1/v1.json)与源码包只写一次；项目无 Git 元数据。

## 限制

没有安装、升级包或改共享库；没有修改袋子线或策略线代码。装袋、YAM 顶面抓尚未验收，袋子线需移除旧的伪造 `bag_rope` 元数据补丁后接入新接口。M3 三项未命中保留；易压物 10 N 的计时配置 10/10 物理夹取失败，速度不代表完整成功搬运成本；硬物执行器 7/10 力峰超过 50 N 却返回 success，已交 I-29，不解释为真实夹爪力上限通过。I-26 未扩大为已关闭。全部历史结果保留。详见[内容物报告](../test/content_model/REPORT.md)“冻结 v1 补充验收（Newton 求解器）”。

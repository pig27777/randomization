# YAM 试夹反馈实验检查

最近检查：2026-11-08，机器 `ubun`。**96 次开发 + 144 次独立测试完成；未证明反馈收益。3 项统计/信息隔离回归通过；18 项实验审计中 17 项通过，腕相机非指尖接触检查失败。**

范围：新增独立的试夹反馈策略实验、完整 YAM linear_4310 动力学接入、证据审计与记录状态渲染。使用项目 `.venv` 的 MuJoCo 3.17.0、NumPy 2.5.4，无新依赖安装。原本构、旧策略入口及共享 YAM 资产未修改。

| 检查/命令（仓库根目录） | 结果 |
|---|---|
| `policy/probe_feedback.py prepare`（用 `.venv/bin/python -B`） | 冻结 v1.3 归档 SHA 与 7 个源文件/表格哈希通过；YAM XML、网格、许可指纹已保存 |
| `policy/probe_feedback.py smoke --workers 6` | 早期运动学夹爪 6 次仅用于历史入口检查；YAM 整机 6 次均成功，未加入正式成功率 |
| `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 flock /tmp/claude-gpu.lock .venv/bin/python -B policy/probe_feedback.py dev --workers 12` | 96/96 完整；81 成功、6 抓取失败、8 过挤压、1 数值失败；未删除失败样本 |
| `.venv/bin/python -B policy/probe_feedback.py fit` | A=14 N、B=10 N；C=k5 两项试夹观测；策略测试前冻结，12 组开发试夹前缀在四个未来目标力下均相同 |
| 同一持锁入口 `policy/probe_feedback.py test --workers 12` | 144/144 完整；A 47/48、B/C 各 46/48；数值失败与过挤压均 0；48 对 B/C 试夹指纹相同 |
| `.venv/bin/python -B -m unittest discover -s test/policy -p 'test_probe_feedback.py' -v` | 3/3 通过：几何不改变反馈决策；配对 bootstrap/数值失败剔除；试夹前缀不同必须拒绝收益结论 |
| `.venv/bin/python -B policy/probe_feedback.py summary` | 聚类统计完成；C−B=0，C−A=−2.08 pp，95% 区间 [−6.25,0] pp；预定收益标准未过 |
| `.venv/bin/python -B policy/audit_probe.py` | **exit 1，17/18 通过**。源/资产/策略哈希、矩阵完整性、同物体同阈值、数据隔离和原生力矩限制通过；7 次 `d405` 腕相机与物体接触，最大 0.6143 mm，故零非指尖接触检查失败，未改检查标准 |
| 持锁入口 `policy/diagnose_probe_contact.py` | 4 个去重条件重放，7 项结果字段均与原试次完全相同；额外接触定位于腕相机外壳，原试次保留 |
| `MUJOCO_GL=egl .venv/bin/python -B policy/render_probe.py` | 由记录 qpos 生成 5 帧整机画面。首次渲染触发可视几何容量警告，隐藏诊断 tendon 并增加渲染容量后完成；仅渲染设置变化，物理记录未变 |

正式测试 IK 最大误差约 0.100 mm；实际末端跟踪误差最大 62.17 mm，各试次 P95 中位数 37.99 mm。未把几何 IK 精度写成实际控制精度。此布置不满足纯指尖抓取验收，也没有真机或宽软包泛化证据。

代码语法与文件组织终检结果见结果目录的 `delivery_check.json`；源码、原始输出、计划快照和阶段元数据均保留。工作开始 `git pull --ff-only` 返回 Already up to date。

[完整实验报告](../test/policy/PROBE_REPORT.md) · [审计与未通过项](../policy/results/yam_probe_feedback_20261108/audit.json) · [原始测试数据](../policy/results/yam_probe_feedback_20261108/test_trials.csv) · [统计检查日志](../policy/results/yam_probe_feedback_20261108/statistical_checks.log)

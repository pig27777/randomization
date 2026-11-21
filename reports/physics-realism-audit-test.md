# 包裹物理真实性审计检查

最近检查为 2026-11-21 18:33（北京时间）。机器 `ubun`，基线 `96a896c1`。CPU 和 MuJoCo 使用本工作树的 `.venv-physical-parcel/bin/python -B`，MuJoCo 4.0.1。该解释器为 Python 3.14，当前安装的 PyTorch 为 CPU 版。CUDA 用同机另一工作树已有的 Python 3.12 解释器加载本工作树内从缓存组装的依赖目录，PyTorch 2.11.0+cu130、RTX 5090。正式运行持 `/tmp/claude-gpu.lock`。

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| 冻结 P6、PE 刚度、敏感性 | `-m src.low_order_parcel.audit_physics_realism`、`test/bag_model/audit_stage1_stiffness.py`、`test/low_order_parcel/audit_stage2_sensitivity.py` | 8,003/8,192 路悬停，343.974884 N。P6 逐路和波形与冻结归档 SHA-256 一致。YAM PE 刚度和敏感性汇总与归档一致。 |
| YAM 闭环及逐步能量 | `test/low_order_parcel/audit_yam.py`、`test/low_order_parcel/audit_energy_passivity.py` | Stage 2 完整侧抓和受控松夹通过。Stage 2.5 能量门禁失败。修正折边入口条件后，YAM 重包首次捕获仍出现 −9.569 mJ，冻结模型静止捕获出现 −6.826 mJ。 |
| GPU 相图 | `test/low_order_parcel/audit_phase_diagrams.py`，每图 32×32 | 三图通过。修复初筛 Phase B 错误的探针入参，四个交叉网格单元的全部帧输入与单环境轨迹相符。 |
| Stage 1 膜片 | `test/bag_model/audit_stage1_scaling.py` | 膜厚倍增后弯曲、拉伸刚度分别为 7.905 倍和 2.000 倍。宏观摩擦斜率为 0.446。绝对垂坠约为解析值 0.44 倍，低速蠕移单列。 |
| UMI 模板 | `test/low_order_parcel/check_real_envelope_containment.py`，空模板与合成输入 | 三种区间判定、覆盖率和缺失机制识别通过。真机测量待填。 |
| 原整包回归 | `flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B -m unittest discover -s test/bag_model -p test_physical_parcel.py -v` | 20/20 通过，8.730 s。 |

首轮 CUDA 启动因指定解释器为 CPU 版失败，随后以兼容解释器和缓存依赖成功复跑。YAM 顶抓初始 90° 姿态不可达，采用已验证可达的 270° 姿态。首个折边补充适配器在接触失效后仍施加入口推力，该异常样本退出模型验收。膜片首次时间步未满足原弯曲积分稳定性上界，按计算上界改用 4 µs。

[主审计报告](../docs/PHYSICS_REALISM_AUDIT_REPORT.md) · [运行记录和日志](../test/runs/2026-11-21/bag/physics-realism-audit-r2) · [回归日志](../test/runs/2026-11-21/bag/physics-realism-audit-r2/physical-parcel-final.log)

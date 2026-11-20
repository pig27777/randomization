# RSS 2027 Stage 2 测试报告

最近验证为 2026-11-20 19:16（北京时间），机器 `ubun`，CPU i9-14900KF，项目环境 `.venv-physical-parcel`。本轮新增依赖为 `torch==2.15.0+cpu` 和 `matplotlib==3.11.2`。

| 检查 | 结果 |
| --- | --- |
| 包络和配置生成 | Milestone 1 的 3 项均通过。Stage 2 配置与生成器输出一致，所用冻结档案及哈希一致。 |
| Gate B 双指代理 | 7 项通过，20.3 s。1 和 2 ms 各 100,000 步，批大小 16，零非有限量、零负余料，峰值张力低于输入推得的上界。 |
| Gate A 刚体与代理对齐 | 两档步长各 6 类全过。预检中 PE 加载脱手前触台。增加探针离台净空后，各正式加载和松夹试验直至脱手的支撑力为 0 N。 |
| Gate C 既有整包回归 | 20 项通过，8.7 s。测试用网格 `crossing_triangle` 产生一条约束警告。 |

```sh
.venv-physical-parcel/bin/python -B src/low_order_parcel/build_stage1_envelopes.py
OPENBLAS_NUM_THREADS=1 .venv-physical-parcel/bin/python -B policy/rss2027/test_stage1_envelopes.py -v
flock /tmp/claude-gpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_stage2_proxy.py -v
flock /tmp/claude-gpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-physical-parcel/bin/python -B src/low_order_parcel/verify_stage2_alignment.py
flock /tmp/claude-gpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-physical-parcel/bin/python -B src/low_order_parcel/verify_stage2_alignment.py --dt .001 --output test/runs/stage2_freeze/dt-1ms
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B -m unittest discover -s test/bag_model -p test_physical_parcel.py
```

重跑对齐时用 `--output` 指定新的空目录。脱手绝对值的主要偏差在气泡膜，PE 在 11.25 mm 进给档的持续双侧承载偏长。耗时为同机器 CPU 单环境实际探针回合。

[冻结报告和偏差明细](../docs/STAGE2_FREEZE_REPORT.md) · [2 ms JSON](../test/runs/stage2_freeze/alignment_report.json) · [1 ms JSON](../test/runs/stage2_freeze/dt-1ms/alignment_report.json)

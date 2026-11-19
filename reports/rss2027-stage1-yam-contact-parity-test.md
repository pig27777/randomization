# Stage 1 YAM 同态接触比对

最近检查时间 2026-11-19 20:59（北京时间）。在 ubun / RTX 5090 持共享锁，以 11 月 16 日 r3 整袋 200 节点 YAM 合指帧 137、零速度和原探针首步控制量，配对执行 CPU MuJoCo / MuJoCo Warp 4.0.1 各一步，步长 25 μs。

```sh
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B \
  test/runs/2026-11-19/bag/rss2027-stage1-yam-contact-parity-r2/compare.py
git diff --check
```

- 初始 GPU 输入和 CPU 的 float32 表示相同，无溢出。原探针首步的触点数、左右力均复现。CPU/GPU 接触数为 103/94，全部激活。差额来自膜–内容物 43/36 和膜–左指 6/4，另外三组膜–右指、膜–支撑、膜–自接触分别为 6/6、45/45、3/3。
- 左指两个 CPU 独有约束来自膜三角面 148 与 `left_round_6/7`，接触点几乎重合，带符号距离均为 −17.09 μm。CPU 独有触点竖向合力为 +0.460 N，占左指首步竖向差 +0.499 N 的 92%。四个共有左指触点的竖向力差为 +0.039 N。这一帧的主要左侧竖向差在约束求解前已有候选几何差异。
- 右指全部六个候选可配对，最大触点位置差 0.092 μm，合力向量差仍为 0.122 N。匹配触点的力差可能来自全局接触集合变化与耦合求解，单个观测无法量化各因素。膜–内容物有 36 个共同元件编号，CPU 独有 7 个。共同元件触点沿 CPU 法向偏移中位数 249.98 μm，且切向见证点也有差异，全部超出 0.10 mm 的几何配对容差。
- 1.046 N 是此前两条轨迹第 7 步的左指最大向量差，本轮复现的是共状态的首步差 0.525 N。完整闭环接触一致性、差异形成于具体候选筛选还是窄相几何计算，以及第 7 步峰值的量化分解都仍待验证。

早期首轮在成对步进后因 MuJoCo Python 无 `model.flex` 命名访问器而未写出数据。第二轮用 `mj_id2name` 修正记录器，沿用原动力学和标准。[首轮日志](../test/runs/2026-11-19/bag/rss2027-stage1-yam-contact-parity-r1/run.log) · [复测计划](../test/runs/2026-11-19/bag/rss2027-stage1-yam-contact-parity-r2/plan.md) · [复测数据](../test/runs/2026-11-19/bag/rss2027-stage1-yam-contact-parity-r2/results.json) · [复测日志](../test/runs/2026-11-19/bag/rss2027-stage1-yam-contact-parity-r2/run.log)

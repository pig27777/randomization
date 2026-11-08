# 位移控制对照测试报告

最近测试：2026-11-02。Newton 1.6.1 / Warp 1.18（`bubble_mailer/.venv`），RTX 5090。接 [REPORT.md](REPORT.md) 第一项基准，只改测试代码，不改模型。

## 目的

判断刚体接触后的持续振动来自哪里：刚体和软体之间的耦合迭代，还是接触生成与计算本身。做法是把工具改成运动学刚体（按给定轨迹移动、不参与 VBD 求解），和原来的力控制对比。

## 改动

- `benchmark.py`：新增 `--drive position --tool-z-mm`（运动学工具）、`--dense-s`（逐子步采样窗口）；每次采样另记活动节点速度中位数和超过 1 mm/s 的节点数。位移控制下没有目标载荷，`support_balance` 改为比较底部支撑反力与工具反力。
- `run_suite.py`：重放时带上新参数。`README.md`：补充说明。
- 力控制路径未改。回归核对：`ctl_force_*` 与原 `guided_base_*` 一致（压板支撑力 16.28 / 16.20 N，节点上手指 1.998 / 1.998 N、下陷 8.127 / 8.127 mm）。

## 方法

与原基准相同的软块、网格（12×10×4）、接触参数（ke 3000、kd 3）、1 ms 子步、10 次迭代、2 秒，末尾加 0.1 秒逐子步采样。位移控制的目标高度取自力控制的稳态：压板取 `guided_iter40_plate` 的 60.148 mm（支撑力误差 0.04%），两种手指都取节点上手指的 57.102 mm（偏置手指没有可信的力控制参考）。

```bash
cd /home/ubuntu/Dev/Projects/random/test/contact_transfer
NP=/home/ubuntu/Dev/Projects/real2sim/bubble_mailer/.venv/bin/python
C="--dt .001 --iterations 10 --seconds 2 --dense-s .1"
$NP -B benchmark.py --cases plate finger finger_off $C --tag ctl_force
$NP -B benchmark.py --cases plate --drive position --tool-z-mm 60.148 $C --tag ctl_position
$NP -B benchmark.py --cases finger finger_off --drive position --tool-z-mm 57.102 $C --tag ctl_position
```

补充运行：`ctl_position_dt0p25_{plate,finger_off}`（0.25 ms 子步）、`ctl_position_ke1000_finger_off`（ke 1000、kd 1）。

## 结果

最后 0.5 秒均值。“中位速度”是活动节点速度中位数。

| 运行 | 支撑力 N | 工具反力 N | 材料残差 | 最大 / 中位速度 mm/s | >1 mm/s 节点 | 下陷漂移 mm | 通过 |
|---|---:|---:|---:|---:|---:|---:|---|
| [ctl_force_plate](results/ctl_force_plate.json) | 16.28 | 21.20 | 32.3% | 36.6 / **4.01** | 459 | −0.232 | 否 |
| [ctl_position_plate](results/ctl_position_plate.json) | 19.61 | 19.80 | 4.6% | 35.1 / **0.10** | 116 | 0.000 | 否（仅静止项） |
| [ctl_force_finger](results/ctl_force_finger.json) | 2.00 | 2.05 | 2.7% | 4.9 / 0.019 | 16 | 0.028 | 否（仅静止项） |
| [ctl_position_finger](results/ctl_position_finger.json) | 2.00 | 2.00 | 0.07% | 0.19 / 0.007 | 0 | 0.000 | **是** |
| [ctl_force_finger_off](results/ctl_force_finger_off.json) | 1.23 | 2.37 | 111% | 490 / 0.97 | 274 | 0.415 | 否 |
| [ctl_position_finger_off](results/ctl_position_finger_off.json) | 1.88 | **15.10** | 107% | 872 / 0.028 | 62 | 0.000 | 否 |
| [ctl_position_dt0p25_finger_off](results/ctl_position_dt0p25_finger_off.json) | 1.88 | 1.84 | 76% | 294 / 0.029 | 18 | — | 否 |
| [ctl_position_dt0p25_plate](results/ctl_position_dt0p25_plate.json) | 19.63 | 19.49 | 15.0% | 30.5 / 0.073 | 87 | — | 否 |
| [ctl_position_ke1000_finger_off](results/ctl_position_ke1000_finger_off.json) | 1.80 | 6.50 | 108% | 823 / 0.052 | 89 | — | 否 |

逐子步记录（`dense_trace`）：

- 压板位移控制：工具反力每个子步在 19.42 / 19.80 N 之间交替，支撑力峰峰值只有 0.0009 N。速度最大的节点每个子步上下翻动约 35 µm。
- 偏置手指位移控制：工具反力以 3 个子步为周期循环（1.7、1.4、37.7 N），接触记录数恒为 39（4 个顶点、19 条边、16 个面），节点速度维持在 0.8–1.0 m/s。0.25 ms 子步时周期仍是 3（2.34、1.80、1.50 N），幅度变小。

## 结论

1. **压板和节点上手指的整体振动来自刚柔耦合迭代。** 位移控制下压板的中位速度从 4.01 降到 0.10 mm/s，下陷漂移从 −0.23 mm 降到 0；节点上手指全部验收通过。所以现在失败的“未静止”项，由少数接触节点的高频抖动造成，不是整体运动。
2. **偏置手指另有一个独立问题：接触计算本身在工具静止时也不稳定。** 工具固定、接触记录数不变，工具反力却以 3 个子步为周期在 1.4–37.7 N 之间跳动，节点速度接近 1 m/s。这和耦合迭代无关。它主要发生在边和面接触占多数的场景（节点上手指以顶点接触为主，是稳定的）。至于“边和面接触的力按重心坐标分到几个节点、这些节点同时更新导致过冲”，目前只是假设，没有验证。
3. **减小步长能削弱但消除不了偏置手指的抖动**（最大速度 872 → 294 mm/s），**降低接触刚度不能解决**（ke 1000 时最大速度仍有 823 mm/s，间隙还减小了）。
4. **原报告里“步长越小残差越大”，并不需要耦合迭代来解释。** 位移控制的压板在 0.25 ms 子步时，材料残差也从 4.6% 升到 15.0%，而工具根本不参与求解。这个瞬时残差主要来自抖动节点的惯性力。
5. **“最大速度 ≤ 1 mm/s”这条验收分不出整体振动和局部抖动。** 它在位移控制压板（中位速度 0.10 mm/s）和力控制压板（4.01 mm/s）上同样不通过，判断时需要结合中位速度、下陷漂移和支撑力的峰峰值。

## 限制

- 频谱只采样了 0.1 秒，频率分辨率是 10 Hz，表中“主频”在力控制运行里只代表慢变化，不能当作真实的振动频率。
- 偏置手指位移控制的目标高度借用了节点上手指的值，没有独立的力控制参考，所以只能检查两侧受力是否一致、是否静止，不能检查载荷大小。
- 每个设置只跑了一次；GPU 结果不完全可重复。
- 只测了平压，没有测夹取、提起或袋子。

## 下一步

- 偏置手指：在 3 个子步的周期内，逐条记录接触点、法线和重心坐标，以及每个节点收到的接触力，确认抖动是否只发生在边和面接触的节点上。
- 刚柔耦合：如果后续夹爪按位移控制，压板和节点上手指这类场景已经可以用；力控制的耦合振动另外处理。

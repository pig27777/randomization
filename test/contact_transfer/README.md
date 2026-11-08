# 接触传力基准

本目录独立保存第一项工作的测试代码、结果与报告，不修改业务模型或原始资产。

## 文件

- `benchmark.py`：Newton VBD 接触实验，可单独选择场景、步长、迭代、网格、接触参数。
- `metrics.py`：双精度材料受力与几何测量；不依赖 Newton 求解结果中的力缓存。
- `test_metrics.py`：用能量差分、合力和力矩校验测量工具。
- `results/*.json`：完整配置、环境、实际步长、逐时刻指标、验收结果。
- `results/*.npz`：采样节点位置、参考网格、四面体、固定节点。
- `REPORT.md`：实验完成后的结论与限制。
- `REPORT_position_control.md`、`REPORT_newton170.md`：位移控制对照；Newton 1.7.0 复测。
- `mujoco_benchmark.py`：MuJoCo 3.17 备选路径，同一四面体网格和 NH 体积弹性力。
- `run_suite.py`：从保存的配置复现；默认给结果加 `replay_` 前缀。
- `planar_contact.py`：本平压基准的覆盖区积分接触；按真实三角形/工具投影交集生成采样，按面积加权刚度。
- `contact_fixes.py`：项目内可选摩擦零速刚度修正，不修改安装环境。
- `diagnose_centroid.py`：形心补丁的条件切换、接触激活和逐节点颜色更新诊断。
- `REPORT_repair.md`：2026-11-03 修复对照和剩余限制。
- `REPORT_diagnosis.md`：边/面接触抖动机制的诊断（脚本在 `contents/newton/`）。
- `build_report.py`：从原始结果生成报告、图、汇总和哈希清单，不运行仿真。

## 环境与命令

使用本机现有环境，无需安装或升级依赖：

```bash
cd /home/ubuntu/Dev/Projects/random
NP=/home/ubuntu/Dev/Projects/real2sim/bubble_mailer/.venv/bin/python
$NP -B -m unittest discover -s test/contact_transfer -p 'test_*.py' -v
$NP -B test/contact_transfer/benchmark.py --cases plate direct_plate finger finger_off membrane --tag baseline
$NP -B test/contact_transfer/benchmark.py --cases plate finger finger_off membrane --dt .0005 --iterations 40 --tag dt0p5ms
python3 test/contact_transfer/run_suite.py --group quick
$NP -B test/contact_transfer/build_report.py
```

结果始终写入此目录的 `results/`，与启动目录无关。同一 `--tag` 和场景会覆盖此前对应结果，复现实验时请用新 tag 保留原记录。

## 场景与边界条件

- 均质软块：300 × 250 × 50 mm，总质量 0.3 kg，Lamé μ = 1500 Pa、λ = 1000 Pa，材料阻尼 1 Pa·s。
- 底层节点三个自由度固定，重力关闭。这样排除地面接触和软块自身重量的干扰。
- `plate`：240 × 200 mm 刚体压头，施加 19.62 N，惯性质量 2 kg。
- `finger`：20 × 20 mm 自由刚体压头，施加 2 N，质量 2/9.81 kg，初始中心位于节点上。
- `finger_off`：相同手指，在 x/y 各偏移 12.5 mm，初始中心位于基准网格节点之间。
- `direct_*`：将相同合力作为均布压力，通过三角形形函数积分分配到节点；仅作诊断，不要求与刚性压头的非均布压力产生相同下陷。
- `membrane`：240 × 200 mm 独立三角形膜片，质量 12 g，膜 μ=5000 N/m、面积系数=2000 N/m，直接向膜片施加分布载荷 19.62 N，观察是否通过膜—四面体接触传至底部。
- 载荷用 0.2 秒平滑增加；默认无摩擦。正式对照用竖直移动副，轴向无驱动、无阻尼；记录横向位移和倾斜，检查导向约束。`--free-tool` 保留自由刚体探索对照，不能把翻倒直接判成接触算法失败。
- 刚体接触使用 SDF 全表面接触；`--vertex-only` 用于诊断网格间隙漏碰。
- `--drive position --tool-z-mm Z`：工具改为运动学刚体，按平滑轨迹移到中心高度 Z（mm）后保持，不参与 VBD 求解，用来区分接触生成与刚柔耦合迭代。此时没有目标载荷，`support_balance` 改为比较底部支撑反力与工具反力（`interface_balance_pct`），材料残差按工具反力归一化。
- `--log-barrier`：仅 Newton ≥ 1.7，刚体–软体法向接触改用对数势垒（只在穿入超过接触半径一半时与二次罚函数不同）。用 1.7.0 时解释器为 `/home/ubuntu/Dev/Envs/newton_latest/bin/python`。
- `--dense-s T`：主运行结束后再推进 T 秒，每个子步采样一次，记录在 `dense_trace`；频率分辨率为 1/T。
- `--planar-quadrature`：仅本基准的竖直平面工具。将软块顶面与工具底面投影相交的多边形分片，每片采用三个积分点，使用真实重心坐标；以面积除以固定参考面积 `0.025² m²` 加权 ke/kd，网格变化时不重复增加总刚度。保留节点间覆盖能力，原生顶点/边/面记录被该表面积分替代。未覆盖任意姿态、侧面、圆角夹爪或膜片；不能和 `--collision-iterations` 合用。
- `--friction-zero-fix`：在首次 Warp 编译前启用刚柔、软软两条路径的连续零速切向刚度。仅修正源码的零速不连续；本轮膜片对照仍未通过静止项，不能视作通用消抖修复。
- `--collision-iterations`：Newton 1.7 的求解器内逐迭代重检测诊断，本轮没有解决问题，默认关闭。
- 每次采样另记活动节点速度中位数（`median_active_speed_mm_s`）和超过 1 mm/s 的节点数，用来区分整体振动与少数节点的局部抖动。
- CUDA graph 每帧包含偶数个子步；`actual_dt_s` 是实际步长，可能比请求值小。

## 测量与验收

最后 0.5 秒取平均：底部弹性支撑反力与已知载荷误差 ≤5%；刚体接触反力误差 ≤5%；活动四面体节点弹性+接触+外载残差的范数之和 ≤载荷的 5%；末段下陷漂移 ≤0.05 mm、软体和工具的平均采样最大速度 ≤1 mm/s。整个采样过程不能有倒置；界面垂直射线测得的几何穿透不得超过 0.05 mm。提前失败的运行没有稳态，末段指标不能当作稳定后的误差。

支撑反力由独立计算的弹性节点力获得，未把粘性支撑当成静态弹性承载；残差同样不包含材料粘性，因此只有达到静态才应通过。刚体接触力使用求解器相同接触记录和瞬时本构重新评估，包含接触阻尼。膜—软块接触力按所有颜色重算，累加后检查内部接触对的合力。

步长减半、迭代增加、网格加密的变化另行比较，不能由单次 `passed` 代替。垂直射线是针对本平压场景的几何检查，不是任意姿态网格无相交证明。倒置和穿透每 50 ms 采样；报告同时说明该采样限制。

这些是平台数值验收标准，不代表材料已经实物标定，也不代表完整袋子或机器人抓取场景已经通过。

## 本轮修复的复现

使用 `/home/ubuntu/Dev/Envs/newton_latest/bin/python`。以下接触设置仍为 ke 3000、kd 3，无摩擦；验收阈值未放宽。积分接触的节点上手指有缓慢衰减的加载瞬态，需要充分静置；不能将延长后的通过记作原 2 秒通过。

```bash
/home/ubuntu/Dev/Envs/newton_latest/bin/python -B -m unittest discover -s test/contact_transfer -p 'test_*.py' -v
/home/ubuntu/Dev/Envs/newton_latest/bin/python -B test/contact_transfer/benchmark.py --cases plate finger finger_off --dt .001 --iterations 10 --planar-quadrature --seconds 8 --tag replay_area
/home/ubuntu/Dev/Envs/newton_latest/bin/python -B test/contact_transfer/benchmark.py --cases finger --dt .0005 --iterations 10 --planar-quadrature --seconds 15 --tag replay_area_half
```

原始结果保存在 `results/repair_*.json` 和对应 NPZ；图和原 `SUMMARY.json` 未重新生成，当前结论以 `REPORT_repair.md` 为准。压密本构的第 2 项设计位于 [compaction_model.md](../../contents/newton/compaction_model.md)，尚未实现。

## 数值对照范围

正式对照包含：1 ms 子步下10/40次迭代，40次迭代下1/0.5/0.25 ms子步；24×20×8网格；接触系数按名义顶面节点面积缩放的细网格对照；SDF体素、tile开关、阻尼、接触刚度、k_start及仅顶点接触的诊断；膜片0.1/0.05 ms与10/20次迭代复核；延长到5秒以及重复运行。

单次 `passed=true` 只说明该次运行通过，必须结合收敛和重复性结果判断平台。`run_suite.py` 返回非零表示复现中至少一个场景未通过，这是报告里失败场景的正常预期。`all` 会运行全部正式配置，耗时明显长于 `quick`。

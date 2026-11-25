# Yam_B 纯仿真接口

入口在现有统一平台的 `yam_platform.yam_warp.SimYAMInterface`。直接加载随平台归档的 I2RT Menagerie `yam_linear_4310.xml` 和网格，按 ABC Yam_B 的桌面坐标安装。MJCF 保留六个转动关节、连杆惯量、外壳碰撞和两条原生等式耦合滑轨。旧抓取试验由原入口继续执行。

```python
from yam_platform.yam_warp import SimYAMInterface

yam = SimYAMInterface(worlds=1, try_capacity=60)
yam.send_cartesian_action([0.3, 0, 0, 0, 0.2, 0, 0.7])
yam.step()
obs = yam.get_observation()
history = yam.history.read()
```

在仓库根目录用 `PYTHONPATH=experience` 导入。批量输入使用指定 CUDA 设备上的 Warp `float32[N,7]`，可由策略框架通过 Warp 设备互操作构造。示例列表适用于单环境。返回的设备视图会随下一次更新变化，需要保留的张量由调用方复制。 `reset(mask)` 支持 CUDA 布尔 `[N]` 掩码。纯机器人验证使用 MuJoCo/MJWarp 3.17.0 与 Warp 1.19.0，现有 PE Flex 需要仓库的 MuJoCo/MJWarp 4.0.1 环境。

## 控制与观测

- `send_joint_command(position)` 接收 SDK 顺序 `joint1…joint6` 的目标角（rad）和总开度（m），可传入对应目标速度和前馈。夹爪的前馈单位是总开度方向的等效 N。目标逐关节限位、限速，然后由有限刚度 PD 跟踪。
- `send_cartesian_action()` 接收 `[dx,dy,dz,drx,dry,drz,g]`，各维范围 −1 到 +1。前六维是 ABC 世界坐标下的平移和旋转向量增量，默认逐轴比例为 4 mm 和 0.04 rad。阻尼最小二乘使用当前姿态的 TCP 雅可比，可在任意可达姿态起作用。平移目标限在配置的工作空间。开度从 −1 关闭到 +1 全开。
- `grasp_site` 是本地 Menagerie 夹持坐标，+Z 沿接近方向，±Y 沿夹爪开合方向。机械臂姿态由原生 FK 计算。夹口总开度取两条滑轨行程之和。零开度主要由真实指垫碰撞约束，内部 PD 目标附小量预压。
- 默认物理步长 1 ms，电机和传感读数 250 Hz，目标插补 125 Hz，动作与历史发布 25 Hz。时钟按整数子步对齐。接口可配置时钟、工作空间和动作尺度。
- `get_observation()` 给出 `joint_pos [N,6]`（rad）、`joint_vel`（rad/s）、`joint_eff`（N·m）、`joint_pos_cmd`（rad），以及各 `[N,1]` 的 `gripper_pos`（m）、`gripper_vel`（m/s）、`gripper_eff`（DM4310 输出轴 N·m）、`gripper_cmd`（m）。共 28 维，默认按 SI 量直接输出。 `include_effort=False` 提供 21 维位置、速度及命令配置。
- `get_command_history_sample()` 提供上一段已执行动作、关节目标速度、仿真采样时刻和传感时间戳。历史在每段动作结束时记录该动作及同一时刻交付的观测。启用延迟时，位置、速度与 effort 使用传感队列同一旧帧，命令仍对应当前发送值。回合重置后，未积满延迟队列时使用重置帧。
- `history.read()` 返回由旧到新的 `[N,15,28]`、`[N,15,7]`、传感时间戳和有效掩码。 `start_try(mask)`、`stop_try(mask)` 控制完整 Try 记录，通过 `read(full_try=True)` 访问。容量按最大 Try 长度配置，`total_frames` 大于容量时表示有溢出的尾部帧。每个环境独立处理掩码和回合边界。
- `get_diagnostics()` 单独提供未裁剪力矩命令、施加力矩、反馈力矩、夹爪施力与求解器约束力。调试时可按世界调用 `contact_torque()`，通过接触约束行求广义力。该路径与 `snapshot()` 会读回主机。Actor 的 28 维只使用 SDK 可对应的状态和已知命令。
- 场景可通过 `add_scene` 扩充 MJCF，用 `compile_model`、`initialize` 接入现有模型编译与初始位置准备。 `before_solve` 接收 GPU 模型和数据，供 Flex 内力或另一个后端的耦合力写入。当前整袋适配器复用仓库 PE Flex 材料与 GPU 铰链弯曲力。Newton/VBD 可从同一接口接入，现有 Newton 整袋实测由 Isaac Lab 线维护。

## 参数与来源

| 参数 | 本实现 | 来源与状态 |
| --- | --- | --- |
| 型号、顺序、方向 | J1–J3 为 DM4340，J4–J6 为 DM4310。CAN ID 1–6，方向全部 +1 | SDK 标准 YAM v1。SDK 从零软件偏移开始，启动时解绕绝对位置。真机机械归零、修订号和电机固件仍待确认。[官方 YAM v1 配置](https://github.com/i2rt-robotics/i2rt/blob/ea116fe4079e991958b22f879874402674ee6590/i2rt/robots/config/yam_v1.yml) |
| 关节位置与速度 | 角域与 SDK MJCF 一致。六轴目标速率 1 rad/s | 项目 Route B 工程上限。MIT 协议速度编码范围分别为 ±10 和 ±30 rad/s，运行许用速度待硬件核对。[官方 MJCF](https://github.com/i2rt-robotics/i2rt/blob/ea116fe4079e991958b22f879874402674ee6590/i2rt/robot_models/arm/yam/v1/yam.xml) · [官方协议常量](https://github.com/i2rt-robotics/i2rt/blob/ea116fe4079e991958b22f879874402674ee6590/i2rt/motor_drivers/utils.py) |
| 轴线、FK 与惯量 | 复用平台 Menagerie CAD、关节链、惯量和 D405。与当版 SDK 三组姿态的轴方向一致，J2–J5 轴线有约 9.5 mm 的位置差。新版 SDK 工具坐标相对旧 `grasp_site` 绕接近轴 −90°，三姿态原点差约 2.9–15.5 mm | 继续使用项目已验证的几何用于仿真。真机侧 TCP 要以实际装配核对，SDK FK 误差随姿态变化。[平台资产](yam_platform/assets/i2rt-yam-menagerie/README.md) · [官方夹爪 MJCF](https://github.com/i2rt-robotics/i2rt/blob/ea116fe4079e991958b22f879874402674ee6590/i2rt/robot_models/gripper/linear_4310/linear_4310.xml) |
| 手臂 Kp / Kd / 力矩上限 | Kp 80/80/80/10/10/10，Kd 5/5/5/1.5/1.5/1.5。六轴逐轴裁剪至 28/28/28/10/10/10 N·m | 前两项为官方控制配置。力矩取 DM MIT 帧满量程及本地执行器限制，单位对应集成电机输出轴和同轴关节。连续热额定、实际固件限流和电流标定待硬件核对。[官方控制配置](https://github.com/i2rt-robotics/i2rt/blob/ea116fe4079e991958b22f879874402674ee6590/i2rt/robots/config/yam_v1.yml) · [官方 MIT 实现](https://github.com/i2rt-robotics/i2rt/blob/ea116fe4079e991958b22f879874402674ee6590/i2rt/motor_drivers/utils.py) |
| 重力补偿 | 逐轴因子 1/1.1/1.1/1.2/1/1。由受补偿连杆的质心、质量与重力计算力矩，只经电机施加一遍 | 因子来自 SDK，质心和有效质量取本地资产。D405 的质量留在刚体动力学中。补偿力矩不含科氏项和接触项。[SDK 电机控制循环](https://github.com/i2rt-robotics/i2rt/blob/ea116fe4079e991958b22f879874402674ee6590/i2rt/robots/motor_chain_robot.py) |
| 阻尼、摩擦、反射惯量 | MuJoCo 原生关节属性。手臂惯量 0.032 / 0.0018 kg·m²，原生摩擦 0.1 N·m，原生黏性阻尼为模型默认 0 | Menagerie 工程参数，待标定。SDK 默认关闭可选的库仑摩擦前馈，YAM v1 配置给出的 0.3/0.3/0.3/0.06/0.06/0.06 N·m 是其补偿系数。[SDK 创建入口](https://github.com/i2rt-robotics/i2rt/blob/ea116fe4079e991958b22f879874402674ee6590/i2rt/robots/get_robot.py) |
| 夹爪 | DM4310，双滑轨耦合，14.4 × 16.4 mm 指垫，总开度 ≤95 mm，力限 50 N，速度上限 0.12 m/s，等效 PD 1000 N/m 和 16.7 N·s/m | 几何来自本地资产。力限取 SDK 默认堵转夹持力。在等效模型中始终限力。工程 PD 和速度待标定。以 SDK 的 96 mm/6.57 rad 计算正向 `τ = F·0.096/6.57`。官方电机端 PD 为 20/0.5，有效开度由启动端点标定。[官方夹爪配置](https://github.com/i2rt-robotics/i2rt/blob/ea116fe4079e991958b22f879874402674ee6590/i2rt/robots/config/linear_4310.yml) |
| 反馈、频率 | SDK 的 effort 解码自 DM 帧 torque 字段。夹爪位置/速度经过端点归一化，原 effort 仍为 N·m。SDK CAN 默认 250 Hz，实测联机速度待确认 | 反馈值由施力经队列构造。可按回合配置传感增益、偏置、噪声、延迟、量化及滤波。当前选择速度/位置/力矩字段。[SDK 驱动](https://github.com/i2rt-robotics/i2rt/blob/ea116fe4079e991958b22f879874402674ee6590/i2rt/motor_drivers/dm_driver.py) · [SDK 观测接口](https://github.com/i2rt-robotics/i2rt/blob/ea116fe4079e991958b22f879874402674ee6590/i2rt/robots/motor_chain_robot.py) |

[名义配置](yam_platform/yam_nominal.json) · [随机化配置](yam_platform/yam_randomization.json)

随机化默认关闭。启用后在各环境重置时采样并冻结参数，分别写入 GPU 控制器、原生关节摩擦/惯量、指垫摩擦、连杆质量/惯量和反馈过程。质量和惯量同倍率，MJWarp 刷新约束所需派生常数。手臂电机 Kp、Kd 与力矩受 SDK 协议上限约束。名义一阶执行器响应、低通和量化均关闭，可在随机化配置开启。参数区间为初始工程候选，全部观测通道保持固定物理单位。

`RealYAMInterface` 提供离线 SDK 字段和归一化开度转换，现场可明确传入测得的开度行程。零样本部署需要真机侧共享相同限位、动作尺度、工具坐标和时间对齐，同时接上急停、超时、关节与工作空间保护及操作者确认。当前完成仿真及离线 SDK 字段验证。

## 复现

```sh
P=.venv/bin/python
O=test/runs/YYYY-MM-DD/yam/yam-smoke
mkdir -p "$O"
for case in asset tracking contact grasp; do
  flock /tmp/claude-gpu.lock env PYTHONPATH=experience "$P" -B test/yam/smoke.py "$case" "$O/$case"
done
flock /tmp/claude-gpu.lock env PYTHONPATH=experience "$P" -B test/yam/smoke.py gpu "$O/gpu-64" --worlds 64
flock /tmp/claude-gpu.lock env PYTHONPATH=experience "$P" -B test/yam/smoke.py gpu "$O/dr-64" --worlds 64 --randomize
PYTHONPATH=experience "$P" -B test/yam/check_sdk.py "$O/sdk-fk.json"
```

[实验报告](../test/yam/REPORT.md) · [开发检查](../reports/yam-pure-sim-test.md)

## ABC Yam_B 工作站

现有 Route B 的 [ABC 场景配置](../configs/route_b_yam_scene.json)、[工作站角色与停驻角](../configs/route_b_workcell.yaml) 和 [单臂篮子尺寸](../configs/route_b_single_arm_bin.yaml) 继续作为场景来源。MJWarp 侧通过 [ABC 场景适配器](yam_platform/abc_workcell.py) 读取相同的桌面、两台基座、顶相机和右腕 D405 帧。抓放验收中 Yam_A 是带碰撞的静态停驻臂，Yam_B 是唯一受控机器。原生 Isaac Lab 双臂装配及 USD 在 [Route B 场景入口](../src/isaac_yam_parcel/yam_scene.py) 和 [已有双臂验收](../reports/route-b-abc-scene-test.md)。

```sh
flock /tmp/claude-gpu.lock env PYTHONPATH=experience .venv/bin/python -B \
  test/yam/smoke.py abc test/runs/YYYY-MM-DD/yam/abc-smoke
```

[ABC 顶视和夹口录像](../test/runs/2026-11-25/yam/pure-sim-r5/media/rigid-grasp-place.mp4) · [同次导出 MJCF](../test/runs/2026-11-25/yam/pure-sim-r5/abc/grasp.xml) · [抓放反馈曲线](../test/runs/2026-11-25/yam/pure-sim-r5/media/grasp-feedback.png)

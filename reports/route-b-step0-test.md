# 路线 B Step 0 盘点与互操作检查

最近检查 2026-11-22 22:32（北京时间）。**环境、设备张量互操作与 ABC 仿真工位资产门禁通过。**

## 原生 Isaac 环境

| 项目 | 本机核验 |
| --- | --- |
| 主机 | `ubun`，RTX 5090，32,607 MiB，驱动 595.80 |
| 解释器 | `/home/ubuntu/workspace/czx/system0/sim/envs/isaacsim/bin/python`，Python 3.12.15 |
| Isaac Sim / Isaac Lab | 6.2.0.0 / v3.0.0（包版本 28.0.0，源码 `e3f7d9ff`） |
| PyTorch / CUDA / Warp | 2.11.0+cu130 / 13.0 / 1.17.0 |
| 同解释器导入 | 重新指向现有源码后 `import torch, isaaclab, isaacsim, src.low_order_parcel` 通过 |
| 冻结来源 | 最终 GPU 复测使用基于 `d57dc10c` 的路线 B 工作树，接触、包裹、双/单臂封装、抗转力矩、随机化、RGB 与配置对 `046384ab` 的 Git 差异为空。主工作树的 RGB 与冻结版有差异。 |

项目旧运行脚本中的 `/home/ubuntu/Dev/Envs/isaaclab` 在本机缺失。使用已安装的 `czx/system0/sim/python.sh` 激活独立环境，并在本次进程的 `PYTHONPATH` 中指定已搬迁的 Isaac Lab 源码和冻结来源的仓库根目录。需要适配的 v3 API 包括 `SimulationContext.has_gui` 已为属性、原生启动器已移除 `--headless`、无显示由禁用 visualizer 实现，以及 `ProxyArray` 的姿态四元数为 XYZW。验证脚本保留入口自己的 `--headless` 并要求 GUI 和离屏渲染均关闭。旧的外力与电流读口有兼容层，当前实现以 wrench composer 和 `actuators.applied_effort` 为新接口。[本机原生环境说明](/home/ubuntu/workspace/czx/system0/sim/README.txt)

| 无渲染 PhysX GPU 探针 | 结果 |
| --- | --- |
| 同进程设置 | RTX 5090 / `cuda:0`，`PhysxCfg`，float32，2 ms。GUI 和离屏渲染均为 false。 |
| PhysX → PyTorch | 位姿 `[1, 7]` 与速度的 PyTorch 视图和 PhysX 返回的 Warp CUDA 缓冲区地址相等。传给冻结包裹核的位姿和速度切片仍共享地址。一次 C6 包裹步进输出 `[1, 2, 6]` 有限腕力矢量。 |
| PyTorch → PhysX | 1 kg 独立探针从 PyTorch 读取 14 N 测试力，Warp 与 PyTorch 地址相等。竖直速度由 −39.2400 变为 −30.8600 mm/s，变化 **+8.379996 mm/s**，解析值 **+8.380000 mm/s**。 |

该探针覆盖 PhysX 设备缓冲区到 PyTorch 的读入、冻结包裹的一次步进，以及 PyTorch 设备缓冲区到 PhysX 的写入。回写使用已知测试载荷，工况级 YAM 闭环待 Step 1。首次使用旧 editable 路径时 `isaaclab.app` 导入失败，改用该进程的源码路径后启动成功。第二轮已跑通并出现向上的速度增量，原断言误要求合速度变为向上。按施力前后的速度差复测通过。[脚本](../test/isaaclab/check_route_b_interop.py) · [原始日志](../test/runs/2026-11-22/isaaclab/route-b-step0-r1/)

## YAM URDF、执行限位和工作站

主资产是 ManiMux 的 I2RT `yam.urdf`，SHA-256 `5b9ccf966aa353ccf1e82ad6e751b899333fab7c0b4d1071316172201da68c4e`。共 9 连杆、9 惯量、9 个可解析的外观网格、6 旋转关节和 2 滑轨，碰撞体、安全控制器软限位、mimic 约束各为 0。另一个 `dream_yam.urdf` 只有 6 个旋转关节。[I2RT 原始 URDF](/home/ubuntu/workspace/xzy/manimux/manimux/embodiments/arm/yam/assets/i2rt/robot_models/arm/yam/yam.urdf) · [DREAM 简化 URDF](/home/ubuntu/workspace/xzy/manimux/manimux/policy_adapter/assets/dream_yam.urdf)

| 关节 | URDF 硬位置限位 | URDF effort / velocity |
| --- | --- | --- |
| `joint1` | −2.61799 至 3.14159 rad | 1 N·m / 1 rad/s |
| `joint2` | 约 0 至 3.66519 rad | 1 N·m / 1 rad/s |
| `joint3` | 0 至 3.14159 rad | 1 N·m / 1 rad/s |
| `joint4` | −1.69297 至 1.57080 rad | 1 N·m / 1 rad/s |
| `joint5` | −1.57080 至 1.57080 rad | 1 N·m / 1 rad/s |
| `joint6` | −2.09440 至 2.09440 rad | 1 N·m / 1 rad/s |
| `joint7` / `joint8` | −46.95 至约 0 mm，物理轴分别为 ±Y | 1 N / 1 m/s |

URDF 统一填 1 的驱动字段与现有仿真配置有差异。项目的 `yam_linear_4310.xml` 使用臂前三轴 ±28 N·m、后三轴 ±10 N·m、夹爪执行器 ±100 N 和单驱动等式联动。它采用不同的滑轨坐标，范围为 −5 至 47.5 mm；这组数值是现有仿真的执行器配置。ManiMux 的通用平滑预设限制臂速 0.25 rad/s、关节绝对位置 3.14 rad。工位级逐关节软限位与实测力矩、速度表仍空缺。[项目参考 MJCF](../experience/yam_platform/assets/i2rt-yam-menagerie/upstream/yam_linear_4310.xml)

| 定义 | 当前可用证据 |
| --- | --- |
| 末端与滑轨 | 原始 URDF 连杆名为 `gripper`、`tip_left`、`tip_right`。后两者是滑轨连杆。原文件没有 `gripper_base`、`left_finger_pad`、`right_finger_pad`。MJCF 有 `left_pad`／`right_pad` 接触盒及壳体碰撞。 |
| 腕相机 | URDF 无 `wrist_camera_link`。MJCF 有 D405 CAD 光学帧 `wrist_camera`，相对 MJCF 父体平移 `[0, 0.070435, 0.077006] m`、四元数 WXYZ `[0, 0, 0.216440, -0.976296]`。双 YAM 配置的左右腕相机 `mount` 均为 `null`，没有工位实测外参。 |
| 双臂工作站 | 双 YAM 配置定义左右臂、相机和关节预置，未给出两臂基座至统一桌面的位姿。研究用单臂 MJCF 基座是 `[0, −0.30, 0.12] m`、绕 Z 转 90°。归档 270° 是该单臂模型下的顶抓目标偏航，待用 Isaac 场景复核。 |

[双臂配置](/home/ubuntu/workspace/xzy/manimux/manimux/configs/embodiment/robot/yam_dual.yaml) · [双臂关节预置](/home/ubuntu/workspace/xzy/manimux/manimux/configs/embodiment/initial_poses/yam.yaml) · [单臂工作台布局](../experience/yam_platform/robot.py#L47) · [270° 试验](../docs/PHYSICS_REALISM_AUDIT_REPORT.md#L28)

初查时缺少 URDF 碰撞体、指垫和工位外参。按用户指定的 ABC 工位补齐后，在冻结提交 `2ef4a394` 持锁复验通过。桌面上表面为 `z=0`，双臂中心距 620 mm，20 个 2 ms 子步后四个指垫内表面离台 104.86–106.52 mm。零拷贝复验读写地址均一致，独立探针速度增量 +8.379996 mm/s，解析值 +8.380000 mm/s。真机部署所需工位实测外参与限位单独确认。[ABC 场景与装配检查](route-b-abc-scene-test.md) · [冻结版复验日志](../test/runs/2026-11-22/isaaclab/route-b-fix41-step0-r1/)

### 并行 YAM URDF 试验

核对工作树 `stage4/fix4-training@e103f1e9` 的 `experience/yam_platform/urdf.py`、`YamPickToAirEnv` 和 `--yam-urdf` 入口。复用 I2RT 关节命名、两轨同号目标以及参考执行器范围。ABC 使用验收后的 USD、CAD 指垫内表面和双臂 270° 起始位姿。并行试验采用 MuJoCo-Warp 和不同桌面布置。其力回写分别有腕部合力／合力矩和双指相向夹持负载两个通道，供 Step 1 对照。[并行提交](https://github.com/pig27777/randomization/commit/e103f1e9) · [ABC 入口](../src/isaac_yam_parcel/yam_scene.py)

并行试验种子 17 的 `training/config.json` 指向同一 I2RT URDF，源文件 SHA-256 与上表相同。在已合入 Fix 4.1 的 `main@ed5fe78c`，两向缓冲区地址对照、已知载荷回写和最终 ABC USD 均重新通过。两臂 270° 初态通过，20 个 2 ms 子步后四个指垫离台 104.887–106.478 mm。[最终资产复验日志](../test/runs/2026-11-22/isaaclab/route-b-step1-r1/)

## 复现

```bash
git diff --exit-code 046384ab -- src/low_order_parcel/{lumped_contact_proxy,unified_payload,dual_arm_parcel_sim,macro_parcel_sim,domain_randomizer,virtual_rgb,pad_torsion}.py configs
route_lab=/home/ubuntu/workspace/czx/system0/sim/IsaacLab/source
route_path="$route_lab/isaaclab:$route_lab/isaaclab_physx:$route_lab/isaaclab_newton:$route_lab/isaaclab_visualizers:$route_lab/isaaclab_ov:$PWD"
flock /tmp/claude-gpu.lock env PYTHONPATH="$route_path" PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/workspace/czx/system0/sim/python.sh -B \
  test/isaaclab/check_route_b_interop.py --headless
```

# 路线 B · ABC 场景装配检查

最近验证 2026-11-22 21:03（北京时间），`ubun` RTX 5090，Isaac Sim 6.2 / Isaac Lab v3.0.0。场景配置和资产已可供单臂及双臂环境加载。 [场景配置](../configs/route_b_yam_scene.json) · [装配与光学坐标实现](../src/isaac_yam_parcel/yam_scene.py)

## 场景定义

以 ABC 台面中心为包裹原点，将源场景平移 `(−0.60, 0, −0.75) m`。左／右臂基座为 `(−0.3475, ±0.310, 0) m`，中心距 620 mm，两臂与桌面共面。顶相机光学中心为 `(−0.51399488, 0, 0.95432053) m`，使用源场景的叠层姿态与 58° 垂直视场角。桌面使用匹配的有限静态盒碰撞，上表面精确为 `z=0`。 [ABC 装配说明](https://abc.bot/hardware.html#assembly) · [固定版 ABC 场景源码](https://github.com/amazon-far/abc/blob/d0832d12651d1b260a652861a14648dc5f3660c7/abc_sim/models/yam_bimanual_empty.xml)

保留 linear 4310 夹爪。腕部 D405 名义外参和指垫来自项目 CAD。通过共同的网格坐标把相机光学帧换算至 URDF `gripper`，把接触盒和指垫内表面换算至 `tip_left`、`tip_right`。桌面、两个固定基座、关节初态和运行时光学／接触帧使用同一配置。相机帧在 CUDA 上由 PhysX 连杆位姿合成，使用环境原点作平移调整，训练入口可直接调用 `YamSceneFrames`。ABC 提供仿真工作站名义外参，真机部署另需目标工位标定。 [linear 4310 CAD](../experience/yam_platform/assets/i2rt-yam-menagerie/upstream/yam_linear_4310.xml)

## 验证结果

| 检查 | 结果 |
| --- | --- |
| CPU 源坐标复算 | 双臂基座与顶相机组合变换通过 |
| URDF 转换 | 自包含 USD，每台 YAM 9 个刚体、11 个碰撞体和 8 个关节 |
| RTX 5090 无渲染 | 双臂固定基座，20 × 2 ms 子步有限，无物理相机 |
| 顶抓初态 | 双臂 PhysX 读回 270°，腕部目标 `(−0.03, ±0.225, 0.25) m`，初态姿态误差小于 0.0001 rad |
| 指垫和光学帧 | 四个末帧指垫内表面高度 104.86–106.52 mm。GPU 光学及接触变换与独立 CPU 计算差小于 1e-6。顶相机可见桌面中心。 |
| 物理范围 | 装配与自由空间短步已检查。真实耦合、抓取、碰撞安全和长程稳定性由 Part II Step 1 验收。 |

首次在仿真进程内转换遇到 Kit 扩展依赖。改为同一原生 Python 环境中离线转换，再由无渲染 PhysX 加载生成的 USD。修正了 ABC/MJCF 的 WXYZ 到 Isaac Lab v3 的 XYZW 顺序，并独立核对基座和腕部姿态。 [CPU 转换日志](../test/runs/2026-11-22/isaaclab/route-b-abc-scene-r1/conversion.log) · [GPU 验证日志](../test/runs/2026-11-22/isaaclab/route-b-abc-scene-r1/scene-verified.log)

## 复现

```sh
native=/home/ubuntu/workspace/czx/system0/sim
site="$native/envs/isaacsim/lib/python3.12/site-packages"
lab="$native/IsaacLab/source"
urdf=/home/ubuntu/workspace/xzy/manimux/manimux/embodiments/arm/yam/assets/i2rt/robot_models/arm/yam/yam.urdf
out=$(mktemp -d)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHONPATH="$PWD:$site/isaacsim/exts/isaacsim.asset.importer.urdf/urdf_prebundle" CUDA_VISIBLE_DEVICES="" \
  "$native/python.sh" -B -m src.isaac_yam_parcel.yam_scene --urdf "$urdf" --output-dir "$out/asset"
route_path="$lab/isaaclab:$lab/isaaclab_physx:$lab/isaaclab_newton:$lab/isaaclab_visualizers:$lab/isaaclab_ov:$PWD"
flock /tmp/claude-gpu.lock env PYTHONPATH="$route_path" \
  "$native/python.sh" -B test/isaaclab/check_route_b_yam_scene.py --headless --usd "$out/asset/yam.usdc"
```

后续环境使用 `table_cfg`、`arm_cfg` 与 `YamSceneFrames` 加载同一场景参数。验证用的 [URDF 转换产物](../src/isaac_yam_parcel/assets/yam.usdc) 已在本机生成。

## 冻结版与 DirectRLEnv 场景入口复验

本轮计划。基于 Fix 4.1 标签 `fix4.1-route-b-base` 持锁复验一次原位 CUDA 读写和一次 ABC 双臂 20×2 ms 自由空间检查，再以 `abc_scene_cfg` 分别启动一次 4 路单臂和 4 路双臂的无渲染 DirectRLEnv。检查每路桌面上表面、双臂基座与环境原点、腕光学帧、指垫内表面和 25 Hz 步进。通过后将该配置作为路线 B 单臂和双臂的场景入口。

最近验证 2026-11-22 21:21（北京时间）。单臂和双臂各 4 路全部通过，另做一次 1,024 路双臂场景启动检查。每个环境的桌面为静态碰撞体，上表面为环境系 `z=0`。扣除克隆环境原点后，基座和腕光学帧符合 ABC／CAD 定义，指垫坐标跨行误差小于 10 µm。每个策略步均执行 20 个 2 ms PhysX 子步，无相机传感器、无 GUI、无离屏渲染。1,024 路的最小指垫离台高度分别为 105.068 和 105.012 mm。[持锁复验日志](../test/runs/2026-11-22/isaaclab/route-b-fix41-step0-r1/)

按上段复现的原生环境再加 `$lab/isaaclab_contrib` 到 `route_path`，运行 `test/isaaclab/check_route_b_direct_scene.py --headless`，可加 `--dual-arm --num-envs 1024`。首次启动发现搬迁后需补上 `isaaclab_contrib`，并改用 v3 的 `cfg.action_space`。PhysX 记录了固定基座关节复制告警，所有基座平移已逐路核对。该版本记录了驱动 CUDA Graph 捕获失败并回退 eager。本轮的结果覆盖场景加载与自由空间，Step 1 的包裹闭环与吞吐另行验收。

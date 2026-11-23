# C5 软衣顶抓穿透复查

最近核验 2026-11-23 17:46（北京时间），ubun，RTX 5090。截图来自 16:14 的 Newton OpenGL 预览。被测物为 PPO 试训的 C5 配置，500 g、240 × 180 × 56 mm、PE 袋、折叠衣物、填充率 60%。原片由 Route B 的脚本控制 PhysX 中的 YAM。训练检查点与该录像的动作来源分别见 [PPO 记录](ppo-pe-soft-edge-test.md) 和 [原录制脚本](../test/runs/2026-11-23/isaaclab/c5-pad-skin-r1/original_recorder.py)。

## 定位与根因

原录制使用单环境、种子 17、2 ms 物理步、25 Hz 控制、7.2 s 和 4 N 滑轨预压。复用录制脚本的包裹、机器人、位姿、抓取参数和动作，复跑的全部 180 个观测与原 JSON 逐值相同。用静止臂配准用户的桌面截图，再比较抓取区域像素，第 84 帧误差最小（平均绝对 RGB 误差 3.74/255，第 85 帧为 4.01）。截图没有内嵌仿真帧号。表内时间是视频 PTS，从第 0 帧起算；物理状态是对应 40 ms 步进后的读数。

1. 原下探目标控制 CAD 内接触面中点，指垫另有约 8 mm 的向下几何延伸。在截图帧和最深帧，冻结核的两指侧接触掩码均为真，顶压分支未加载，轨迹仍继续下探。侧接触解除时，第 2382 个子步（4.764 s）出现 19.849 N 瞬时顶压力。25 Hz 原日志只采到 1.823 N。约 4 N 的电机保持力属于滑轨。
2. 原显示器先旋转四块，再用它们的全部 32 个角点建包围盒。核的台面支撑使用每块中心高度与动态半厚。块组倾斜时，一个远角会高于核上表面，另一个会低于核支撑面。原全段最大顶面偏差 1.455 mm。刚性外盒也没有表示指下凹陷。
3. 新演示使用 CAD 内面下沿作为顶压端口，速度、IK 雅可比和力矩臂随同该端口换算。先在包顶上方完成水平定位，再向折叠衣物局部区域下压；指令下探限于未压包顶以下 20 mm，上一控制帧顶压超过 3.4 N 时上调 Z 指令。压后斜向撤出。反力取冻结核原始输出。3.4 N 是控制触发线，实际峰值另用全部子步检查。
4. 新预览以核的四个竖直高度区间生成外袋。每帧由实际可见手指 CAD 三角形及指垫八角确定凹坑范围，向外加入网格单元余量和平滑过渡，捕获标志控制夹口膜颈。原轨迹和修正后的 C5 实跑都没有双指捕获，本片只验证下压与撤出。内核接触、载荷方程及 C5 材料参数均无本次改动。

## 同帧定量对照

高度和深度单位 mm，力单位 N。高度均以台面为 0。物理顶／底取四块 `中心 z ± half_thickness(d_surf)` 的最大／最小值。旧几何可由每块八角的均值和存档压陷量复核；新读数直接从物理状态输出。视觉顶／底分别取旧盒角点或新蒙皮极值。负的 `depth_phys` 表示已离开顶面。L／R 是抓取臂两个滑轨的顺序。

| 指标 | 原截图 #84 | 修复后 #84 | 原最深 #116 | 修复后 #116 | 修复后最深 #88 |
|---|---:|---:|---:|---:|---:|
| 视频时间 s | 3.36 | 3.36 | 4.64 | 4.64 | 3.52 |
| 物理顶／底 | 53.163／0.000 | 56.000／0.000 | 55.664／0.000 | 56.000／0.000 | 56.000／0.000 |
| 视觉顶／底 | 54.124／−0.961 | 56.000／0.000 | 55.778／−0.114 | 56.000／0.000 | 56.000／0.000 |
| 顶面偏差 | +0.961 | 0.000 | +0.114 | 0.000 | 0.000 |
| 指垫下沿 L／R | 10.483／10.319 | 40.996／41.038 | 7.755／7.754 | 108.242／108.560 | 40.745／40.791 |
| CAD 内面中点 L／R | 18.768／18.771 | 49.275／49.274 | 15.956／15.957 | 116.958／116.952 | 49.032／49.031 |
| 真实压入深度 | 42.844 | 15.004 | 47.910 | −52.242 | 15.255 |
| 竖直顶压反力 | 0.000 | 2.140 | 0.000 | 0.000 | 2.216 |
| 双轨外载 L／R | −0.220／+0.218 | 0.000／−0.069 | −0.006／+0.004 | 0.000／0.000 | 0.000／−0.075 |
| 电机保持力 L／R | +3.998／+3.998 | +3.998／+3.998 | +3.998／+3.998 | +3.998／+3.998 | +3.998／+3.998 |

截图中旧盒的 43.805 mm 和原最深帧的 48.024 mm 是相对**视觉外包络**的插入值。物理口径见上表。静置时两侧预览包顶均为 56.000 mm、包底均为 0。

## 全段验收

修复后全部 180 帧的视觉顶面误差均为 0.000 mm。将顶面三角形与两片 CAD 手指及指垫的保守投影范围逐帧相交检查，最小竖向间隙为 1.550 mm。按全部 3,600 个物理子步另算，最大压入 15.257 mm，指垫到桌面与物理包底的最小间隔均为 40.743 mm，顶压峰值 2.320 N。第 80–88 帧处于 2–4 N 加载带，进入和撤出阶段包含零反力。两侧均未将包裹提起。膜颈仅在捕获标志测试输入上检验了网格生成，本回合未形成承载膜颈。

逐项数值检查及导出使用 `audit_metrics.py`。成片经 `ffprobe -count_frames` 核对为 H.264／yuv420p、1600 × 676、25 fps、180 帧、7.20 s，使用 `ffmpeg -xerror` 全片解码通过。从实际成片抽取第 45、84、145 帧，人工核验左右接近、加载与撤回的时序和遮挡。原始尝试只限中点仍出现过压，另一次按姿态逐帧解 IK 的跨资产回放误差达 8.86 mm；正式左片采用关节原轨迹与固定 CAD 标定，配准误差见下节。

[最终分屏视频](../test/runs/2026-11-23/isaaclab/c5-pad-skin-r1/c5.mp4) · [同帧前后图](../test/runs/2026-11-23/isaaclab/c5-pad-skin-r1/c5.before-after.png) · [成片三帧拼图](../test/runs/2026-11-23/isaaclab/c5-pad-skin-r1/c5.keyframes.png) · [全段指标](../test/runs/2026-11-23/isaaclab/c5-pad-skin-r1/audit.json) · [原片逐帧 CSV](../test/runs/2026-11-23/isaaclab/c5-pad-skin-r1/original-frames.csv) · [修复逐帧 CSV](../test/runs/2026-11-23/isaaclab/c5-pad-skin-r1/repaired-frames.csv) · [ffprobe 读回](../test/runs/2026-11-23/isaaclab/c5-pad-skin-r1/c5.ffprobe.json)

## 分屏与复现

左侧以项目的 MuJoCo 4 原生 flex 双片 PE 薄壳和三段可压缩内容物计算。采用实验平台完整 YAM，在 ABC 标定下逐个 2 ms 重放右侧六轴和双轨读数，滑轨换算为张开正方向，原生 flex 每 0.2 ms 步进一次。按全部子步核对两指 CAD 内面点，最大位置误差 0.00143 mm。左侧总质量同为 500 g，初始化膜外形 280.01 × 185.75 × 56.17 mm；两种模型的外形和材料离散化有差异。原生左片的采样接触穿透峰值 0.732 mm。右侧是 PPO 使用的 C5 配置与程序化外观，动力学为截图对应的 Fix 4.1 Route B。两侧统一机位、时钟和几何着色。影片展示规定动作下的响应，不作为策略成功率或等材料参数的评测。

在配置好项目 Isaac Lab 原生解释器的终端中，从仓库根目录运行以下命令。脚本保留已存在的运行目录；重跑时使用新输出目录。原包裹基线对应 `7d131ac1`。对应文件的冻结模型源码未随此修复修改。

```bash
out=/tmp/c5-preview-reproduction
mkdir -p "$out"
native=/home/ubuntu/workspace/czx/system0/sim
lab="$native/IsaacLab/source"
export PYTHONPATH="$lab/isaaclab:$lab/isaaclab_physx:$lab/isaaclab_newton:$lab/isaaclab_visualizers:$lab/isaaclab_ov:$lab/isaaclab_contrib:$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
flock /tmp/claude-gpu.lock "$native/python.sh" -B test/isaaclab/record_c5_preview.py "$out/right" --viz newton_gl
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -B test/isaaclab/replay_c5_flex.py "$out/right/geometry.npz" "$out/flex.npz"
flock /tmp/claude-gpu.lock "$native/python.sh" -B test/isaaclab/record_c5_preview.py "$out/left" --viz newton_gl --flex-surface "$out/flex.npz" --reference "$out/right/geometry.npz"
run=test/runs/2026-11-23/isaaclab/c5-pad-skin-r1
.venv-physical-parcel/bin/python "$run/inspect_video.py" "$run/original.mp4" "$out/right/simplified.mp4" "$out/left/flex.mp4" "$out/c5.mp4"
```

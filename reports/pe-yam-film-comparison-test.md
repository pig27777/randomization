# UMI 实测尺寸的 PE 折叠衣物包悬空验帧

最近核验 2026-11-23 18:03，机器 ubun。本轮交付 8.00 s 悬空单帧，两侧各有一个夹口特写。整段视频等待外观确认。清晰度检查确认双侧特写与 1920 × 828 PNG 可读。[左右分屏原图](../test/runs/2026-11-23/bag/umi-pe-yam-hold-r1/hold_comparison.png) · [逐边及包覆测量](../test/runs/2026-11-23/bag/umi-pe-yam-hold-r1/hold_measurements.json)

| 8.00 s 实测 | 原生 MuJoCo flex 薄壳 | 四块代理 + 外观蒙皮 |
| --- | ---: | ---: |
| 指尖夹点到衣物顶面垂直距离 | 9.65 cm | 10.03 cm |
| 水平投影短边 × 长边 | 23.77 × 33.44 cm | 23.04 × 34.32 cm |
| 全部 1,056 条唯一网格边的最大 L/L₀ | 1.000085 | 1.039342 |
| 指尖至四个边中点连线的倾角范围 | 33.7°–40.7° | 31.2°–42.2° |
| 袋面穿入衣物的三角形 / 袋外衣物角点 | 0 / 0 | 0 / 0 |

两侧平铺袋均为 **39 × 29 × 2 cm**，取实测区间内的 **31 × 22 × 2 cm** 衣物。按平铺长宽计算，单侧余量分别为 4.0 和 3.5 cm。总质量沿用此前 65 g，用户本轮提供的是几何实测。取指尖夹点中点的世界 z 与其正下方衣物上表面的 z 之差。以衣物水平长短轴投影整个外袋网格取跨度；JSON 同时保留世界 XY 包围盒。L₀ 取两张平铺 PE 膜的材料边长，包括封边；投影、物理和渲染使用实际三角网格。

**左侧**固定 YAM 于共享轨迹的 8.00 s 位姿，在实测指垫间夹住 9 × 22 mm 的中心膜片，并沿两指真实姿态折入夹缝，原生 MuJoCo flex、三段自由衣物、服装边缘圆角接触和重力独立求解 8 s。从预包覆的悬挂状态开始，边缘自由运动，衣物仅靠接触受力。PE 弯曲使用 20 MPa、60 μm、泊松比 0.3。每条材料边设单向长度上限，积分步长 0.10 ms、120 次求解迭代上限，加载瞬态的物理节点位置及速度另有边长投影。每个接纳子步测量边长，整个 8 s 共 80,000 步，最大比为 **1.031507**；0 步触发补充投影。求解器投影前的最大试探值为 1.031507。本次左图验收的是已夹住后的形态，重新从台面自主抓起的动态回合仍待录制。后端为 MuJoCo flex。[实现](../test/runs/2026-11-23/bag/umi-pe-yam-hold-r1/native.py) · [原始稳态与积分记录](../test/runs/2026-11-23/bag/umi-pe-yam-hold-r1/native_pad_8s.npz)

**右侧**从台面回放相同 YAM 轨迹至 8.00 s，冻结核的两侧都已捕获，台面支撑力为零。以实测间距标定柔索原长 98 mm，跟随四块实际质心、共同材料坐标系及各块厚度。蒙皮对夹口施加有限宽度约束，进行 30 轮边长与四块三角面防穿模投影。独立使用三角形–盒体分离轴和角点绕数检查包覆。代理物理保留四块和两条膜颈，右侧边长指标来自其外观网格。冻结文件 `lumped_contact_proxy.py` 与 `unified_payload.py` 经 Git 对比与 `2ef4a394` 一致。[右侧物理回放](../test/runs/2026-11-23/bag/umi-pe-yam-hold-r1/proxy_episode.npz) · [PBD 与检查](../test/runs/2026-11-23/bag/umi-pe-yam-hold-r1/project.py)

两侧使用相同 CAD、灯光、PE 纹理、主机位与特写，ubun 上 CPU 离屏 EGL / llvmpipe。两次正式物理测量均使用 `flock /tmp/claude-gpu.lock`。原始旧视频第 200 帧属于代理蒙皮，最大边长比 3.3911、低指垫至最高内容物顶面 4.35 cm；原径向拉升方法已排除。[旧片诊断](../test/runs/2026-11-23/bag/pe-yam-film-comparison-r1/diagnostics/rejected-skin-8s.json)

## 复现单帧

从仓库根目录运行。新建输出目录，本轮轨迹、网格和检查结果均已保留。源码中的相对路径用于读取 YAM CAD 和模型配置。

```sh
N=test/runs/2026-11-23/bag/umi-pe-yam-hold-r1
D=test/runs/2026-11-23/bag/pe-yam-film-comparison-r1
P=.venv-physical-parcel/bin/python
O=$(mktemp -d)
flock /tmp/claude-gpu.lock "$P" -B "$D/proxy.py" "$N/q_ref.npz" "$O/proxy_episode.npz" --cloth-xy .22 .31 --neck-length .098
flock /tmp/claude-gpu.lock "$P" -B "$N/native.py" "$O/native.npz" --seconds 8 --dt .0001 --patch-half .0045 --bend-factor 1 --initial-gap .0935
"$P" -B "$N/project.py" "$O/proxy_episode.npz" "$O/proxy.npz"
"$P" -B "$N/compare_still.py" "$O/native.npz" "$O/proxy.npz" "$O/hold_comparison.png"
git diff --exit-code 2ef4a394 -- src/low_order_parcel/lumped_contact_proxy.py src/low_order_parcel/unified_payload.py
```

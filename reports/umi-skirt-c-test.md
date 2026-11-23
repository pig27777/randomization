# UMI 厚裙 C 同轨迹悬挂检查

2026-11-23 18:50（北京时间），ubun。只渲染 t = 8.00 s 单帧。左右使用同一 2 ms 关节与夹爪文件（SHA-256 为 13bf5fde405eacc18898bc265460ae0e83d5e9679e310b31592276a38cb20707），同一 YAM CAD、桌面、PE 着色、机位和灯光。按实测使用 39 × 29 cm 平铺 PE 外袋和 2 cm 厚裙。实测未给质量与裙子独立平面尺寸，两侧统一取 500 g 和 34.5 × 24.5 cm。左侧求解器是 MuJoCo 4 原生 flex。本机盘点未发现可运行的 NVIDIA FleX 后端。

| 8.00 s | 抓点到底 cm | 悬空长 × 宽 cm | 最大边长比，平铺 / 初始包覆 | 四条主测线角度 | 膜穿入 / 外露角点 |
|---|---:|---:|---:|---:|---:|
| MuJoCo flex | 6.13 | 38.09 × 28.16 | 1.00134 / 1.00250 | 13.4–17.5° | 0 / 0 |
| 冻结代理与 PBD 外袋 | 6.00 | 37.98 × 27.69 | 1.02500 / 1.02500 | 12.0–14.6° | 0 / 0 |

高度按两指内面下沿中点到全部袋膜节点最低点计算，长宽按工位水平轴投影。角度取夹点到四个轴向上表面测点的直线与水平面的夹角。主测线角度逐项保留于 JSON。最大伸长在每条唯一三角边上独立对平铺尺寸和实装初态取比。左侧对实际裙片凸网格做三角形相交与包围检查，右侧对 4 个实测子块的有限完整长方体做夹剪和角点绕数检查。两个抓点读数的差为 0.00000 mm。

左侧在 t = 2.5908 s 关合后，有限宽度的膜片和衣物中心由抓取约束随动，其余裙片及袋膜在重力和接触下求解。它使用抓牢后的理想夹持约束，录像不能独立证明摩擦抓取能力。PE 膜面内与弯曲共同求解，另有限制受拉的边长约束。裙片弯曲系数与袋内有效压力是按 C 拟合的参数。前期仅以单向边限配合弯曲的运行出现脱离台面或局部收缩，已排除。较早的静态挂点图未用于本次抓起验证。左侧积分全程最大边长比 1.04307。

右侧直接夹衣使用零名义自由膜颈、有限指垫和每侧 92.5 N/m 局部等效载荷路径。四节点由冻结模型步进，蒙皮以 Gaussian 权重传递子块运动，再对材料边与四块碰撞体交替投影 30 次。C 的投影高度、裙片刚度和平面尺寸并未写入训练参数。之前使用外围楔形裙片代替完整代理块的诊断渲染未作验收依据。

在 6.56–9.56 s 两侧持续抓住，左侧最低膜节点净空至少 22.55 cm，右侧最低物理子块净空至少 22.83 cm、台面支撑力为零。右侧质心最大速度 0.458 mm/s。左侧 25 Hz 质心差分速度最高 0.184 mm/s。单回合 CPU 求解（持共享锁）左侧 174.67 s、137.4 步/s，右侧 10.24 s、468.9 步/s。两种积分步长分别为 0.4 ms 和 2 ms，测速不含渲染。

## 文件与复核命令

[8 秒单帧](../test/runs/2026-11-23/bag/umi-skirt-c-r1/skirt_c_8s.png) · [数值与独立几何检查](../test/runs/2026-11-23/bag/umi-skirt-c-r1/hold_checks.json) · [四模态 DR 检查](umi-four-modes-dr-test.md)。仅完成单帧复核，整段视觉蒙皮逐帧伸长与穿模仍待检验。旧 65 g 单侧视频的锥形失真已记录。DR 字段保持只读。以 git diff 对冻结基线 2ef4a394 比较两份核心文件，差异为零。

从仓库根目录运行，输出到独立目录。

```sh
N=test/runs/2026-11-23/bag/umi-skirt-c-r1
B=test/runs/2026-11-23/bag/pe-yam-film-comparison-r1
P=.venv-physical-parcel/bin/python
O=$(mktemp -d)
flock /tmp/claude-gpu.lock "$P" -B "$B/proxy.py" "$N/q_ref.npz" "$O/right.npz" --cloth-xy .245 .345 --direct-textile
flock /tmp/claude-gpu.lock "$P" -B "$N/run_native.py" "$O/native.npz" --seconds 9.6 --dt .0004
"$P" -B "$N/skin_blocks.py" "$O/right.npz" "$O/box-fit.npz"
"$P" -B "$N/compare_c.py" "$O/native.npz" "$O/box-fit.npz" "$O/skirt_c_8s.png"
"$P" -B "$N/check_hold.py" "$O/native.npz" "$O/box-fit.npz" "$O/right.npz" "$O/skirt_c_8s.png"
```

## 正式分屏复核

2026-11-23 19:28（北京时间），ubun。每侧复用一次已归档的 9.6 s 动力学实跑。左侧沿用 MuJoCo 原生薄壳和有限夹持约束。右侧保留同一冻结代理输出，蒙皮在离台过渡时保持短边有内收余量、底片随四块起抬后再随夹口摆动。第一轮逐帧检查发现离台过渡最大边长比 1.1249。修正过渡后，8.00 s 右侧节点与上一轮单帧相差小于 0.0004 mm。

| 项目 | 验证结果 |
| --- | --- |
| 左侧原生薄壳全部物理子步 | 最大边长比 1.04307 |
| 右侧蒙皮全部显示帧 | 对平铺和初始包覆的最大边长比均为 1.02796，膜与四块相交数和外露角点数均为 0 |
| 6.56–9.56 s 稳悬 | 左 / 右可见袋底最低净空 225.5 / 226.6 mm，质心峰值速度 0.110 / 0.458 mm/s |
| 编码及读回 | H.264 / yuv420p，1920 × 720，25 fps，240 帧，9.60 s。FFmpeg 解码全段通过 |
| 从成片抽取 | 双闭合 2.60 s、双离台 3.84 s、稳悬 8.00 s |

右侧袋底在台面接触期间最深视觉重叠 0.66 mm。稳态薄壳最陡测线 17.5°。历史空载 GPU 数据在片头与本次单环境 CPU 吞吐分别标注。FleX 需求仍由本机的 MuJoCo flex 薄壳替代。冻结方程与训练 DR 配置经 diff 核查一致。

[完整分屏 MP4](../test/runs/2026-11-23/bag/umi-skirt-c-video-r1/compare_flex_vs_proxy_side_by_side.mp4) · [三个关键帧拼图](../test/runs/2026-11-23/bag/umi-skirt-c-video-r1/keyframes.png) · [逐帧检查汇总](../test/runs/2026-11-23/bag/umi-skirt-c-video-r1/video_audit.json) · [编码读回](../test/runs/2026-11-23/bag/umi-skirt-c-video-r1/ffprobe.json)。

从仓库根目录执行，可复用当前已存的动力学轨迹。两个单侧视频和分屏均在同机位离屏以 CPU Mesa llvmpipe 渲染，投影与验收由正式运行锁保护。

```sh
D=test/runs/2026-11-23/bag/umi-skirt-c-video-r1
P=.venv-physical-parcel/bin/python
O=$(mktemp -d)
flock /tmp/claude-gpu.lock "$P" -B "$D/make_video.py" prepare "$O"
LP_NUM_THREADS=6 "$P" -B "$D/make_video.py" left "$O"
LP_NUM_THREADS=6 "$P" -B "$D/make_video.py" right "$O"
"$P" -B "$D/make_video.py" compose "$O"
ffprobe -v error -select_streams v:0 -show_streams "$O/compare_flex_vs_proxy_side_by_side.mp4"
ffmpeg -v error -xerror -i "$O/compare_flex_vs_proxy_side_by_side.mp4" -f null -
```

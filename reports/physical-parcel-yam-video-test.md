# 双膜片YAM录像与穿透修复检查

完成于2026-11-16 14:08（Asia/Taipei），机器ubun。视频已导出，抓取结果为失败，整体物理验收尚未通过。

## 修复与检查

- 旧初始化仅推开膜顶点，膜面采样切入盒体1.2539 mm、40面相交、8角外露。新增三角面分离初始化、完整分离轴相交及角点包含检查，最大初始边应变0.4861%。
- 首次静置复核的反力刷新遗漏MuJoCo4离散有效度量阶段，支撑读数0或NaN。改用完整加速度阶段刷新，新增“不推进时间且反力与step一致”的回归，通过复核。
- 初态修复后的YAM r2在2.220025秒出现3.8156 mm几何重叠并自动停止。定位到box—triangle专用路径漏掉边—面穿越，同状态原box漏碰两面，同形凸mesh对全部13个相交面生成接触。内容物与压板采用原生八顶点凸mesh盒体接触。
- 共享锁下执行 `python -B -m unittest discover -s test/bag_model -p test_physical_parcel.py -v`，11/11通过。新小夹具首次缺flexcomp名称而失败，补齐后通过。
- `physical_parcel_checks.py --case support --output .../physical-parcel-mj4-face-clearance-r3`，一次0.3秒静置通过，10 ms采样几何相交0、角点外露0，支撑3.20137 N、误差0.2001%、警告0、裸物—桌面接触0。

源码及文档范围的`git diff --check`、运行源码快照一致性和新报告链接存在性检查通过。全量暂存检查发现3份平台生成的XML各有3处行尾空格，原始实验快照保留这些空白。

## 视频结果

通过统一YAM平台运行r3，CPU MuJoCo4.0.1、dt=25 μs、0.3 kg盒、350×250 mm袋、夹紧目标10 N、pinch_coeff=0。仿真4.46855秒进入lift后，双侧指尖接触丢失超过50 ms，结果`grasp_fail`，盒子留在桌面。224次20 ms采样中膜面—内容物相交0、角点外露0，原生全接触穿透峰值0.25624 mm，墙钟729.29秒。

Mesa llvmpipe回放视频为20fps、90帧、4.5秒、1倍速、1280×480双视角，源状态20 ms并线性插值。已用本机已有FFmpeg完整解码，退出0，并检查首帧与合爪末帧。编码器缺少drawtext，省略共享通用3D标签。

当前证据限于采样时刻和这一配置，稳定摩擦承载尚未实现。当前凸mesh版本的GPU速度和容量未重测，旧8192环境数字对应旧构型及box路径。

[视频](../test/runs/2026-11-16/bag/physical-parcel-mj4-yam-video-r3/yam-attempt.mp4) · [结果](../test/runs/2026-11-16/bag/physical-parcel-mj4-yam-video-r3/trial/result.json) · [视频检查](../test/runs/2026-11-16/bag/physical-parcel-mj4-yam-video-r3/video-verification.json) · [主题报告与命令](../test/bag_model/PHYSICAL_PARCEL_REPORT.md)

## 同版抓起补充（2026-11-16，历史记录）

用户明确要求观看同形凸mesh盒体接触的这版双膜片袋实际抓起效果。静态指尖box/凸mesh对照未发现漏碰，前次闭合末实际间隙3.5436 mm，夹紧不足后进入lift。当前使用15 cm高、70×50 mm窄支撑摆放，右侧yaw90°、向下30°接近，选用共享平台已有approach接口。内容物0.3 kg、膜材料、接触摩擦与10 N目标采用该模型配置，承载由指尖—膜—盒体接触传递。

- 初始8姿态检查，1组可达、7组按原IK阈值不可达，可达组位置误差0.002071 mm、姿态误差0.000125°。
- 新增摆放、支撑尺寸、几何与配置回归，13/13通过。
- 支撑同样使用同尺寸凸mesh，记录膜—支撑几何深度，1 mm门槛同时检查内容物与支撑。首个支撑box入口在approach早段主动停止，结果保留。
- 正式r2目录为 `test/runs/2026-11-16/bag/physical-parcel-mj4-yam-raised-side-r2/`，CPU MuJoCo4.0.1、dt25 μs，6.65005秒lift_hold因仍有支撑而grasp_fail，实际视频保留。

## 最外缘封口与随机化录像（2026-11-16 20:16，ubun）

已生成 `kraft_edge_sealed_35.npz`，原正反膜沿最外轮廓共享34个封边节点，142节点、280面、420边、边界边0、Euler数2，初始最大边应变0.8262%。16项回归通过，新增断言最初对列表使用abs而报错，修正为np.abs后通过，首次日志保留。该回归发生在随机化接口及压紧配置修改之前。

新封口版0.5秒静置通过，膜—盒体及支撑几何相交0。四次原闭合限制的抓取均失败，r2/r3仅提动约1.24 cm后滑脱，原生接触峰值分别1.289/1.749 mm，超出1 mm预算，作为诊断记录。用户已查看r2真实视频。共享程序闭合指令下限0阻止使用模型原有负目标压紧行程，用户已授权本次使用单轨5 mm余量及35 cm抬升。两者通过配置设置，默认分别0和25 cm，原10 N反馈目标及驱动力范围继续使用。

用户要求减少测试并追加袋子、内部物体共同随机化和顶面搓起，本轮缩为两次录制。种子20261116，袋尺寸/E/厚度/密度/摩擦/阻尼及物体尺寸/质量/弯曲铰链随机采样，Et、D和面密度关联计算。组1刚体侧抓，组2三段可弯曲物体顶抓，类型由演示安排指定。仅运行每组必要的0.5秒准备、初态检查及10秒以内真实轨迹，复用共享平台与视频入口。参数是工程预设，结果按实际阶段、抬升和接触记录报告。

[封口几何与回归](../test/runs/2026-11-16/bag/physical-parcel-edge-sealed-checks-r1/) · [随机化配置与日志](../test/runs/2026-11-16/bag/physical-parcel-edge-sealed-randomized-r1/) · [主题报告](../test/bag_model/PHYSICAL_PARCEL_REPORT.md)

两次录制已完成，准备均通过，20 ms采样的膜—内容物/支撑几何相交0、角点外露0、警告0，源码快照哈希匹配。侧抓刚体348 g，最高抬升21.03 cm，满足50 mm/0.05 N条件的连续悬挂2.74秒，10秒窗口止于carry、incomplete。原生接触峰值1.1972 mm超出1 mm预算，跟踪峰值153.84 mm，整体物理验收仍未通过。顶抓为167 g三段可弯曲体，5.0333秒lift接触丢失、grasp_fail，未抓起，原生接触峰值0.1215 mm。

视频包含袋与内部物体参数、最高实际抬升及真实结果。两段及连续视频均已用FFmpeg完整解码，退出0，并查看侧抓悬挂和顶抓末帧。1280×600、20fps、1倍速，连续视频约15.15秒。`git diff --check`通过。本轮按用户要求控制为两次录制，实际配置和命令保存在同一运行目录。

[随机化连续视频](../test/runs/2026-11-16/bag/physical-parcel-edge-sealed-randomized-r1/randomized-bag-and-contents.mp4) · [顶面单段](../test/runs/2026-11-16/bag/physical-parcel-edge-sealed-randomized-r1/case-2-top/top-randomized.mp4) · [视频与几何检查](../test/runs/2026-11-16/bag/physical-parcel-edge-sealed-randomized-r1/verification.json)

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

## 同版抓起补充（2026-11-16，运行中）

用户明确要求观看同形凸mesh盒体接触的这版双膜片袋实际抓起效果。静态指尖box/凸mesh对照未发现漏碰，前次闭合末实际间隙3.5436 mm，夹紧不足后进入lift。当前使用15 cm高、70×50 mm窄支撑摆放，右侧yaw90°、向下30°接近，选用共享平台已有approach接口。内容物0.3 kg、膜材料、接触摩擦与10 N目标采用该模型配置，承载由指尖—膜—盒体接触传递。

- 初始8姿态检查，1组可达、7组按原IK阈值不可达，可达组位置误差0.002071 mm、姿态误差0.000125°。
- 新增摆放、支撑尺寸、几何与配置回归，13/13通过。
- 支撑同样使用同尺寸凸mesh，记录膜—支撑几何深度，1 mm门槛同时检查内容物与支撑。首个支撑box入口在approach早段主动停止，结果保留。
- 正式r2目录为 `test/runs/2026-11-16/bag/physical-parcel-mj4-yam-raised-side-r2/`，CPU MuJoCo4.0.1、dt25 μs，完整动作运行中。目标为内容物离支撑≥50 mm、总支撑<0.05 N，并交付实际录像。

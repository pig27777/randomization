# 三档袋参数顶面完整流程录像检查

最近检查时间为2026-11-16 20:50 CST，机器ubun，MuJoCo4.0.1 CPU。

## 结果

三档完整动作视频和同步对照视频已生成，4/4完整解码通过。三组顶面搓起均失败，起抬后续为空手动作。

## 范围与方法

- 三档E为4/7/10 MPa，厚度100/150/200 μm，统一内容物与顶面动作。每档一次0.5秒准备、一次原停止流程录制、一次完整动作补录，正式动力学均持共享锁。三档步长与显式弯曲条件的模型编译检查通过。
- 原流程在约5.03秒起抬时停止。为满足用户完整流程要求，共享平台增加 `complete_motion` 录像选项，默认False，保留首次抓取失败结果并继续原动作，失败后下降走完原高度。数值、几何及挤压停止仍生效。
- 运行采用 `prepare_physical_parcel.py`、`record_physical_parcel.py --config ... --output ...` 与 `render_physical_parcel_video.py`。逐档完整参数、命令、日志、源码和机器人资产指纹保存在运行目录。
- 对三档result、trace逐项核对，`motion_completed=true`，九个阶段全部覆盖，首次失败判定保留。默认选项、语法解析、源码快照SHA256及三档源码一致性检查通过。
- 使用运行配置记录的FFmpeg执行 `-i <视频> -map 0:v:0 -progress pipe:1 -f null -` 完整解码，全部退出0。三段259/258/258帧，同步对照259帧，均20fps。查看接近、搓合、最终松爪抽帧与第6秒失败提示。
- `git diff --check -- experience/yam_platform/protocol.py experience/README.md test/bag_model/PHYSICAL_PARCEL_REPORT.md` 通过。未新增或执行独立单元测试。

## 关键数据与限制

| 档位 | 首次失败/s | 完整仿真/s | 内容物最高抬升/mm | 原生穿透峰值/mm | 夹紧滤波峰值/N |
| --- | --- | --- | --- | --- | --- |
| 薄软 | 5.0393 | 12.8758 | 0.000678 | 0.132806 | 0.732 |
| 中等 | 5.0294 | 12.8650 | 0.011549 | 0.097825 | 0.680 |
| 厚硬 | 5.0321 | 12.8678 | 0.019546 | 0.066169 | 0.721 |

采样膜—内容物及支撑几何深度0、角点外露0、数值警告0、状态有限。三组抓起指标均未通过，机械臂跟踪峰值169–170 mm，夹紧滤波峰值约0.68–0.73 N。材料采用工程预设，未做实物标定。同步对照末尾最多定格0.05秒，视频状态由20 ms轨迹采样插值。

[主题报告与复现入口](../test/bag_model/PHYSICAL_PARCEL_REPORT.md) · [原提前停止记录](../test/runs/2026-11-16/bag/physical-parcel-top-parameters-r1/) · [完整录像与证据](../test/runs/2026-11-16/bag/physical-parcel-top-parameters-r2/) · [解码记录](../test/runs/2026-11-16/bag/physical-parcel-top-parameters-r2/video-verification.json)

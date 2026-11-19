# experience

统一的 YAM 抓取实验平台。机械臂、夹爪、IK、抓取动作、角度矩阵、事件判定、轨迹记录、视频回放和 CPU 测速集中在 randomization 的 experience 目录；袋子和内容物通过模型适配器接入。

## 代码地图

| 文件 | 职责 |
| --- | --- |
| `yam_platform/robot.py` | 完整 YAM linear_4310、数值 IK、有限驱动力、跟踪与碰撞诊断 |
| `yam_platform/geometry.py` | 20×40 mm、2 mm 圆角指尖 |
| `yam_platform/protocol.py` | 公共场景装配、接触测量、v2.2 抓取阶段与结果记录 |
| `yam_platform/pinch.py` | 既定 v1.3 聚拢弹簧；材料连接由模型适配器提供 |
| `yam_platform/angles.py` / `run_suite.py` | 单次试验、14 条件行为矩阵、自定义配置列表 |
| `yam_platform/benchmark.py` | 同机串行与 28 进程 CPU 测速 |
| `yam_platform/render_video.py` / `render_comparison.py` | 已保存轨迹的 CPU 视频回放与并排比较 |
| `yam_platform/export_tetra.py` / `tetra_view.html` | 四面体状态的离线播放 |
| `yam_platform/rigid_control.py` | 不依赖外部内容物代码的最小适配器 |
| `yam_platform/assets/i2rt-yam-menagerie/` | YAM 自身的完整 XML、网格、来源和许可 |

## 运行

本目录只归并已有统一实验平台，并包含 YAM 自身的资产包。其他袋子、内容物及传送带资源不纳入。

在 `experience/` 目录，使用已有 MuJoCo 环境即可，不必安装包。从 randomization 中可使用 `../.venv/bin/python`。输出目录必须不存在。

```sh
../.venv/bin/python -B -m yam_platform.angles --smoke-seconds .02 --output outputs/entry-check
../.venv/bin/python -B -m yam_platform.run_suite behavior --output outputs/matrix
flock /tmp/claude-gpu.lock ../.venv/bin/python -B -m yam_platform.benchmark --output outputs/speed
../.venv/bin/python -B -m yam_platform.render_video outputs/entry-check outputs/entry-check/grasp.mp4
```

默认使用刚体入口对照，检查装配流程；这不代表硬物顶面抓取验收通过。`--smoke-seconds` 只检查入口，不给出抓取结论。正式运行前写实验计划，并按宿主项目要求持共享锁。视频回放使用 Mesa llvmpipe，系统还需 FFmpeg、Mesa/EGL；Python 渲染依赖见 `pyproject.toml` 的 `video` 可选项。

也可在独立虚拟环境安装 `pip install -e .`，使用 `yam-grasp`、`yam-suite`、`yam-benchmark`。平台代码不导入 randomization 的材料模型，直接加载本目录内的 YAM 资产。

## 模型适配器

`--backend 可导入模块名` 选择模型。最小例子是 `yam_platform.rigid_control`；适配器提供以下接口：

| 接口 | 作用 |
| --- | --- |
| `Config(TrialConfig)` | 共享选项及模型参数 |
| `scene(cfg)` | 提供 `built.xml/desc/cfg`，调用 `attach_robot`；仿真对象提供 `m/d/step()/com()` |
| `measure(robot)` | 双指接触、支撑、顶面接触、穿透；可复用 `measure_contacts` |
| `attachment(robot,cfg)` | `active/load/events/try_bind/check_load/release`；无捏取也须提供空实现 |
| `check_state(sim)` | 返回失效原因或 None，以及健康指标字典 |
| `summarize(trace)` / `dimensions(sim)` | 模型指标与规模 |
| 可选 `source_files()` | 模型依赖源码路径字典，追加到快照和哈希 |
| 可选 `RELEASE_SHA` | 冻结内容物版本指纹；无冻结版本时记录 null |
| 可选 `refresh(sim)` | 刷新采样状态，默认 MuJoCo forward |
| 可选 `record_state(sim)` / `STATE_SPEC` | 记录有材料历史的完整 integration state |
| 可选 `built.compile_model(xml)` / `save_model(sim,output)` | 模型专用编译及 MJB 保存；视频优先读 MJB |

直接调用 `run(..., on_sample=callback)` 可观察已记录样本，不参与控制。`simulation_seconds` 默认为 30 秒。其他引擎需先提供耦合适配，不能仅更换模块名。

原 `test/flex3d/angles.py`、矩阵、回放、测速入口保留为兼容转发，默认仍为原 native-flex 模型。`native_flex.py`、`segmented.py` 和 `contents/continuum/grasp_adapter.py` 留在模型所属项目。策略线 `policy/yam_probe_robot.py` 只适配冻结内容物场景，机械臂实现复用本目录。历史结果与快照保持原样。

## 固定协议与边界

本次归并保持 270 mm 安装高度、原 IK/有限力矩、指尖、步长策略、10 N 目标力、v2.2 动作和判定不变。默认行为矩阵是两类×6 姿态，加两类纯接触对照，共 14 次。平台可运行不等于全部角度可达、模型抓取可靠或材料完成实物标定；`setup_fail`、`compute_limit`、`execution_error` 单列，行为耗时不作吞吐数据。

每次新运行保存实际共享平台和模型源码、SHA256、机器人资产指纹、模型、接触轨迹、关节状态与结果。旧目录拒绝覆盖。

抬升行程通过 `--lift-height` 设置，默认0.25 m，速度0.1 m/s，保持与搬运阶段使用同一高度。2026-11-16用户授权袋子演示单次使用0.35 m，运行配置保存实际行程。

`--gripper-overtravel` 设置单轨零位后的闭合余量，默认0 m，并校验模型已有控制范围。2026-11-16用户授权本次薄膜演示使用0.005 m，10 N反馈目标和原驱动力限制继续生效。

袋模型顶部试验可在运行 JSON 中设定 `press_target_N` 与 `press_timeout_s`，启用法向力反馈。`press_start_offset_m` 表示初始抓取点高于目标表面的距离，默认 0.10 m。以实测和指令折算指尖中较低者为准，距表面 2 mm 时切换低速。达到目标力的 80%–120% 并持续 10 ms 后进入闭合；窗口内指垫直接支撑反力须 ≤0.05 N。诊断用 `press_only=true` 达标即结束。适配器需在 `measure()` 中单列 `direct_support_N`，`physical_parcel_yam` 已提供。默认 `press_target_N=null` 使用既有接近动作。

`physical_parcel_yam` 支持 `mode="empty"` 顶抓，同一两片封口资产仅生成袋膜，空袋质心按膜节点质量计算。运行 JSON 可显式设定 `ccd_tolerance_m`，该值随模型与准备状态签名归档。

同一袋型可用 `bag_profile` 保留外观和接触皮肤，用 `material` 指定 `young_pa`、`thickness_m`、`density_kg_m3` 和可选的 `poisson`。模型用同一组材料参数计算膜拉伸刚度、弯曲刚度与面密度。

`payload_offset_xy` 以米指定内容物相对袋中心的平面偏置。选项进入包覆和内容物装配，封口参考坐标与刚度设置沿用原袋参数。

`close_gap_hold_s` 可为到达闭合指令下限后的等待配置时间，以便实际夹口跟踪。默认 0 s。达到持续夹持力或闭合时限时仍会进入抬升。

算力试验可在 JSON 中使用 `pickup_only=true` 在提拉与悬停后结束。`grasp_opening_m` 与 `lift_speed_m_s` 默认分别为 0.040 m 和 0.10 m/s，仍按原桌面支撑和失触条件判定 `pickup_ok`。

整袋松夹试验可在原生接触的 `pickup_only` 模式设置 `pickup_release_seconds`。完成离台悬停并确认两指承载后，目标开口按该时长线性回到抓取前开口，松夹轨迹每 2 ms 采样。持续 50 ms 双指失触后记录 `pickup_release_ok`。

独立侧抓工况可设 `side_insert_speed_m_s` 与 `closing_gain_m_per_N_s`，默认分别为 0.01 m/s 与 0.002 m/(N·s)。插入深度、夹持力目标与闭合速度限值沿用平台设置。

## 来源

公共流程迁自 randomization 的 `test/flex3d/`，机械臂迁自 `policy/yam_probe_robot.py`。接触统计和弹簧保留冻结 v1.3 的实现，圆角几何来自 `bag_sim/mj_loose.py`。本次没有修改材料本构或进行物理重新标定。

YAM 原始模型来自 MuJoCo Menagerie 的 `i2rt_yam`，提交 `f054586a8e90465d49ee5be15335c4a0c7f57caf`，MIT 许可（I2RT）。仅复制这个完整包，共 34 个文件、约 9 MB；包内 MANIFEST、来源元数据、许可及上游文件保持原样，单位和局部原点不变。原共享包保留在原处。

归并检查见 [任务测试报告](../reports/yam-experience-test.md)。

## 完整动作录像

`TrialConfig.complete_motion=True` 用于录完接近、闭合、抬升、搬运、下降和松爪。闭合后发生的抓取失败或脱落会记录在结果的 `grasp_failure` 中，机械臂继续完成动作，最终 `outcome` 保留首次失败判定。失败后的下降按原目标高度完成，防止留在桌面的物体支撑读数提前结束空手下降。`motion_completed` 表示动作是否完成。数值、几何、挤压力及计算预算触发的停止仍正常生效。该选项默认False，2026-11-16用于用户要求的三档袋参数顶面完整流程录像。

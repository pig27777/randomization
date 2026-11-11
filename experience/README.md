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

## 来源

公共流程迁自 randomization 的 `test/flex3d/`，机械臂迁自 `policy/yam_probe_robot.py`。接触统计和弹簧保留冻结 v1.3 的实现，圆角几何来自 `bag_sim/mj_loose.py`。本次没有修改材料本构或进行物理重新标定。

YAM 原始模型来自 MuJoCo Menagerie 的 `i2rt_yam`，提交 `f054586a8e90465d49ee5be15335c4a0c7f57caf`，MIT 许可（I2RT）。仅复制这个完整包，共 34 个文件、约 9 MB；包内 MANIFEST、来源元数据、许可及上游文件保持原样，单位和局部原点不变。原共享包保留在原处。

归并检查见 [任务测试报告](../reports/yam-experience-test.md)。

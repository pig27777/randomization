# 统一机械臂抓取试验平台

公共平台源码已集中到 [`../../experience/`](../../experience/README.md)。本目录保留模型适配器、历史报告与结果，原启动脚本转发到 experience；动作、阈值和默认模型保持原样。后续平台开发请修改 experience，不再在这里维护第二份实现。

**后续使用约定（2026-11-09，用户指定）**：以后新增或对比内容物建模方案，统一在此平台上测试机械臂抓取行为和速度。模型只提供适配器，复用同一机械臂、指尖、动作、角度矩阵、事件判定、轨迹记录与测速入口，不再各建一套抓取流程。`flex3d`是首个试验留下的目录名，公共平台也能接入非四面体模型。

当前平台入口和模型替换接口已经运行验证；当前3D flex候选未通过可靠抓取验证。全角度机械臂可达性、当前模型抓取成功、κ覆盖和GPU训练接入均不因平台存在而自动通过。模型专属本构测量可独立进行，端到端抓取对比仍回到本平台。

本轮计划、真实结果和限制统一见 [REPORT.md](REPORT.md)。用户已取消平压，当前重点为完整YAM机械臂从不同角度抓取。代码不修改袋子线、内容物线或策略线。

公共平台固定机械臂、20×40 mm/2 mm圆角指尖、姿态、v2.2动作、结果判定与记录；`native_flex.py`仅负责3D flex本体、表面顶点捏取及翻转检测。`rigid_control.py`提供第二种模型的入口控制，验证公共流程不依赖四面体；没有将它作为硬物顶面抓的通过证明。

`segmented.py`只读接入冻结v1.3分段内容物，保留发布版bushing/阻尼/限位，使用其0.4ms步长；首轮14次原任务结果为4 success、5 grasp_fail、5 setup_fail，成功样本的机械臂跟踪峰值约42–43mm，不能称整套平台物理验收通过。完整结果见报告末节。

分段模型可通过`--model-options '{"size":[0.32,0.12,0.05],"mass":0.29}'`指定实际长宽高（米）和质量（kg），沿用本类别映射表；不设置时保持原锚点。冻结模板仅支持长方体包络，此处形状变化指长宽高比例，换几何后的响应/稳定性需另验。

## 运行

在仓库根目录，使用已有`.venv`，不安装包。每次输出目录必须不存在。

```sh
.venv/bin/python -B test/flex3d/angles.py --yaw 45 --output test/flex3d/results/NEW_RUN/one
.venv/bin/python -B test/flex3d/angles.py --backend segmented --dt .0004 --yaw 45 --output test/runs/YYYY-MM-DD/contents/NEW_RUN/one
.venv/bin/python -B test/flex3d/angles.py --model-options '{"young":20000,"poisson":0.3}' --output test/flex3d/results/NEW_RUN/material
.venv/bin/python -B test/flex3d/run_suite.py behavior --workers 2 --output test/flex3d/results/NEW_RUN/matrix
flock /tmp/claude-gpu.lock .venv/bin/python -B test/flex3d/run_suite.py speed --locked --workers 1 --trials 1 --output test/flex3d/results/NEW_RUN/speed1
flock /tmp/claude-gpu.lock .venv/bin/python -B test/flex3d/run_suite.py speed --locked --workers 28 --trials 28 --output test/flex3d/results/NEW_RUN/speed28
.venv/bin/python -B test/flex3d/render_video.py test/flex3d/results/NEW_RUN/one test/flex3d/results/NEW_RUN/one/grasp.mp4
.venv/bin/python -B test/flex3d/render_video.py test/runs/YYYY-MM-DD/contents/NEW_RUN/one test/runs/YYYY-MM-DD/contents/NEW_RUN/detail.mp4 --view detail --fps 12
.venv/bin/python -B test/flex3d/angles.py --yaw 45 --model-options '{"shape":"pillow","surface_refinement":2}' --output test/flex3d/results/NEW_RUN/pillow
.venv/bin/python -B test/flex3d/render_video.py test/flex3d/results/NEW_RUN/pillow test/flex3d/results/NEW_RUN/pillow/closeup.mp4 --tetra --focus
.venv/bin/python -B test/flex3d/export_tetra.py test/flex3d/results/NEW_RUN/pillow test/flex3d/results/NEW_RUN/pillow/playback.html
```

行为矩阵为两类×6姿态+两类纯接触对照，共14次。角度是0/45/90/135°水平朝向，另加±30°倾斜。`--smoke-seconds .02`只检查入口，不产生抓取结论。当前安装高度270 mm、抓取位置x=-250 mm、搬运终点x=250 mm；不可达姿态记`setup_fail`，不混入内容物物理失败。

`run_suite.py behavior --cases PATH`可运行JSON配置列表（各项为共享/模型选项，默认矩阵不变）；大小/比例展示的六项列表见`test/runs/2026-11-09/contents/segmented-size-shape-r1/cases.json`。保存所有试次后，用`render_comparison.py RUN_DIRECTORY`按该轮`manifest.json`的`variants`和`comparisons`导出单段及并排视频。近景使用相同距离/角度；短视频尾部保持末帧并标记STOPPED，不补造轨迹。新参数只影响回放排版与帧率，不改变动作或物理积分。

速度输出同时给出evaluate完成数/秒、成功抓取数/秒、仿真时间/墙钟时间与每类结果数。早停失败和完整成功的成本不同，比较模型时必须使用同协议、同机器、同日志配置并列出组成；尚未接入的分段模型不能拿历史速度作正式同机对照。行为运行耗时不作为吞吐数据。28进程采用spawn，限制进程内数值库线程的环境变量应由测速命令统一设置。

视频从保存状态回放，左右分别为机械臂全景和内容物近景；仅在采样状态之间插值显示，不重新生成物理轨迹。渲染强制Mesa llvmpipe并检查实际renderer，不占GPU。标题包含配置和本次任务事件结果；请同时看报告中的穿透、翻转、连接与跟踪误差，不能只凭视频判断模型可靠。

`--tetra --focus`隐藏机械臂，显示完整内容物与局部四面体；只回放末0.8 s，慢放10倍并停留末帧3 s。红色边和顶点表示体积比非正的单元。`shape=pillow`实际改变三维参考网格：240×160×50 mm外包络、中央50 mm厚、周边缝口6 mm、圆角轮廓、体积内填满四面体；不是显示时变形。质量保持0.29 kg，但这是新的未标定形状，不能套用长方体κ测量结论。

## 接入另一种模型

新增一个可导入的适配器模块，通过`--backend MODULE`选择，不改公共阶段或结果阈值。以`rigid_control.py`为最小例子，模块提供：

当前公共执行环境是MuJoCo完整YAM；直接接入的模型需提供MuJoCo的`m/d`状态。其他物理引擎需要先实现耦合适配，GPU批量也需要单独的后端验证，不是改一个模型名称即可支持。

| 接口 | 职责 |
|---|---|
| `Config(TrialConfig)` | 继承共享任务选项；新增本模型参数，经`--model-options`传入，禁止覆盖共享选项 |
| `scene(cfg)` | 生成本体XML及模型自己的状态容器；调用`attach_robot`接入同一个YAM；`sim`提供`m/d/step()/com()`，机器人提供`size/mass` |
| `measure(robot)` | 返回指尖接触、支撑、顶面接触、穿透。常规MuJoCo目标可直接调用`measure_contacts`并给出geom/flex编号 |
| `attachment(robot,cfg)` | 可选捏取；提供`active/load/events/try_bind/check_load/release`，不用捏取时返回空连接实现 |
| `check_state(sim)` | 返回`(失效原因或None, 记录指标字典)`；如四面体翻转、段模型越限等，由各模型负责，公共平台不猜测 |
| `summarize(trace)` / `dimensions(sim)` | 模型测量摘要与自由度/网格规模 |
| 可选 `refresh(sim)` | GPU等后端自行刷新采样状态；未提供时仍用原CPU forward |
| 可选 `record_state(sim)` / `STATE_SPEC` | 保存含塑性等历史的完整integration state，额外写入NPZ，不改变原qpos记录 |
| 可选 `source_files()` | 补充依赖源码快照与指纹 |
| 可选 `built.compile_model(xml)` | `MuJoCoSim`调用模型提供的编译器；用于mjSpec专有材料等，默认仍直接编译XML |
| 可选 `save_model(sim,output)` | 保存XML不能表达的模型设置；SNH适配器写`model.mjb`，公共回放优先读该完整模型 |

直接调用`run(..., on_sample=callback)`可观察已记录样本并渲染UI画面，不参与控制或判定。适配器配置可提供`simulation_seconds`限制时长；没有此字段仍为原30秒。连续体UI显式使用10秒，失败事件提前停止。后端逐步有效性检查若抛出且`check_state`给出模型失效原因，平台保存最后安全导出的几何、完整后端失败状态与`sim_fail`结果；未知运行异常仍交原异常处理。

原生flex仅用MuJoCo原生StVK弹性和discrete积分。捏取连接到真实掌部，反作用回传机械臂；冻结v1.3的聚拢弹簧是既定抓取效果规则的一部分，不是补充材料本构。I-37只计掌部向下的拉离反力和水平分量。`pinch_coeff=0`关闭这一效果，用于区分原生接触与捏取规则。

每次保存配置、模型XML、平台/模型源码快照与SHA256、资产指纹、事件、接触轨迹、关节状态和结果。已有目录拒绝覆盖；`setup_fail`/`compute_limit`/`execution_error`单列。一次试验只表明该配置的结果，尚无重复性或实物标定结论。

## 后续实验的固定做法

1. 在本主题报告中先写要回答的问题、模型/形状/参数、角度、重复数、已有验收条件与停止规则。使用新的run-id，保留失败。
2. 新适配器先跑短入口，核对质量、坐标、几何和接触目标；再运行同一角度矩阵。初始化没有找到姿态解的条件单列，不能冒充内容物失败，也不能在看过结果后悄悄删掉。
3. 模型之间比较时保持机械臂安装、指尖、初始位置、动作、目标力、步长策略与结果判定一致。共有设置确需改变时，先说明原因，在计划和新结果中标清；改变阈值遵循项目规则。
4. 行为、模型有效性与速度分开汇报。视频能提起不等于接触/本构可靠；早停失败的evaluate数/秒不等于成功抓取数/秒。正式CPU测速与所有GPU运行持共享锁，优先让内容物GPU G3–G5；同机对照并记录后台负载。
5. 结果填回同一份主题报告，`reports/`写任务检查报告，`log/progress.md`只写3–5行。完成一个阶段后提交推送，别的会话按保存的版本和配置复用。

# 连续体内容物与调参抓取UI检查

2026-11-09，ubun，RTX5090；隔离MuJoCo/MJWarp4.0.0、Warp1.19.0。

固定B的三维StVK/J₂内容物、真实体积形状参数化、GPU Newton推进、完整塑性状态记录和统一YAM适配已实现。UI可调参后运行、显示真实样本并生成录像，单次上限10秒仿真。**后续已取得一例窄小实体完整原生接触success；可靠抓取与随机κ训练仍未验收。**

| 范围 | 方法 / 结果 |
| --- | --- |
| 构建与形状 | `validate_continuum.py --mode construction`；三材料×四形状12例、24随机形状、8非法参数通过，同拓扑，质量按实际体积变化 |
| GPU材料与状态 | 初版`verification/`全部通过；当前精化版三材料通过，但塑性/状态复测分别被1.00081e-5/1.00122e-5残差拦截，阈值未放宽 |
| 原统一YAM | 四形状短入口通过；完整三组为硬盒提起失夹、两软形状塑性非有限，无成功抓取，不冒充通过 |
| UI API | `check_grasp_ui.py --url http://127.0.0.1:8767 --output NEW`；四真实预览、7非法输入、同源校验通过 |
| UI JS | `node test/content_model/check_grasp_ui.cjs test/runs/2026-11-09/contents/continuum-gpu-r1/ui-api-fixed`；DOM/事件契约通过，非浏览器视觉检查 |
| HTTP→GPU→录像 | 用户从8766提交的`202500-7ed5369a`；真实采样更新、失夹结果与MP4闭环完成，GPU Newton，无效果连接 |
| 公共平台回归 | `.venv/bin/python -B -m unittest discover -s test/flex3d -p 'test_*.py'`，6项通过 |
| 纯GPU速度 | 初始硬盒1world中位412.93步/s，2.42172ms/步；24κ×1/32world完成：19/24、13/24组三次有效；中位588.02 step/s、9162.14 world-step/s |

正式GPU命令均持`flock /tmp/claude-gpu.lock`。纯GPU矩阵命令：

```sh
flock /tmp/claude-gpu.lock .venv-continuum/bin/python -B test/content_model/benchmark_continuum.py \
  --matrix --worlds 1 32 --steps 2000 --repeats 3 \
  --output test/runs/2026-11-09/contents/continuum-gpu-r1/resident-matrix
```

时间推进与计时期间物理状态常驻CUDA；不同κ分模型组上传，尚非异κGPU在线随机化或纯GPU闭环机械臂控制。编译、IK、UI控制、诊断读回和软件渲染不称为GPU计算。无CG/CPU步进回退，无压密，自碰撞关闭；未标定材料与质量分配、袋膜耦合及大规模训练仍待验证。CUA未提供可用浏览器，未声称浏览器布局验收。

初次入口导入、四面体绕向、Warp变量类型/外力shape、UI预览重复关键字和classic GL上下文兼容失败均保留，详见[主题报告](../test/content_model/CONTINUUM_REPORT.md)。[全部原始数据](../test/runs/2026-11-09/contents/continuum-gpu-r1/)；[形状预览](../test/runs/2026-11-09/contents/continuum-gpu-r1/shapes.png)。

纯GPU矩阵源码未变化，48组中32组有效、16组数值无效；无效组全部检测到非有限状态，其中15组伴随溢出/残差失败。有效组才计速度；32world相对单world的13个配对有效κ，中位合计吞吐增益22.106倍。全部测试已结束，细表和失败快照见主题报告与原始数据。

收尾补测：当前易弯枕形1s上限运行未重现先前失败，实际2000步、完整状态有限（`failed-state-recording/`）；这不是稳定性全域通过。易压枕形在695步非有限，公共平台正确保存`sim_fail`、18个安全几何样本及含非有限PLASTIC的完整失败状态（`failed-state-compressible/`），没有导出失效接触。矩阵结束后仅给适配器增加GPU尚未建立时保留原始异常的保护；GPU计算源码未改。

## 分段GPU速度对照补测（2026-11-09）

新增`benchmark_segmented.py`，复用冻结v1.3的同YAM场景和原GPU材料/压密步进器。三预设、1/32world、相同240×160×50mm与0.29kg、固定初始控制，CUDA事件段内无主机读回；每次1s仿真、3次重复。分段0.4ms需2500步，连续体0.5ms需2000步。

严格首轮6组均因线搜索上限警告拒绝，未冒充容量溢出或通过。另跑`--diagnostic-linesearch`保留valid=false及raw=1024，只测带警告的完整1s诊断成本：硬物分段/连续体单world为0.900/4.998s，易弯8.651/3.335s，易压13.005/3.052s；32world分别0.977/7.976、10.384/4.938、15.478/连续体无效。所有分段诊断状态有限且本体/容量阈值未失败，仍因警告不能认作有效吞吐。源版本/积分器/响应未等价标定，不能将上述成本比当表示法普遍加速倍数。

[详细对照表与限制](../test/content_model/CONTINUUM_REPORT.md#对照结果) · [原始数据和源码快照](../test/runs/2026-11-09/contents/segmented-gpu-speed-compare-r1/)。GPU调用均持锁；视频agent仅做CPU回放，没有重算动力学。

用户另指定agent录制两类模型视频，已完成4单片+1原生接触并排，全部1280×480/20fps、ffprobe/全解码与关键帧检查通过；父会话验证观看页与主视频HTTP可用。播放入口`http://127.0.0.1:8768/`；[视频检查报告](contents-model-videos-test.md)。标签明确辅助捏取/源CPU或GPU/材料不匹配/提前失夹，不将视频回放当速度或物理验收。

## 成功片及翻转失败片（2026-11-09，用户后续要求）

沿用原统一平台、同B、GPU Newton、pinch_coeff=0，仅调整X。首例120×50×50mm/E10kPa在3.665326s单元#43翻转（J=-.000772392），已保留失败片与标红慢放。第二例120×45×50mm/40.5g/E100kPa/ν.1/yield.2，在15.02556s完成原生接触完整success。runtime无失效、overflow或warning，Jmin.75958，残差峰9.99988e-6，穿透峰.88285mm；提起保持/搬运采样桌面反力0，最终质心x=.25366m。跟踪峰40.60mm单列。本次plastic=0，不将该成功例说成软塑性大物体成功或成功率验证。

数据：`test/runs/2026-11-09/contents/continuum-grasp-success-r1/`。新片分别为`continuum-native-success.mp4`、`continuum-failed-narrow.mp4`、`continuum-tet-inversion.mp4`，放于原视频目录和`docs/assets/`。取得成功后停止剩余参数探索，全部失败保留；按用户要求未额外检查/重跑，录像进程正常结束。

## 大软枕头与四面体显示（2026-11-09，ubun）

用户明确恢复240×160×50mm/实际0.29kg大软枕头，并要求不渲染外表面。本轮统一E5kPa、189节点/576四面体、GPU Newton、原YAM动作、pinch0，每配置一次，共三次；命令、源码及完整状态在`test/runs/2026-11-09/contents/continuum-pillow-grasp-r1/`。

| 配置 | 真实结果 |
| --- | --- |
| StVK/ν=.3 | 4.371451s合拢时单元247翻转，J=-.0002525；失败保留 |
| StVK/ν=.45 | 3.312351s接近阶段6个塑性分量非有限；位置速度仍有限 |
| 实验性SNH/ν=.3 | 7.938893s提起失夹；无数值/几何失效、overflow或warning，Jmin.43276、残差峰9.999868e-6、穿透峰.35394mm；夹持峰1.568N、跟踪峰25.99mm，未提起 |

适配器增加显式网格与原生SNH选择；SNH通过mjSpec编译、完整MJB归档，回放优先读取；公共动作与门槛不变。四面体视图两相机均隐藏FLEXFACE/FLEXSKIN，显示全体积边线，物理体积/接触未删除。已按计划停止扫参；当前没有大软枕头成功抓取。按用户要求未增加独立回归或视频/页面QA；仅执行实际任务与内置有效性门槛，不能沿用旧StVK验收宣称SNH已验收。[主题报告](../test/content_model/CONTINUUM_REPORT.md#大软枕头抓取本轮计划2026-11-09ubun运行前)。

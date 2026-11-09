# 连续体内容物与调参抓取UI检查

2026-11-09，ubun，RTX5090；隔离MuJoCo/MJWarp4.0.0、Warp1.19.0。

固定B的三维StVK/J₂内容物、真实体积形状参数化、GPU Newton推进、完整塑性状态记录和统一YAM适配已实现。UI可调参后运行、显示真实样本并生成录像，单次上限10秒仿真。**当前完整抓取未验收通过，不能据此进入训练。**

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

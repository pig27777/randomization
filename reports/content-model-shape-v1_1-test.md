# 内容物 v1.1 形状检查

最近检查：2026-11-05 21:08（UTC+8）。**V1、V6 通过，V2 未通过；停止本阶段，未打 v1.1 标签，未进入 v1.2。** v1 冻结包保持不变。

## 范围与方法

- 超二次包络，格子裁剪凸网格，低于格体积 10% 的段及连接移除；实际四面体体积积分、完整惯量、正密度质量偏置实现 κ 的 COM。
- `shape_p/q` 同步进入约束采样与 RJ；有效截面积/二阶矩修正以 p=q=12 作为既有换算表的参考，未更改本构、0.4 ms 或 Newton 设置。
- V1 的 18 种构建与 MuJoCo 自身凸网格积分独立对照；V2 用解包 v1 源码做三锚点 R1y/z、R2、R3 对照；V6 从真实顶视深度图计算两个观测。
- V3/V5 未运行、V4 未完成，因 V2 失败按用户要求停止；不把未运行项写成通过。验收线保持项目计划。

## 探索检查与修正记录

- 18 种 V1 构建、非零 COM、R3/RJ 配对、原采样速度检查的探索结果通过，单个软物构建/编译约 0.3–0.4 s，仅确认入口，非正式吞吐数据。
- 最初整套探索检查 27 项有 4 个子项失败：圆角单元 COM 不再等于格中心，原卸载检查将“关于格原点的纯轴向载荷”误当作“关于 COM 的零力矩”，外功也只计平动。修正为同一原点的零力矩检查，并补计实际力矩做功；**没有删除断言，原 1e-10、1% 能量、5% 恢复等阈值均保留**。随后 10 项本构检查通过，4.972 s。[初始失败](../contents/segment/results/shape_v1_1/logs/exploration_unit.log)、[修正后检查](../contents/segment/results/shape_v1_1/logs/exploration_bushing_corrected.log)。
- 深度图有 10 个孤立边缘光栅异常像素；使用统一 3×3 中值处理，完全依据图像，不按真实尺寸裁剪。初检盒/圆柱/椭球的填充比约 0.990/0.990/0.784，边缘高度比约 0.973/0.794/0.677；待正式 V6 确认。
- 原探索使用绝对截面修正，发现会重复作用于既有盒参考；正式运行前改为 p=q=12 归一化的连续形状修正。原正式命令尚在等待共享锁、未产生结果时取消并重新排队，没有覆盖已跑正式数据。

## 正式命令

工作目录 `contents/segment/`，每次运行使用新目录；完整正式链在同一 `flock /tmp/claude-gpu.lock /bin/bash -c ...` 内串行执行：

```bash
segment_python=/home/ubuntu/Dev/Envs/mjwarp/bin/python
"$segment_python" -B validate_shape.py geometry --run-id geometry_formal
MUJOCO_GL=egl "$segment_python" -B validate_shape.py observation --run-id observation_formal
"$segment_python" -B validate_shape.py responses --run-id v1_reference --reference-root results/shape_v1_1/reference_v1 --workers 6
"$segment_python" -B validate_shape.py responses --run-id box_regression --workers 6
```

## 限制

当前工作区为未冻结 v1.1；正式 v1 基线使用 [v1 源码包](../contents/segment/results/freeze_v1_newton/release_v1/v1.tar.gz)。模型的采样前自重筛查仍是名义满格估计，实际裁剪后的质量/惯量由网格积分；动态可靠性必须看 V5，不从几何或代数检查推断。任务阈值和其他线代码未改，未安装包或修改共享库。主报告：[v1.1 验收](../test/content_model/REPORT.md)。

## 正式结果

- **完整单元检查 28/28 通过，44.425 s**，保留初始失败和修正记录。[日志](../contents/segment/results/shape_v1_1/logs/geometry_formal.log)。
- **V1 通过**：18 组总质量最大误差 1.11e-16 kg、归一化 COM 最大误差 2.27e-8、惯量相对误差 2.08e-8；删段数为 0 或 8，连接一致性检查通过。[构建数据](../contents/segment/results/shape_v1_1/geometry_formal/summary.json)。
- **V2 未通过**：三锚点四项响应、每配置一次，与解包 v1 比较；24 次均无确认仿真失败。易压物 R2 0.358499→0.280066（−21.88%）、R3 滑移 0.756037→1.491081 mm（+97.22%）；硬物 R2/R3 也明显变化。全部 12 个记录值仅 7 个差异 ≤5%，硬物 R1 为下限不当成精确刚度。[基线](../contents/segment/results/shape_v1_1/v1_reference/summary.json)、[候选](../contents/segment/results/shape_v1_1/box_regression/summary.json)。
- **V6 通过**：盒/圆柱/椭球的（填充比，边缘高度比）分别为 (0.990079,0.972979)、(0.990079,0.793982)、(0.783784,0.677721)，组合可区分。[原深度与特征](../contents/segment/results/shape_v1_1/observation_formal/summary.json)。
- [汇总及指纹](../contents/segment/results/shape_v1_1/acceptance.json)核对几何、观测、候选响应三批与当前源码一致；独立 v1 基线的全部源码指纹与冻结标签一致，原源码包 SHA256 未变。

V2 的候选根因为接触外形/整体位姿、实际质量与惯量、截面换算的组合效应，尚未分离证明。建议先作原因分离再标定；硬物近零量的相对回归解释交总体判断会话和用户，不自行更改 5% 标准。已记 I-31。正式链已结束、本线锁已释放；不继续 V3–V5 或顶面抓。

## 交互 UI 查看（2026-11-05T21:21:53+08:00）

按用户要求启动 `DISPLAY=:1 view_m2.py compressible center --rate 0.5`；仅在进程内隐藏 site 标记，未修改源码或物理参数。用 `xwininfo -root -tree` 确认 `MuJoCo : segment_content` 窗口已创建，启动日志无异常。

这是当前未冻结 v1.1 的中心横夹/提起演示，不作为正式验收或速度数据；顶面抓尚未实现。[UI 日志](../contents/segment/results/ui/shape_candidate_compressible_center_20261105.log)。

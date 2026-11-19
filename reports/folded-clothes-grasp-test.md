# 近满面薄层包裹抓取视频

2026-11-19，ubun，MuJoCo4.0.1。状态：视频交付检查通过；抓取物理验收未通过。

- 目标：袋350×250 mm，内容物330×230×6 mm、200 g，既有三段可弯曲体近似叠衣外形，保留142节点/280面封闭袋膜。
- 修复范围：box_face_clearance初始化，处理共享封边重复投影、三角面跨盒角及初始化地板；未改变参考拓扑、材料、物理步接触或加入夹持绑定。
- 首轮8 mm配置因60面穿盒被拒绝；修正后8 mm初始最大边应变7.486%超过原5%标准，未运行。6 mm配置最大初始应变4.617%，0面相交、8盒角包络内，通过。失败原始记录在folded-clothes-flat-r1。
- `flock /tmp/claude-gpu.lock env PYTHONPATH=test/bag_model:experience:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-physical-parcel/bin/python -B -u -m unittest discover -s test/bag_model -p test_physical_parcel.py -v`：17/17通过，含新近满面薄层回归。
- 0.5秒静置准备通过：膜—内容物、膜—支撑的几何深度上界均0，无原生warning及非有限值。
- 共享YAM顶抓完整录制已完成。零超行程、无袋—爪/袋—内容物绑定，complete_motion=True仅在保留原失败判定后继续动作，不表示抓取成功。
- 本轮仅一次示例，不作成功率或真实织物标定结论。

[主题报告](../test/bag_model/PHYSICAL_PARCEL_REPORT.md) · [数据与配置](../test/runs/2026-11-19/bag/folded-clothes-6mm-r1/) · [回归日志](../test/runs/2026-11-19/bag/folded-clothes-6mm-r1/regression.log)。

## 最终检查（2026-11-19T21:12:18.206819+08:00）

- 最终17项回归通过（5.990 s），增强初始化限于共享封边邻面；200/142节点常规初态与原版本逐位相同，6 mm初态与录像版本逐位相同。详见[兼容性检查](../test/runs/2026-11-19/bag/folded-clothes-6mm-r1/compatibility.json)。
- 顶抓最大升高0.06663 mm，4.66 s失去接触；侧抓最大升高175.96785 mm、最长连续离支撑采样跨度2.20 s，最后COM落位失败，原生穿透1.22326 mm超原1 mm预算。两次均保留grasp_fail，整体物理验收未通过。
- 两次轨迹无warning、全有限，独立膜—物和膜—支撑几何深度均0，辅助材料承载连接为0。侧抓跟踪峰值166.43 mm，不能视作精确笛卡尔搬运。
- CPU Mesa渲染（llvmpipe）及FFmpeg完整解码通过。三段输出为1280×560/20fps，顶抓11.00 s、侧抓12.90 s、连续23.90 s；抽帧核对闭合、空手抬升、侧抓悬挂与松爪。详见[视频验证](../test/runs/2026-11-19/bag/folded-clothes-6mm-r1/video-verification.json)。
- 历史整袋提拉、动量、GPU验证未重跑；三段体仅近似叠衣外形，未建织物层间摩擦或完成真实衣物标定。用户已要求CPU/释放GPU，本会话无GPU计算进程、不持GPU锁。

[连续视频](../test/runs/2026-11-19/bag/folded-clothes-6mm-r1/folded-clothes-grasp.mp4) · [顶面视频](../test/runs/2026-11-19/bag/folded-clothes-6mm-r1/top-grasp.mp4) · [侧缘视频](../test/runs/2026-11-19/bag/folded-clothes-6mm-r1/side-grasp.mp4)。

最终维护源码/文档的暂存差异检查通过；全量`git diff --cached --check`返回2，仅有两份实际运行model.xml中的6处行尾空格。为保留模型快照原字节，归档文件未重写；没有测试断言被删除或放宽。

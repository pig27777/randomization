# 分段内容物尺寸/形状比例视频：检查报告

最近检查：2026-11-09，机器 `ubun`。

范围：统一平台分段适配器的实际size/mass参数、可指定配置列表的公共行为入口、CPU保存轨迹回放与并排视频。冻结本构和公共动作/阈值不改；“形状”仅指长方体长宽高比例，未实现圆柱/圆枕。

- 集成检查：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /tmp/claude-gpu.lock .venv/bin/python -B -m unittest discover -s test/flex3d -p test_segmented.py -v`，4项通过，含实际编译几何包络、总质量、72条连接和非法尺寸拒绝。[日志](../test/runs/2026-11-09/contents/segmented-size-shape-r1/adapter-checks.log)。
- 六配置均完成20ms入口，无穿透/非指尖接触、模型失效或警告；短入口不作抓取验收。`--view detail --fps 12`短回放为640×480、llvmpipe，ffprobe确认单帧入口成功。
- 六配置完整任务：5次success、扁平包1次grasp_fail，均无模型失效/警告。大包穿透峰1.065mm，跟踪峰整体30.45–48.18mm；不以任务success宣称物理验收通过。[结果](../test/flex3d/REPORT.md#六配置结果)。尺寸组同密度、质量随体积变化；比例组同体积、同质量；均保留原易弯响应κ和分段数，未验证新几何下的响应命中。
- 六项归档XML包络/质量及运行源码核对通过，四个Python源AST通过。早期质量审计用了比冻结XML 8位有效数字更严的1e-12请求值比较而失败；现核对归档XML实际质量和编译结果，保留1.6e-9kg的最大序列化误差，没有改写历史轨迹或放宽物理标准。
- 视频：`render_comparison.py`从已保存轨迹导出六段640×480、两段1920×480，12fps/1倍速；全部llvmpipe，统一近景距离/角度。六段原片帧数/渲染记录一致，并排时长/帧数与组内最长原片一致，八个文件完整解码通过。人工查看尺寸首帧和形状13.5s，字幕与STOPPED保持末帧标记有效。[完整检查](../test/runs/2026-11-09/contents/segmented-size-shape-r1/videos/video-checks.json)。
- 仅单次展示；形状范围、原生摩擦抓取、接触可靠性和机械臂跟踪未通过新验收，不能把这批视频当作训练放行或真实包裹表现。

[本轮配置和原始结果](../test/runs/2026-11-09/contents/segmented-size-shape-r1/)

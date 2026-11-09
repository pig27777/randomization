# 整体响应κ交互预览：检查报告

最近检查：2026-11-09，机器 `ubun`。

结论：本地κ调参界面已连接冻结v1.3的真实CPU步进；参数提交、暂停和重建、输入边界及绘图契约检查通过。当前没有可用浏览器，未宣称浏览器布局或真实拖动验收。

- 范围：`test/content_model/parameter_ui.py`、同名HTML、`check_parameter_ui.cjs`、`tests/test_parameter_ui.py`；不修改冻结本构。横向/厚度刚度和下垂范围直接读取冻结映射，阻尼0.05–1.5、边界外载0–15N。
- CPU物理检查：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /tmp/claude-gpu.lock .venv/bin/python -B -m unittest discover -s test/content_model/tests -p test_parameter_ui.py -v`，3项通过。覆盖两类×三种预览边界各0.1s；同一5N载荷下，ky=0.12与1.0N/mm各计算1.2s，软设置的实际压缩率比硬设置多于2个百分点；非有限值/越界输入被拒绝。
- HTTP：页面/元数据200，5项非法请求400，真实配置代次生效并进入步进，暂停生效；本机HTTP检查须绕过环境HTTP代理。详见[HTTP记录](../test/runs/2026-11-09/contents/segmented-v13-parameter-ui-r1/http-check.json)。
- JS：`node test/content_model/check_parameter_ui.cjs test/runs/2026-11-09/contents/segmented-v13-parameter-ui-r1`通过900/360px契约，覆盖滑块、两端精确边界、锚点/模式切换、暂停、旋转/缩放/复位及有限绘图坐标。对数滑块端点已精确回写上下界，避免浮点舍入造成表外请求。[检查记录](../test/runs/2026-11-09/contents/segmented-v13-parameter-ui-r1/ui-contract-check.json)。
- 限制：CUA返回apps/browsers均为空；上述宽度检查是代码契约而非浏览器截图。界面展示的是指定载荷/固定端下材料响应，非统一YAM抓取验收，也不以本预览测量替代参考响应协议。映射范围内的任意参数组合未全部验收，实际越限或求解警告直接显示。

[运行说明](../test/content_model/TOOLS.md) · [物理检查日志](../test/runs/2026-11-09/contents/segmented-v13-parameter-ui-r1/physics-checks.log)

# 内容物 UI 启动检查

最近检查：2026-11-04（Asia/Taipei）。

结论：当前分段内容物的 MuJoCo 原生界面已打开，默认展示易压物（4×4×3，共 48 段）。未修改内容物源码或实验配置。

## 范围与方法

- 只读导入 `contents/segment/model.py`、`mj_adapter.py` 和现有换算表，构建硬物、易弯物、易压物三个锚点的临时 MJCF。
- 用现有 `/home/ubuntu/Dev/Envs/mjwarp/bin/python`、`DISPLAY=:1` 和 `mujoco.viewer.launch_from_path` 启动原生查看器。
- 用 `xwininfo -root -tree` 确认窗口；用 MuJoCo Renderer 生成模型视口图片并目视检查。

## 结果

- 三个锚点的 MJCF 均成功编译；易压物为 288 个自由度、824 根 tendon。
- 已发现可见窗口 `MuJoCo : Content - compressible`，客户区 1706×960。
- [模型预览](/tmp/random-content-ui-6svvpr9z/compressible.png) 为 1080×720，构建后的 qpos 全部有限。
- 三类临时模型位于 `/tmp/random-content-ui-6svvpr9z/`；查看器仍需使用，予以保留。

重开当前界面：

```bash
DISPLAY=:1 /home/ubuntu/Dev/Envs/mjwarp/bin/python -B -c "import mujoco.viewer; mujoco.viewer.launch_from_path('/tmp/random-content-ui-6svvpr9z/compressible.xml')"
```

## 限制

- 这是原生模型查看界面，尚无针对 κ 的独立参数面板；视口图片不包含查看器侧栏。
- 未运行正式物理实验、稳定性或响应验收，本检查不代表第 2 轮通过。
- 该环境没有 Pillow；预览直接由 MuJoCo 渲染，使用 Python 标准库编码为 PNG，未安装依赖。

## 同日追加：夹紧—松开动态演示

- 初次界面仅展示无夹爪驱动的静置模型，因此静置后不再运动。已改用临时预览脚本 `/tmp/random-content-ui-6svvpr9z/animate.py`，只读复用内容物的 `Sim`、`Runner`、`Closer`，不修改实验代码。
- 启动命令：`DISPLAY=:1 /home/ubuntu/Dev/Envs/mjwarp/bin/python -B -u /tmp/random-content-ui-6svvpr9z/animate.py`。
- 目视窗口入口确认：`MuJoCo : segment_content`；运行日志已连续出现两轮 `squeeze → hold → release`，指垫单侧行程约 37 mm，两侧接触力约 9 N；这只确认预览动作在执行，不作为正式物理验收或吞吐数据。
- 空格暂停/继续，R 重置；异常时预览暂停并输出原因，不静默忽略。查看器保持运行。

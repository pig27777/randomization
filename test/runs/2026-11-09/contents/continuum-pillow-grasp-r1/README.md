# 大软枕头原生接触抓取

机器：ubun；240×160×50mm真实枕形、实际质量0.29kg、GPU Newton、无辅助捏取。

计划与结果维护在 `test/content_model/CONTINUUM_REPORT.md` 的“大软枕头抓取”节；每个候选独立目录保留原始结果与失败。`soft-5k-options.json` 是首个候选的完整模型参数；调用现有 `test/flex3d/angles.py`，不另建执行器。

本轮三例已结束：StVK首例4.371451s翻转；ν=.45例3.312351s塑性非有限；SNH例7.938893s提起失夹、无数值/几何失效。三者均真实大软枕头，不缩小/加硬。本轮停止扫参。

SNH完整模型为 `soft-5k-snh/model.mjb`，MJCF没有elastic3d属性，单独读取该例model.xml不能恢复SNH本构。复现见各*-command.json和*-options.json，依赖源码快照随试次保存。

观看目录 `../contents-model-videos-r1/` 的新片为 `continuum-pillow-soft5k-failed.mp4`（有外表面）、`continuum-pillow-soft5k-tetra.mp4`（首例全四面体）、`continuum-pillow-snh-tetra.mp4`（SNH全四面体）。三者均保留真实终止结果。

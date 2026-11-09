# 内容物建模版本

本目录维护内容物各版建模实现、参数与设计。现状、限制和下一步由 [HANDOFF.md](HANDOFF.md) 维护；测试、验收、分析、测速与查看器集中在 [test/content_model/](../test/content_model/TOOLS.md)，按日期查记录见 [test/runs/](../test/runs/README.md)。

## 版本索引

| 版本或方案 | 建模入口 | 版本依据与边界 |
| --- | --- | --- |
| 分段模型 v1 | `segment/model.py`、`segment/bushing.py`、`segment/mj_adapter.py` | [冻结标签](segment/results/freeze_v1_newton/release_v1/v1.json)；历史包保持原样 |
| 顶面捏取 v1.2 | 分段本体与 `segment/pinch.py` 的历史 weld 规则 | [冻结标签](segment/results/pinch_effect_v1_2/v22_review/release_v1_2/v1.2.json)；复现使用标签旁的归档，当前源码已演进 |
| 顶面捏取 v1.3 | `segment/pinch.py`，共享分段本体 | [冻结标签](segment/results/pinch_connect_v1_3/release_v1_3/v1.3.json)；connect 连接，当前冻结参考 |
| 捏取 v1.4 候选 | `segment/pinch_v14.py`、`segment/pinch_v14_reference.json` | 本机候选；参数依据见 [标定方法](pinch_calibration.md)，未冻结 |
| Warp / MJWarp 后端 | `segment/warp_*`、`segment/mjwarp_backend.py`、`segment/mjwarp_v13.py` | 区分 v1.2 与 v1.3 的规则；接入状态以交接和实验报告为准 |
| Newton 四面体 | `newton/unified_soft.py`、`newton/newton_soft.py` | 历史建模与代理评估依赖，保留生成器；独立检查在 `test/contact_transfer/` |
| 旧粒子键模型 | [legacy/particle/](legacy/particle/) | 已停止路线；代码与原结果一起归档 |
| 早期刚体核心模型 | [legacy/rigid/](legacy/rigid/) | 历史原型；代码与原结果一起归档 |

同一模型的测量、采样、参数换算和任务适配接口可以随模型维护；独立的 `validate_*`、`benchmark_*`、`analyze_*`、`diagnose_*`、`run_round*` 及单元测试统一在 `test/`。已有版本名和冻结包是版本依据，不复制整套当前源码来制造新版本。

## 数据与历史路径

`segment/results/`、`newton/results_*` 中已存在的换算表、冻结包、源码快照和原始证据保留原位，便于原记录按哈希复现。它们是历史数据，不是新测试的输出目录。新实验按 [测试管理约定](../test/runs/README.md) 指定日期、对象与唯一运行标识。

日常 `rg` 搜索通过根 `.ignore` 排除历史快照；`rg --no-ignore-dot` 可将它们包含进来。Git 的跟踪与历史内容不受该搜索设置影响。

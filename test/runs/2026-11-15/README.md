# 2026-11-15 运行索引

| 对象与版本 | 问题、机器与状态 | 报告与数据 |
| --- | --- | --- |
| 袋子与降阶内容物，既有双膜片网格，MuJoCo/MJWarp4.0.1 | ubun；纯接触支撑与外力提拉完成，捕获未通过；最大已测8192环境，9216 Graph创建OOM；用户要求停止 | [主题报告](../../bag_model/PHYSICAL_PARCEL_REPORT.md) · [任务检查](../../../reports/physical-parcel-test.md) · [全部原始运行](bag/) |

入口为 `test/bag_model/physical_parcel_checks.py`、`physical_parcel_benchmark.py` 与统一YAM平台的 `physical_parcel_yam` 适配器。完整命令、配置、重复次数及原始结果分别保存在主题报告和各 `bag/physical-parcel-*` 目录。

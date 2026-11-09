# 实验导航

这里存放实验协议、计划、逐轮结果及相关试验入口。同一主题在原报告中追加轮次；开发任务的检查记录另见 [reports/](../reports/README.md)。

## 按问题查找

| 主题 | 入口与报告 | 主要实现或数据位置 |
| --- | --- | --- |
| 机械臂抓取与模型对比 | [统一平台与接入接口](flex3d/README.md) · [实验报告](flex3d/REPORT.md) | `test/flex3d/`、`test/flex3d/results/` |
| 内容物响应定义与建模验收 | [R1–R4 测法](content_model/README.md) · [实验报告](content_model/REPORT.md) | `contents/segment/`、`contents/segment/results/` |
| 外袋及松装建模 | [实验报告](bag_model/REPORT.md) | `bag_sim/`、`bag_sim/outputs/`、`test/bag_model/` |
| 抓取策略与共享任务协议 | [任务协议](policy/README.md) · [策略报告](policy/REPORT.md) | `policy/`、`policy/results/` |
| 同外袋、不同内容物的快递堆抓取 | [实验报告](policy/HIDDEN_CONTENTS_REPORT.md) | `policy/parcel_heap.py`、`policy/results/parcel_heap_20261109/` |
| YAM 探测反馈 | [实验报告](policy/PROBE_REPORT.md) | `policy/probe_feedback.py`、`policy/results/yam_probe_feedback_20261108/` |
| 刚体与柔体接触基准 | [说明](contact_transfer/README.md) · [主报告](contact_transfer/REPORT.md) · [修复](contact_transfer/REPORT_repair.md) · [Newton 1.7 复测](contact_transfer/REPORT_newton170.md) | `test/contact_transfer/`、`contents/newton/` |
| 夹爪抓取接触验收 | [实验报告](gripper_grasp/REPORT.md) | `test/gripper_grasp/` |
| 袋子与内容物联合接触 | [基准说明](bagged_contact/README.md) · [实验报告](bagged_contact/REPORT.md) | `test/bagged_contact/` |
| 薄膜与自由颗粒测速 | [实验报告](flex_granular/REPORT.md) | `test/flex_granular/`、`test/flex_granular/results/` |
| 代理真实测试集 | [实验报告](proxy_eval/REPORT.md) | `proxy_eval/`、`proxy_eval/results/` |
| Isaac Lab 可行性 | [实验报告](isaaclab/REPORT.md) · [运行说明](../isaaclab/README.md) | `isaaclab/`、`isaaclab/results/` |

## 复现与后续实验

先读对应报告中的本轮计划、版本、机器、配置、通过标准与结果路径，再运行其指定入口。报告里的历史通过结论仅适用于当时的代码与配置。

新增内容物模型或对比机械臂抓取行为、角度、轨迹、视频与 CPU 速度时，通过适配器接入 [test/flex3d/](flex3d/README.md)。`flex3d` 沿用首个模型的目录名，承载的是公共平台；模型专属的本构或材料检查仍可留在本线。

每轮先在原主题报告登记问题、配置、重复数、通过标准和下一步；运行后填回同一节。结果使用新的运行标识，保留配置、源码指纹、失败与必要轨迹。协作、排队与验收标准变更遵循 [AGENTS.md](../AGENTS.md#8-本项目协作约定)。

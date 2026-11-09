# randomization

面向物流软包抓取的物理建模、响应空间随机化与策略验证。内容物、外袋与抓取任务分别维护，通过统一试验平台比较模型；论文目标见 [RSS 2027 计划](paper/rss2027_plan.md)。

## 从哪里开始

| 想了解什么 | 入口 |
| --- | --- |
| 项目目标、内容物现状和下一步 | [内容物交接](contents/HANDOFF.md) |
| 外袋模型、接口与环境 | [袋子说明](bag_sim/README.md) |
| 最近发生了什么 | [进度流水](log/progress.md) |
| 实验计划、结果与复现证据 | [实验导航](test/README.md) |
| 机械臂抓取、模型对比、轨迹与测速 | [统一抓取平台](test/flex3d/README.md) |
| 已讨论的方向与待办 | [决策](log/decisions.md) · [想法池](log/ideas.md) |
| 展示视频及其限制 | [视频说明](video/README.md) |
| 修改前的协作与文件约定 | [AGENTS.md](AGENTS.md) |

模型是否通过验收，以对应报告的版本、配置和最新结果为准。工作区源码、冻结发布包、历史快照各有用途；复现实验时使用报告指定的那一份。

## 目录地图

```text
randomization/
├── contents/        内容物模型、参数设计与交接；segment/ 为当前分段模型主线
├── bag_sim/         外层袋子模型、几何、材料与运行输出
├── policy/          策略、YAM 场景、冻结的内容物副本与结果
├── proxy_eval/      代理真实测试集与评估入口
├── isaaclab/        Isaac Lab 可行性验证及独立环境入口
├── test/            各主题的协议、实验报告、试验脚本和结果
├── reports/         开发任务的测试/检查报告
├── log/             进度、决策、想法与文献清单
├── paper/           论文计划、稿件、参考文献与模板
├── envs/            已记录的依赖版本
├── video/           供浏览的精选视频与来源说明
├── archive/         本机导入资料的归档说明；聊天原件不进入 Git
└── soft-deformable-four-assets-only-20261022/
                    原始四资产包，保留包结构、来源、许可与校验信息
```

`contents/particle/`、`contents/newton/`、`contents/rigid/` 的定位见[内容物文件地图](contents/HANDOFF.md#4-文件地图)。它们承载历史模型、对照或诊断，不能仅凭目录名判断可删除。原始四资产包的使用方式见[包说明](soft-deformable-four-assets-only-20261022/README.md)。

## 环境与运行

命令默认在仓库根目录执行。本机已有 `.venv/`；MuJoCo / MJWarp 的版本记录在 [envs/mjwarp-requirements.txt](envs/mjwarp-requirements.txt)。可先检查解释器与包版本，此命令不启动仿真：

```sh
.venv/bin/python -B -c 'import sys; from importlib.metadata import version; print(sys.version); print({name: version(name) for name in ("mujoco", "mujoco-warp", "warp-lang", "numpy")})'
```

实际运行沿用各主题文档的命令：机械臂与新模型对比从[统一抓取平台](test/flex3d/README.md)进入；内容物从[交接中的环境与入口](contents/HANDOFF.md)进入；袋子和 Isaac Lab 分别见[袋子说明](bag_sim/README.md#2-环境与运行)与[Isaac Lab 说明](isaaclab/README.md)。历史文档中的 `/home/ubuntu/Dev/...` 是原机器路径，先核对本机环境；不同后端使用的 MuJoCo / Newton 版本并不相同，依赖文件不是所有实验的通用安装清单。

正式运行遵循 [AGENTS.md 第 8 节](AGENTS.md#8-本项目协作约定)：先登记计划、记录机器名，按要求持 `/tmp/claude-gpu.lock`，新输出使用唯一的运行标识。

## 文件放在哪里

| 内容 | 存放与维护方式 |
| --- | --- |
| 模型与策略源码 | 放回对应工作线；维护入口保持唯一 |
| 实验协议、计划与结论 | `test/<主题>/`，同主题在原报告追加轮次；完整列表见[实验导航](test/README.md) |
| 任务测试/检查报告 | `reports/<任务名>-test.md`，见[报告约定](reports/README.md) |
| 进度与想法 | `log/`；流水只写关键结论与报告链接 |
| 运行数据、日志、轨迹、源码快照 | 沿用该线的 `results/`、`results_*` 或 `outputs/`，按有意义的运行标识分组 |
| 展示视频 | `video/`，在说明里关联原始试次与实验报告 |
| 导入的聊天记录及压缩包 | `archive/conversations/<日期>/`，仅在本机保留，见[归档说明](archive/README.md) |
| 虚拟环境和可再生缓存 | 使用现有忽略规则；不要混入源码或提交结果 |

结果目录中的 `reference_v*`、`source_v*`、`driver_source/` 以及 `policy/content_snapshot/` 是复现或隔离依赖所需的快照。保留其原始内容、来源和指纹；日常修改回到工作区的维护入口。实验输出和失败记录也属于证据，不能用整目录忽略或批量删除代替归档。

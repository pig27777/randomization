# 2026-11-14 运行索引

对象为随机袋装包裹候选，内容物为单刚体/三段关节，机器 ubun。问题、配置、标准、结果及限制见 [主题报告](../../content_model/parcel/REPORT.md)。所有正式 GPU 命令均加 `flock /tmp/claude-gpu.lock`，Python 为 `/home/ubuntu/Dev/Envs/mjwarp/bin/python`。

| 运行 | 入口与配置 | 状态与数据 |
|---|---|---|
| parcel-unit | test/content_model/parcel/test_physics.py | 7项通过，bag/parcel-unit/run.log |
| parcel-yam-acceptance | yam_suite.py --output <此目录>/bag/parcel-yam-acceptance | 六类各一次，3 success、3材料不足，完整轨迹/源码/模型 |
| parcel-fixtures-acceptance | validate.py --output <此目录>/bag/parcel-fixtures-acceptance | 8环境×6类×5工况，有限状态检查完成，含曲线 |
| parcel-halfstep-acceptance | validate.py --dt .00025 --output <此目录>/bag/parcel-halfstep-acceptance | 平均位移差0.141 mm以内 |
| parcel-material-r2 | material_checks.py --output <此目录>/bag/parcel-material-r2 | 悬臂刚度单调性与极值有限状态通过 |
| parcel-slip / parcel-slip-halfstep | slip_fixture.py [--dt .00025] --output <此目录>/bag/<run-id> | B隔离夹具，低/高摩擦对照与步长检查 |
| parcel-benchmark-acceptance | validate.py --mode benchmark --steps 3000 --output <此目录>/bag/parcel-benchmark-acceptance | 两拓扑，N=1/64/512/2048/4096，三模块层级，静置测速 |
| parcel-long-stability | validate.py --mode stress --n 512 --seconds 30 --output <此目录>/bag/parcel-long-stability | 两拓扑，NaN/Inf 0/1024 |

其他 parcel-* 为本轮调试/修正记录，summary、metadata 和关键日志保留，重复临时轨迹已清理。完整命令均从仓库根目录执行，脚本路径前缀为 test/content_model/parcel/。

45°补充运行 `bag/parcel-yam-yaw45-seed7-r1/`，原型3、seed=7、yaw45°、tilt0°、dt0.5ms，ubun，success。命令与关键数字见同一主题报告，目录保存模型、轨迹、源码、日志和视频。

侧面补充 `bag/parcel-yam-side-seed7-r1/` 为纯水平初始化不可达；`bag/parcel-yam-side-pitch30-seed7-r2/` 为侧面向下30°接近，close阶段grasp_fail。两次均原型3/seed7/region1、ubun，命令与结果见主题报告，r2保存真实视频和完整轨迹。

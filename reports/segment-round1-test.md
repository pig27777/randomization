# 分段内容物第 1 轮：测试报告

最近测试：2026-11-03。实验结论与详细数据见 [test/content_model/REPORT.md](../test/content_model/REPORT.md) 第 1 轮结果；本文件只记录检查范围和结果。

## 范围

`contents/segment/` 新增代码：描述层 `model.py`、MuJoCo 3.17 适配层 `mj_adapter.py`、测法 `protocols.py`/`metrics.py`、换算表 `calibrate.py`、第 1 轮入口 `run_round1.py`、Newton 1.7 检查 `newton_check.py`、MJWarp 吞吐 `mjwarp_speed.py`。

## 命令（正式运行均经 `flock /tmp/claude-gpu.lock`，在 `contents/segment/` 下）

```
/home/ubuntu/Dev/Envs/mjwarp/bin/python calibrate.py --workers 28
/home/ubuntu/Dev/Envs/mjwarp/bin/python run_round1.py t0 t1 t2 t5 t6 t7 t8 --workers 28
/home/ubuntu/Dev/Envs/mjwarp/bin/python mjwarp_speed.py --steps 500 [--no-tendon-damping]
/home/ubuntu/Dev/Envs/newton_latest/bin/python newton_check.py [--limit-ke 1e8 --anchors compressible]
```

日志：`contents/segment/results/logs/`；结果：`contents/segment/results/*.json`。

## 结果

| 项 | 结果 |
|---|---|
| T0 MuJoCo（质量、质心、惯量） | 通过，8 例误差 0 |
| T0 Newton | 闭环可建可解；硬物 R1 稳定；易压物 R1 失败（D6 压缩限位不生效，超限 1.2 倍） |
| T1 静置 + 能量 | 通过 |
| T2–T4 锚点 R1/R2/R3 | 不失败；易压物 R2 竖直合力 0.884，超出 ±5% |
| T5 响应命中 | 软物 18 项中 14 项 ≤ 20% |
| T6 可控性 | 夹紧单调；下垂低端平台，1 例弹飞 |
| T7 随机 200 例 | 失败 13 例（6.5%），标准 ≤ 1%，未通过 |
| T8 活动质量 | 通过（摆幅 ×1.9） |
| MJWarp | 带 tendon 阻尼发散；去阻尼后测吞吐 |

## 遗留问题与限制

- T7 失败机理（顶住后的段被挤出）未解决；阻尼和锁死上限的对照只部分有效。
- MJWarp 的 tendon 阻尼发散原因是推测（隐式积分未含 tendon 阻尼），未查源码。
- Newton D6 限位不生效是根据限位刚度 1e6 / 1e8 结果相同推断，未查源码。
- 测法补充（夹爪力上限 50 N、“到…的 N%”按间隙、标定时提起 0.3 m）与 4 处测法问题待用户决定。
- 中途的探索运行（未排队）只用于定位问题，不进结论；第一次正式运行的结果已被最终代码的正式运行覆盖，T7 第一版的失败率 10.5% 在实验报告中保留。

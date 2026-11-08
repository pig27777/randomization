# 袋子第 1 轮（C1–C3，松装表示）测试检查报告

最近测试：2026-11-05。实验报告：[test/bag_model/REPORT.md](../test/bag_model/REPORT.md) 第 1 轮结果；原始结果在 `bag_sim/outputs/round1/`。

## 范围

- 新代码：`bag_sim/mj_loose.py`、`loose_protocols.py`、`loose_speed.py`、`c1_newton.py`、`wrap.py`（`wrap_generic` 等从 `demo_contents.py` 移出，`demo_contents` 保留同名导入，供 `proxy_eval/objects.py` 使用）。
- 未改内容物代码（`contents/segment/`）、任务协议、判断标准和共享环境。

## 检查方法与结果

| 检查 | 命令 / 方法 | 结果 |
|---|---|---|
| 模块移动后旧入口可用 | Newton 环境 `import demo_contents`，确认 `wrap_generic` 来自 `wrap`；`c1_newton.py` 冒烟（`C1_TMAX=0.4`，R2、R3 各一次） | 通过；`proxy_eval/objects.py` 的 `from demo_contents import wrap_generic` 不变 |
| 静置 | 薄板片 + 松装硬盒静置 0.5 s（CG / Newton 求解器各一次） | CG 残余 0.1 m/s、100 次迭代不收敛；Newton 1 次迭代、残余 2 µm/s，内容物不出袋 |
| 求解器对内容物的影响 | `outputs/round1/solver_check.py`：易弯物、易压物 × R2 端部、R3 × CG/Newton，共 8 次 | 滑移、夹持力、承重一致；易弯物 R2 下垂比 0.177 对 0.109，易压物 R3 内部点运动 13.2 对 0.87 mm（已记 I-27） |
| C1 Newton | `outputs/round1/run_c1.sh`（`flock /tmp/claude-gpu.lock` 排队）：松装/满装 × R2/R3 | 4 次完成；满装 R3 有穿透（指垫间距 137 mm < 盒宽 150 mm），记为仿真失败 |
| C1 对照 | `outputs/round1/c1_control.py`：两种硬盒 × R2/R3，内容物模型、无袋 | 4 次完成 |
| C2 | `outputs/round1/run_c2.sh plates plates_rigid '{"compress": 0.0}'`（3 种内容物 × 6 项）；软化变体、30 mm 板片、flex 各一组 | 薄板片面内刚性 18 项全部完成；flex 4 项完成（其中 R2 无效）；变体的结果和失败方式见实验报告 |
| C3 | `loose_speed.py`：单环境 1 次 + 28 进程随机参数 | 薄板片：硬盒 0.20 倍实时、0.41 次/s，易弯物 0.035、0.070 次/s；flex 硬盒 0.0058、0.016 次/s；共 140 次，0 错误 |
| 确定性 | 同一 CLI 参数连跑 2 次 | 结果逐位相同；ψ 相差 4e-6 rad 时结果从夹取失败变为掉落（混沌敏感，已写入报告） |

全部 C2/C3 运行都没有仿真失败（非有限值、越限、内容物段中心出袋）。

## 遗留问题与限制

- R2 的“沿袋滑动”由记录的末端位移和按几何估算的起点算出（±5 mm），不是逐帧量。
- C1 对照硬盒用的是 MuJoCo 内容物模型，不是 Newton。
- 速度测量时同机有 1–3 个其他运行；C2 的单次实时率是在 20–40 个进程同时运行时测的，只作相对参考。
- flex 的盒角点在三角面外（最高 88%）没有目视确认原因。
- 探索阶段的运行（原型、调试）未进入结论，未排 GPU 锁（都是 CPU）。

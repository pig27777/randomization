# Plan B 单臂入篮测试报告

2026-11-23 19:05，北京时间。主机 `ubun`，RTX 5090，原生 Isaac Sim 6.2 / Isaac Lab 3.0。GPU 测试均使用 `flock /tmp/claude-gpu.lock`，2 ms 物理步，25 Hz 控制。

## Step 1 同载荷顶抓

C5 扫描在 20 分钟上限内停止。每档一条，首轮 12 条，第二轮 30 条。第二轮调平指垫、按 CAD 切换探测下沿，并扫描下压速度、闭合开始时刻和水平扫程。两轮的闭环动作都由预定轨迹与 YAM FK 生成。

| 检查 | 结果 |
| --- | --- |
| C4 硬书顶压 | 2 ms 峰值 11.696 N，300 帧零捕获，净空 0 mm |
| C5 首轮 | 双指捕获 0/12。顶压峰值不超过 12 N 的轨迹，最佳双侧聚拢 0.914 mm |
| C5 调平和扫掠 | 双指捕获 0/30。最佳双侧聚拢 0.450 mm，对应峰值 24.942 N。单侧捕获最高只让包裹短暂抬起 15.564 mm |
| C6 及 C3 | 沿用 18:14 已验收的真实闭环，C6 连续稳悬 3.79–3.93 s，C3 高夹末 3 s 最小净空 42.46 mm |
| 空载 0 / 4 / 30 / 35 N | 四档有效法向力与捕获峰值均精确为 0 |
| 双通道端口功与电流隔离 | −0.191999963 mJ，`wrist_port_pass: true`。6 N/指静态去皮电流等效 +5.98164 N，隔离通过 |

C5 的问题仍出现在压着闭合却未能双侧聚拢的阶段。更新后的闭环门禁要求有效闭合段处于 8–12 N、双侧聚拢均超过 1.50 mm，且实际双指捕获和完全卸载离台出现在同一帧。C5 未放行，Step 1 未打标签。C10 按用户指示搁置。

冻结的 `lumped_contact_proxy.py` 和 `unified_payload.py` 对 `fix4.1-route-b-base` 路径级 `git diff --exit-code` 通过。外载端口 `yam_load_port.py` 对 `736db9b2` 路径级检查相同。仓库树包含冻结后已批准的路线 B 文件，因此仓库全树与冻结提交有差异。

## 单臂入篮场景

默认篮口 42.0 × 30.2 cm，可配置范围 40.0–44.0 × 30.0–30.6 cm。中心为 `(−0.03, +0.03, 0) m`，四面静态壁，篮沿 12 cm。随机起始偏航和 Y 位置受真实近侧篮壁几何约束，大包的可用偏航区间随包络变窄。包裹剐蹭采用四个子块的包络与篮壁重叠检查，斜穿墙角时判定偏保守。演员为 63 维，动作 7 维，评论家 171 维。四路受控状态检查覆盖正常越沿及落篮、低位跨越、超高松爪和包裹局部越界。该检查只验证终态逻辑。

| 8,192 路默认篮口空载压测 | 测量 |
| --- | ---: |
| 计时物理子步 | 240 |
| 耗时 | 5.306 s |
| 物理 env-steps/s | 370,531 |
| GPU 增量显存 | 4,146 MiB |
| GPU 总占用 | 5,455 MiB |
| 指垫中点平均待机高度 | 189.993 mm |
| 每 2 ms 检查的最大 Z 漂移 | 0.00018 mm |
| 起始避开篮壁 / 提前结束 | 8,192 / 0 |

完整抓取入篮轨迹、4,000 条成功示范、BC 闭环、PPO 和真机导出仍待 Step 1 放行。

## 复现

按路线 B 原生 Isaac Lab 运行环境加入 Isaac Lab 源目录和仓库根目录到 `PYTHONPATH`。以下均在 GPU 锁内执行。

```sh
"$native/python.sh" -B test/isaaclab/check_route_b_wrench_power.py --headless
"$native/python.sh" -B test/isaaclab/check_route_b_closed_loop.py --headless --empty-clamp
"$native/python.sh" -B test/isaaclab/check_route_b_single_arm_bin.py --headless --num-envs 4 --benchmark 2 --check-gates --opening-m 0.44 0.306
"$native/python.sh" -B test/isaaclab/check_route_b_single_arm_bin.py --headless --num-envs 8192 --benchmark 12
```

逐档结果、C4 独立复跑和 C5 扫描源码见[运行目录](../test/runs/2026-11-23/isaaclab/route-b-final-gates-r1)及[汇总 JSON](../test/runs/2026-11-23/isaaclab/route-b-final-gates-r1/plan-b-step1-summary.json)。工位配置见[route_b_single_arm_bin.yaml](../configs/route_b_single_arm_bin.yaml)。

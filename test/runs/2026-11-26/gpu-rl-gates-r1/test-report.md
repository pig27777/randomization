# GPU RL 门禁验证记录

## 范围与环境

`ubun`，NVIDIA GeForce RTX 5090，`.venv-physical-parcel` 内 MuJoCo / MJWarp 4.0.1。以仓库 `b5d89681` 和 Policy V3 基线 `9d8d201d` 为起点。GPU 一律经本轮 `run_gpu.sh` 入口取得 `/tmp/claude-gpu.lock`。独立 VBD 用同一解释器，通过 `benchmark-deps` 载入原夹具使用的独立 Newton 后端。

## 验证方法

正式运行的输出目录须为空或尚未创建。以下路径从仓库根目录解析。

```bash
run=test/runs/2026-11-26/gpu-rl-gates-r1
cpu=test/runs/2026-11-25/bag/pe-mvp-r1/through-r3
pe=.venv-physical-parcel/bin/python
PYTHONPATH="$PWD/src:$PWD/experience:$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$pe" -B -m unittest discover -s tests/policy_v3 -v
"$run/check_frozen.sh"
for config in a1 a2 a3; do
  "$run/run_gpu.sh" "$run/flex_fixture.py" "$config" "$run/$config" "$cpu"
done
"$run/run_gpu.sh" "$run/arm_grasps.py" "$run/arm-body" --tuning a2 --modes film
"$run/run_gpu.sh" "$run/probe_replay.py" "$run/probe-selective-limit" --modes left
PE_EXTRA_PYTHONPATH="$PWD/.venv-physical-parcel/benchmark-deps" \
  "$run/run_gpu.sh" "$run/vbd_fixture.py" "$cpu" "$run/b1" --scale .06
PE_EXTRA_PYTHONPATH="$PWD/.venv-physical-parcel/benchmark-deps" \
  "$run/run_gpu.sh" "$run/vbd_fixture.py" "$cpu" "$run/b2" --scale .025
"$pe" -B "$run/plot_evidence.py"
```

实际执行中三位置 1 mm 试探使用 `probe_replay.py` 的 `--modes left center right`，修正防旁路和 OBB 后开始。随后才将试探指垫－膜分离距限额调到 2 mm。整臂隔袋第一次采集使用 `arm_grasps.py --tuning a2`，以归档 qpos 在实际刚体坐标重算几何。过渡版本的坐标弃例由主报告注明。

## 结果

- Policy V3 既有回归通过 12／12。新增断言覆盖夹具－内芯过滤、夹具－袋膜接触、编译网格重定向后的内芯 OBB，以及模拟 2 N 触发后的平扫、撤回和 `v1` 采集。模拟触发检查只覆盖控制逻辑。 [单测日志](policy-v3-regression.log)
- 冻结分支与运行前基准的限定路径检查均退出 0。原作者的历史差异按本轮起点核对。 [冻结检查](frozen-check.log)
- 路线 A 三组隔袋夹具均失败，两个整臂抓取入口均在执行接近时触发指垫－膜距离硬停。正向力、伸长、侵入和稳持数据见主报告。 [门禁报告](rl_gates_report.md)
- 路线 B 两组回授降力均失败，2 ms 回读峰值为 85.35／76.38 N，两组有效稳持均为 0 s。 [B1](b1/summary.json) · [B2](b2/summary.json)
- C0 熔断后未执行 100 回合、C2、C3 和多卡吞吐测算。PNG 由同轮实际力轨迹生成。 [C1 状态](gate2_c1_100ep_summary.json) · [力曲线](gate1_probe_and_grasp_forces.png)

# Flex 薄膜修复测试记录

环境为 `ubun` 的 RTX 5090，使用 `.venv-physical-parcel`，以 Policy V3 `2ee467a3` 为代码基线。数值进程设 OMP/OpenBLAS/MKL 各 1 线程。GPU 各命令经过 `run_gpu.sh` 及 `/tmp/claude-gpu.lock`。

## 验证命令

以下目录从仓库根目录解析，复跑需使用空输出路径。

```bash
run=test/runs/2026-11-26/flex-film-fix-r1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PE_PYTHON=/home/ubuntu/workspace/gzl/randomization/.venv-physical-parcel/bin/python
PYTHONPATH="$PWD/src:$PWD/experience:$PWD" "$PE_PYTHON" -B -m unittest discover -s tests/policy_v3 -v
"$run/check_frozen.sh"
"$run/run_gpu.sh" --step1 "$run/fixture_retest.py"
"$run/run_gpu.sh" "$run/arm_retest.py" "$run/first-left" --modes left
"$run/run_gpu.sh" "$run/continue_retest.py"
"$PE_PYTHON" -B "$run/plot_evidence.py"
```

## 核验

- 既有 Policy V3 回归全部 12 项通过。 [日志](policy-v3-regression.log)
- 冻结标签、原配置及本轮提交基准的限定路径检查均退出 0。 [日志](frozen-check.log)
- 编译模型读回 A2 为 200 µs、360／75 次、`impratio=8`、`margin=0.8 mm`。A3 为 100 µs、450／100 次、`impratio=10`、`margin=1.0 mm`。两配置夹具与内芯的碰撞掩码均禁止直接碰撞。
- CPU Warp 的三条接触记录选择器检查通过。左指垫－膜使用 4 ms 覆盖值，指垫－台面和另一几何体－膜逐位保留原值。该试验覆盖配对与参数写入逻辑。
- CPU Warp 求解力护栏的合成记录用例通过。覆盖 2 N 软截断与触发深度记录、同帧超限时硬停、直触内芯终止和单指 25 N 上限。[日志](safety-kernel-smoke.log)
- `2026-11-26 00:53–01:40` 完成正式 GPU 阶段。A2/A3 在单指力 31.39／26.66 N 停机。三位置均在 0.3–0.8 mm 区间触发软截断，右位完整通过。捏膜夹具及两条整臂试次均未到稳持。
- 补充了同帧力硬停与露角并存的回归断言，期间修正测试夹具漏导入，复跑 12／12 通过。原生探针的几何与力状态分列在汇总中。
- 最终未放行 C1。未生成胜出抓取视频。[裁决报告](flex_fix_report.md) · [试探图](probe_3pos_forces.png) · [抓取图](grasp_a2_a3_forces.png)

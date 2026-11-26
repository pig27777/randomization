# Yam_B GPU 求解器提速与捏膜提拉测试

最近验证于 2026-11-27（北京时间），机器 ubun，RTX 5090，MuJoCo / MJWarp 4.0.1。

- 求解压测按 `yam_gpu_solver_sweep.py` 每档实测 5 帧，GPU 命令外层用 `flock /tmp/claude-gpu.lock`。800 µs Newton 档在 64 路墙时由 15,348 ms/帧降为 3,523 ms/帧，4.36×。256 路 56.52 env-steps/s，进程显存峰值 1,637 MiB。
- 按 `yam_gpu_acceptance.py` 做整臂物理筛选。800 µs 与 500 µs 的 A2 各有 1/3 试探过关，nominal 666.7 µs 为 2/3。16 条捏膜 0 条入篮，两条边缘候选完成提拉阶段但没有带起内芯。C1–C3 仍受物理门禁阻塞。
- Policy V3 原有 12 项与子步接触回归 4 项通过。冻结路径按已有 `check_frozen.sh` 核查全部退出 0。

[实验报告、复现命令与原始数据](../test/runs/2026-11-26/yam-gpu-speedup-lift-r1/speedup_and_lift_report.md)

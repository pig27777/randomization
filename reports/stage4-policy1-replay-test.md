# Stage 4 Policy 1 录像检查

2026-11-22 19:47（北京时间），`ubun` RTX 5090。以训练分支 `86a76e1c` 的 Policy 1 权重和固定角点种子 88003／传感器种子 88004，持共享 GPU 锁回放 4,096 路。逐角成功数为 0、512、512、512、512、12、0、512，与正式盲测完全一致。

录像选取 C3 第 2 行成功回合和 C6 第 5 行失败回合，行号从 0 开始。两条均首选侧抓，均走正常提拉。25 mm 微提拉段两条净空均为 0 mm。C6 随后离台且末帧双指仍捕获，整回合未达到稳悬成功条件。画面使用冻结模型的四块载荷和指垫代理状态。

| 检查 | 结果 |
| --- | --- |
| 首轮入口 | 记录变量误用环境历史窗口字段名，启动失败。已改用独立变量并修正抽样厚度索引。 |
| 回放与视频 | `ffprobe` 确认为 H.264、1280×584、334 帧、13.36 s。FFmpeg 全片解码成功，抽查末帧的对象、结果和数值标注。 |
| 桌面播放 | `ffplay -loop 0` 启动，X11 窗口标题为 `Policy 1 replay | C3 vs C6`。 |

```sh
PYTHONPATH="$PWD/.venv-physical-parcel/stage3-cu130:$PWD" OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl flock /tmp/claude-gpu.lock \
  .venv-physical-parcel/bin/python -B test/low_order_parcel/replay_stage4_policy1.py \
  test/runs/2026-11-22/policy/stage4-policy1-ours-s17/training <新输出目录>
```

[MP4](../test/runs/2026-11-22/policy/stage4-policy1-replay-r2/policy1-c3-c6.mp4) · [轨迹与摘要](../test/runs/2026-11-22/policy/stage4-policy1-replay-r2/) · [首轮日志](../test/runs/2026-11-22/policy/stage4-policy1-replay-r1.log)

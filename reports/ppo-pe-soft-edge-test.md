# PPO 训练用 PE 软包边缘提拉

2026-11-23 15:34（北京时间），ubun。C5 为 500 g 折叠衣物，PE 接触代理，24 × 18 × 5.6 cm，体积填充率 60%。训练工作树 `stage4/fix4-training` 的 `e103f1e9` 收录最近本地 YAM URDF pilot。训练时的三个源码 SHA-256 与运行入口逐一一致。第 3 轮检查点 SHA-256 为 `fc5eb43d39266441f96207878656da2124389d2bfcda7a72dd811044935dda6d`，URDF 与网格指纹均与训练结果相同。本次只运行推理和包裹仿真。

| 试验，各一次 | 指垫及材料位置 | 结果 |
| --- | --- | --- |
| 隔离包裹响应，CPU 受控指垫，3,000 个 2 ms 步 | 复用标准侧抓输入，名义 6 N/指、约 15.5 mm 进给，固定指垫轨迹，不经过策略或关节控制。指垫与材料坐标均指定在前边 y=−90 mm。原双臂接触核从零捕获自行决定闭锁、滑动、失夹与撕裂。 | 0.090 s 双侧捕获；1.224–6.000 s 所有采样均双侧捕获、支撑 0 N、净空 >1 mm。倾角最大 56.52°；最后 1 s 为 51.26–53.20°，末帧 52.90°，净空 74.96 mm。 |
| 训练版 YAM 与固定策略，RTX 5090，6,163 个 2 ms 步 | 指定顶抓与前边内 4 mm 的水平目标。下压、夹紧、微提拉及验证来自既有策略和动作。保留训练中线挂点。 | 12.326 s 内两指均未捕获，台面支撑维持 4.905 N。闭合末帧实际指尖中点在边内 4.94 mm，距袋顶约 99.5 mm。未产生悬挂姿态。 |

倾角取近侧两块的平均中心指向另一排平均中心的高差，用三维水平投影长度计算。水平为 0°，竖直为 90°。录像按原 2 ms 状态采样，画面为四块内容物及模型指垫。模型以有余料和有限拉力的接触代理表示 PE 膜。边缘工况调用原生 `mount_xy`，四块插值把传力点限制在最外排子块中心 y=−45 mm。结果只描述模型可表示的边缘承载。模型净空沿用训练求解器输出。

## 复现与检查

从仓库根目录执行，`out` 为新建的空目录。GPU 训练环境依赖使用试训原虚拟环境；隔离试验使用该环境的 CPU。

```sh
base=$PWD
wt="$base/.git/local-worktrees/stage4-fix4"
out=$(mktemp -d)
cd "$wt"
sha256sum -c test/runs/2026-11-22/policy/stage4-yam-urdf-pilot-s17/source.sha256
export PYTHONPATH="$wt/.venv-physical-parcel/stage3-cu130:$wt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=.venv-physical-parcel/bin/python
tools="$base/test/low_order_parcel"
flock /tmp/claude-gpu.lock "$py" -B "$tools/replay_ppo_edge.py" test/runs/2026-11-22/policy/stage4-yam-urdf-pilot-s17/training "$out"
flock /tmp/claude-gpu.lock "$py" -B "$tools/ppo_edge_pad_probe.py" "$out/pad-probe.npz"
"$py" -B "$tools/render_edge_reaction.py" "$out/pad-probe.npz" "$out/edge-hang.mp4" --probe
"$py" -B "$tools/render_edge_reaction.py" "$out/trajectory.npz" "$out/policy-grasp.mp4"
```

轨迹全步有限，独立检查相邻离台采样连续、尾段持续双侧承载、无撕裂。YAM 回合零缓冲溢出。[受控提拉录像](../test/runs/2026-11-23/policy/ppo-yam-pe-edge-r1/edge-hang.mp4) · [YAM 边缘指令录像](../test/runs/2026-11-23/policy/ppo-yam-pe-edge-r1/policy-grasp.mp4) · [原始状态与日志](../test/runs/2026-11-23/policy/ppo-yam-pe-edge-r1/)

两段 H.264 录像均经 ffprobe 核对尺寸（1152 × 620）和帧数（150／309），并以 `ffmpeg -v error -xerror -i <录像> -f null -` 完整解码通过。

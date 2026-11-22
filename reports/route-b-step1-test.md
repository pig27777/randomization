# 路线 B Step 1 · 腕部回写无源性门禁

2026-11-22 22:33（北京时间），`ubun` RTX 5090，Isaac Sim 6.2 / Isaac Lab v3.0.0 / PyTorch 2.11.0+cu130。**Step 0 通过。Step 1 的腕部回写功率门禁失败，暂停。** Fix 4.1 冻结提交为 `2ef4a3943183c527946c3ebc0ba171ea6c163094`，标签 `fix4.1-route-b-base`。冻结物理源文件核对一致。[Step 0 资产盘点](route-b-step0-test.md)

| 检查 | 实测结果 |
| --- | --- |
| ABC 实际滑轨运动 | 由 PhysX 读回，两轨各以 0.400 m/s 相对腕部闭合。腕部角速度取实际读回。 |
| 冻结 Fix 4.1 配对 | 65 g 已捕获的构造工况，一个活跃处、0.10 m 膜颈端距、6 N 法向力。冻结阻尼给出 0.480 N 横向力。以子步起始速度和 2 ms 折算，完整配对做功 **−0.192 mJ**。 |
| 腕部合力／力矩回写 | 同一力在腕部中心等效合算，与包裹侧力功合计 **+0.192 mJ**，对应净耗散 **−0.192 mJ**，低于 **−0.1 mJ** 门限。缺失的滑轨相对功为 **−0.384 mJ**。 |
| 对称夹持负载 | 每指 6 N 的共线相反反力，腕部合力 0 N、合力矩 0 N·m。按实际运动轴投影后，滑轨广义外力约为 **−6.000 N／轨**。 |

端点回写功等于腕部合力和力矩的功，再加上滑轨相对运动的功。每指缺失的功率等于该指反力与指垫扣除腕部平移和转动后的相对速度的点积。有载闭合时，遗漏项可改变阻尼功的符号。静态对称夹持也会在腕部通道消失。减小步长可减小单步能量量级，功率符号与夹持反馈仍需通过接口解决。

本项使用真实 PhysX 关节运动与冻结阻尼函数，在固定载荷条件下计算子步起始端口功。C6 实际抓取、完整六工况、关节电流估力和 1,024／2,048／4,096 路吞吐均待实现或验收，BC/PPO 未启动。

待确认的接口方案。保留腕部局部系合力和力矩。将每指反力沿滑轨运动轴投影成广义外力，写入 PhysX 的滑轨外载通道，与电机命令和电机电流读数分开。以同一功率共轭方向补入组合有效惯量，再复验 2 ms 的无源性和稳定性。方案无需将笛卡尔外力直接写到轻质指垫连杆。

## 复现

```sh
native=/home/ubuntu/workspace/czx/system0/sim
lab="$native/IsaacLab/source"
export PYTHONPATH="$lab/isaaclab:$lab/isaaclab_physx:$lab/isaaclab_newton:$lab/isaaclab_visualizers:$lab/isaaclab_ov:$lab/isaaclab_contrib:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
flock /tmp/claude-gpu.lock "$native/python.sh" -B test/isaaclab/check_route_b_wrench_power.py --headless
```

预期脚本输出 `wrist_port_pass: false` 并以 1 退出。首次探针的门禁结果正确，退出码检查发现 Kit 在离开启动上下文时提前退出。将失败状态在 Kit 上下文内返回后复验。 [复验日志](../test/runs/2026-11-22/isaaclab/route-b-step1-r1/wrist-power.log)

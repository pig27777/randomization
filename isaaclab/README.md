# Isaac Lab 可行性验证

本目录由 Isaac Lab 线维护。计划、通过标准和实验结果统一写在 [第 1 轮报告](../test/isaaclab/REPORT.md)，检查记录在 [任务检查报告](../reports/isaaclab-round1-test.md)。其他线代码只读调用。

## 独立环境

- 环境：`/home/ubuntu/Dev/Envs/isaaclab`。
- 官方源码：环境内 `src/IsaacLab`，tag `v3.0.0`，commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`。
- 使用官方 `uv.lock`，用户已批准加入 `isaacsim` extra（6.2.0.0）。发布 tag 为 v3.0.0，内部核心包版本为 28.0.0；安装状态以报告为准。
- 独立环境采用 `requirements-compat.txt`：NumPy 2.3.1 满足 Numba、Isaac Sim 和 cmeel-boost 的交集；ONNX 使用机器人组件声明的 GPU 发行包，避免 CPU/GPU 发行包覆盖同一模块。
- 缓存与临时文件位于环境内 `cache/`，不改共享安装、驱动、全局配置或现有 Python 环境。

从项目根目录运行以下入口；`run.sh` 直接使用独立环境解释器，避免自动同步移除 Isaac Sim 或恢复锁文件中无法被 Numba 导入的 NumPy 2.5.1。

```bash
./isaaclab/run.sh <Python 脚本及参数>
```

安装复现记录位于 `results/install/`。注册表依赖通过官方锁导出的 `pylock.initial.toml` 安装，保留逐包来源与 SHA256；普通多索引 requirements 会错误选中不在锁定来源上的 CUDA torchaudio，失败记录已保留。Kit、Kit SDK、Open3D 与兼容补充包使用 `download_wheel.py` 分段下载，整包匹配官方锁或 PyPI 元数据 SHA256 后才安装；12 个 Isaac Lab 包直接从固定 tag 的本地 workspace 以 editable 安装。未修改官方源码或锁文件。

官方根配置明确覆盖 Isaac Sim 的旧 `fsspec`/`urllib3` 固定版本，分别保留 2026.6.0/2.8.0；通用依赖检查仍会报告这两项元数据冲突，不能记为检查全通过。

## 检查入口

`cloth_benchmark.py` 使用未修改的官方 `Isaac-Lift-Cloth-Franka` / `newton_mjwarp_vbd_proxy`，统计预热后的完整环境步；显存为 0.5 秒间隔采样的进程占用峰值，另列 PyTorch 分配器峰值。它不统计成功抓取，也不能替代 L5/L6。

```bash
OMNI_KIT_ACCEPT_EULA=yes flock /tmp/claude-gpu.lock ./isaaclab/run.sh isaaclab/cloth_benchmark.py \
  --num_envs 1 --warmup 100 --steps 1000 --output isaaclab/results/l1/n1-r1.json
```

`check_isaacsim.py` 单独检查 Isaac Sim 无窗口启动、USD stage 和 10 次更新，使用环境内 portable 缓存。首次使用须遵循 NVIDIA Isaac Sim / Omniverse 许可。

```bash
OMNI_KIT_ACCEPT_EULA=yes flock /tmp/claude-gpu.lock ./isaaclab/run.sh \
  isaaclab/check_isaacsim.py --output isaaclab/results/install/isaacsim-smoke.json
```

`cpu_reference.py` 只读调用本线 `results/reference_v1/` 内经校验的冻结源码副本。必须用 CPU 参考环境，不能改用 Isaac Lab 自带的 MuJoCo 3.12 来替代 MuJoCo 3.17 参考。

```bash
/home/ubuntu/Dev/Envs/mjwarp/bin/python -B isaaclab/cpu_reference.py --batch
```

batch 中每个配置分别持共享锁；默认三类锚点 × 静置/R1-y/R1-z/R2 端夹/R2 中夹/R3 × 3 次。`--test describe` 只导出同一冻结描述，不运行动力学。结果文件拒绝覆盖，重跑时指定明确的新输出目录，并更新同一报告。

`segment_adapter.py` 检查原生 D6 图或 `--loop-mode proxy` 的 D6+WELD 替代图，`--rest` 增加 0.3 s 静置；跳过闭环返回2，非有限返回3。它尚不是通过R1–R3验收的完整内容物后端。`parameter_probe.py` 用隔离测具检查逐世界参数的引擎读回和运动响应；`structure_probe.py` 检查混合图/同结构分桶的构建，不能替代分桶动力学。配置、失败与仍未完成的项目统一见实验报告。

## 结果归属

- `results/preflight.json`：安装前状态，包含输入指纹；其 pending 授权字段为当时快照，后续授权见报告。
- `results/install/`：下载与安装日志、启动检查。
- `results/reference_v1/`：冻结包的运行副本，保持原样，不作为新的维护入口。
- `results/descriptions/`：三类锚点的描述与质量/质心检查。
- 其余正式动力学输出按报告中的配置归档。

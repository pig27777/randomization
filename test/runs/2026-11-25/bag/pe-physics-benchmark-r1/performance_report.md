# PE 袋基准的 GPU 功能和计时边界

机器为 `ubun`，Intel Core i9-14900KF，RTX 5090 32,607 MiB，驱动 595.80。实测环境为 MuJoCo 4.0.1、MJWarp 4.0.1、Warp 1.19.0、Newton 1.6.1。Newton 装在本次工作树隔离的依赖目录。

| 最小用例 | MuJoCo CPU | MJWarp CUDA |
| --- | --- | --- |
| 1D Flex 与边长等式 | 扰动产生非零等式力 | 3 行有非零等式力 |
| 2D Flex 面内弹性 | 扰动加速度范数 2150.671 | CUDA 为 2150.665，无弹性对照均为零 |
| 2D 原生弯曲 | 扰动加速度范数 0.068301 | CUDA 为 0.068301 |
| 3D Flex 面内/体弹性入口 | 此装配加速度范数 268.731 | CUDA 为 268.731 |
| 2D 边长等式 | 16 行 | 读取等式力范数 3.95×10⁻⁵ N |
| 2D 膜层自接触 | 4 个正法向力接触 | 2 个正法向力接触，力值与 CPU 有差异 |
| 2D 膜对台面摩擦 | 法向 0.980 mN，切向 0.490 mN | 在此用例与 CPU 一致到所列位数 |
| 异质复位的两路 CUDA Graph | 基本用例参考 | 两路不同状态各前进 50 μs，容量检测为零 |
| 原仓库 142 节点 PE 袋 GPU 路径 | 原二面角弯曲参考 | 两路 Graph 各推进 100 μs，状态有限，容量检测为零 |

本机安装源码的能力检查覆盖 `io.py` 的 Flex 导入及边长等式行、`passive.py` 的膜弹性和弯曲、`forward.py` 的离散积分、`support.py` 的接触力读回。实测中，MuJoCo 4.0.1 的原生弹性使用 `discrete`。使用 `implicitfast` 启动原生 2D 弯曲或 3D 弹性会被 CPU 模型编译器拒绝。旧版环境的功能范围仍需单独探测。[最小用例数据](../pe-physics-capabilities-r1/capabilities.json) · [原袋 CUDA Graph 数据](../pe-physics-parcel-cuda-r1/capabilities.json) · [MuJoCo XML 参考](https://mujoco.readthedocs.io/en/stable/XMLreference.html)

Newton 通过 CUDA 运行现有 VBD 模型，空袋等无材料移动夹具的工况可在接触距离加载斜坡结束后捕获子步。运行日志保留冷 JIT 与缓存构建的实测时间。正式 B12 五类工作负荷、批量规模和有效环境步/s 因基础物理门槛尚未通过而留空。两路 CUDA 最小检查只证明运行能力。Newton 的 CPU、MJWarp CUDA 和 VBD CUDA 结果分别记录，当轮未用 CPU 替代 GPU。GPU 运行均显式记录设备，探索入口日志与持锁正式运行分开。[VBD 运行日志](../pe-physics-vbd-contact-r2.log) · [Newton 1.6.1 SolverVBD](https://newton-physics.github.io/newton/1.6.1/api/newton_solvers.html)

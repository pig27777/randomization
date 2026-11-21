# 统一内容物模板检查报告

## Step 1 · 接口审查

本轮计划：在 `ubun` 上核对当前工作树和已抓取的 `origin/main` 各一次，追踪探针动作、接触输入、挂点力和腕部输出。用指定 Python 的标准库检查字段顺序，用单侧水平拉力检查顶部偏移的力矩。接口一致且载荷功率配对才判为可直接接入；发现冲突则记录所需口径。本轮是设计审查。

最近检查时间为 2026-11-20 20:12（台北）。本地版本为 `df6b256f`，远端参照为 `39faeab0`。`git pull --ff-only` 已抓取远端，两端历史分叉，快进失败。以下结果同时核对了两个版本。

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| 冻结接口 | 用 `.venv-physical-parcel/bin/python -B` 的 `ast` 解析工作树和 `git show 39faeab0:<path>` | 两个版本均通过 5 个具名元组字段顺序检查，覆盖 ContactInputs、ContactStep、ParcelState、ParcelAction、ParcelStep。代理源码在两个版本间无差异。 |
| 坐标与力 | 逐项追踪挂点生成、`proxy.step` 调用和 `ParcelStep` 返回 | SI，世界 +z 向上，桌面 z=0，刚体姿态为 wxyz。左右顺序固定。初始顶部挂点相对几何中心为 (±0.024, 0, hz) m，质心偏移先在局部扣除。线速度与角速度均按世界轴使用。 |
| 力矩反例 | 取中面水平、顶部高 0.016 m、挂点受 +x 向 1 N，对质心算 y 向力矩；取 y 向角速度 1 rad/s 检查瞬时功率 | 顶部力矩为 0.016 N·m，按权重将力平移到中面后为 0。两种分配的功率相差 0.016 W。 |
| 执行环境 | `.venv-physical-parcel/bin/python -B -c 'import torch'` | 当前机器该环境缺少 torch。本轮仅执行静态检查和标量反例。P1–P6 及物理包裹 20 项回归留待实现阶段。 |

### 接入决定建议

1. 在 `MacroParcelSim` 构造时注入可选模板，批内材质差异只来自 X。模板状态随每个环境的步进和重置显式传递，并为原有探针提供 com、姿态、速度与 contacts 视图。四块的质心偏移由质量权重与几何决定，避免与旧 `com_offset` 重复叠加。基于材料坐标选择左右挂点，刚性标定须回到原先 ±24 mm 的位置。特权状态单独提供给 Critic。
2. 冻结输入的四组位置／速度是世界系 [B,2,3]，质心为 [B,3]，法向压缩力、带符号内收速度、进给深度和对齐余弦各为 [B,2]。接触状态 [B,2,6] 的末轴依次为 slack、capture、slip_cum、hold_time、feed_depth、peak_T。返回的 force 作用在包裹上，方向从挂点指向指垫，torque 为绕包裹质心的力矩。现有腕力为两指绳力之和取负再加向上直压力，腕矩为各指垫反力绕双垫中点的力矩之和。压陷、触底及夹口反力需按指成对记账；同一顶压载荷只计算一次。
3. 进给、对齐和夹口法向力可逐步经 ContactInputs 注入。S_bag 是构造参数，代理在聚拢速率和全局余料裁剪两处读取它。严格限定只经 ContactInputs 映射时，动态全局余料无法实现。建议明确允许封装层为每个模拟实例独占一份预算张量，保存独立 S_bag,0，每步在代理调用前填入受几何上限约束的 S_bag,eff。这样可保留冻结方程并隔离同时运行的模拟实例；回弹时预算收缩仍须检查左右余料之和。顶压探针用预设力加载，P3 的真实压入和释放需另给逐指位移轨迹。
4. 挂点若包含随状态变化的表面法向偏移，速度应由完整挂点运动学给出，力应以同一映射的雅可比转置回传。平移部分仍用双线性权重；法向旋转产生力偶，厚度变化还涉及压陷方向的功。压陷为给定位移更新时，应显式核算该项输入功与耗散。此处须明确采用完整虚功配对。顶压硬盒几何增量应逐项严格为零。
5. 零柔度固定了几何约束，四点质量的转动惯量、逐块桌面支承与旧实体盒仍有差别。P1 必须在注入模板的路径上复用原门禁，用 4 材质 × 3 档压力验证双指零捕获／零张力，并显式核对四材质 3.0 s 悬停与脱手；现有 `side_metrics` 悬停门槛为 0.20 s。P2–P5 先验证模板本构和守恒关系。P2 的对角线中点错位衡量翘曲，若用其代表对折，需同时查看四块轨迹和支承时序。P6 复用远端已有 `probe_actions` 与 CUDA Graph 计时口径，覆盖模板状态的重置、图回放与逐步有限性检查。原 8,192 路的 1.119 s、进程显存采样 1,870 MiB 和 PyTorch 预留峰值 2,002 MiB 分别比较。YAM 适配器在计算反力时也重建刚性挂点，接入该平台时应复用模板真实挂点。

依据为 [刚体框架](../src/low_order_parcel/macro_parcel_sim.py)、[冻结代理](../src/low_order_parcel/lumped_contact_proxy.py)、[门禁 A 判据](../src/low_order_parcel/verify_stage2_alignment.py) 及 [Stage 2 冻结报告](../docs/STAGE2_FREEZE_REPORT.md)。

## Step 2 · 四块模板与 P2–P5

本轮在 `ubun` 使用指定项目解释器，从本机已有 PyTorch 环境补充模块搜索路径。P2 以两种纺织预设各运行一次 25 mm 微提拉与 150 ms 停留，检查四块依次离台、起抬张力连续、停留张力接近持平、离台后折弯角超过 0.3 rad。P3 按一次压入到触底再放开检查凸增反力、指数回弹和几何余料上限。P4 对偏心松散预设运行一次倾斜与悬空平衡试验，检查死区、止挡冲量与转动载荷，并核对平衡载荷。P5 用固定种子和极端参数锚点在 RTX 5090 跑 32 路各 100,000 步，逐步累计有限性、压陷上下界与桌面不穿透。GPU 正式运行持共享锁。通过后接入冻结代理。失败时记录工况并修正后重跑。

本轮复测通过。复用 `/home/ubuntu/Dev/Envs/isaaclab/lib/python3.12/site-packages` 的 PyTorch 2.11.0+cu130，入口仍为指定解释器。首轮 P2 单角载荷允许自由倾转，衣物折弯角只有 0.013 rad。改为两垫沿对角抬升，并在给定区间内选用 0.50 kg、0.03 m/N 作为折叠衣物预设。0.70 s 起抬后静置 0.15 s，保留原折角和停留验收。P4 首次将投影后的距离当成止挡发生条件，复测改用未约束预测位置判定首次触及死区边界。

| 检查 | 结果 |
| --- | --- |
| P2 | 四块依次达到离台阈值。衣物／被子停留拉力保持率 1.023／1.057，悬空折角 0.406／0.593 rad。 |
| P3 | 20% 至 100% 压陷的准静态力为 0.706、2.117、4.939、9.878、17.640 N。150 ms 后压陷为峰值的 0.829。中等压入使 PE 代理捕获，硬盒对照零捕获。超深压入触发硬核反力，进给增量封顶 20 mm。 |
| P4 | 倾斜预设的对角止挡在 0.140 s 启动，最大传递冲量 0.004834 N·s。逐步通过整体线动量检查，独立的悬空平衡样本为 5.3955 N。模板阶段由外部施加提拉力。 |
| P5 | RTX 5090，32 × 100,000 步，float32。有限性、压陷边界及逐块桌面净空通过，最小净空 0 m，最大内部半径 0.5259 m。用时 61.922 s（单测整程）。 |
| 挂点虚功 | 浮点 64 位下，双指到四块的合力、表面偏移力矩，以及包括压陷功在内的瞬时功率均匹配。 |

```sh
PYTHONPATH=/home/ubuntu/Dev/Envs/isaaclab/lib/python3.12/site-packages OMP_NUM_THREADS=1 .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_unified_payload.py -v -k p2 -k p3 -k p4 -k test_top
flock /tmp/claude-gpu.lock env PYTHONPATH=/home/ubuntu/Dev/Envs/isaaclab/lib/python3.12/site-packages OMP_NUM_THREADS=1 .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_unified_payload.py -v -k p5
```

## Step 3 · 接入与最终回归

最近验证为 2026-11-21 15:32（台北），机器 `ubun`。项目解释器从已有环境加载 PyTorch 2.11.0+cu130，未升级共享环境。模板的 P1–P6 与挂点虚功检查均通过，整包原有 20 项通过。冻结代理文件与 `8184257e` 的 Git 差异为空。

首次 P1 的局部高度计算误触发气泡膜加载的触底反力。改用随板面旋转的法向后，六类门禁全通过。开环 P6 只得到 78 路达标悬停，改用受限的夹爪位置反馈后达到 8,003/8,192 路。原悬停判据一直保留。最终总吞吐 11,618、有效吞吐 11,350 仿真秒/墙钟秒，3.363 s 墙钟，采样显存 1,810 MiB，PyTorch 预留峰值 1,968 MiB。仍有 189 路悬停未达标。

```sh
PYTHONPATH=/home/ubuntu/Dev/Envs/isaaclab/lib/python3.12/site-packages OMP_NUM_THREADS=1 .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_unified_payload.py -v -k p1 -k p2 -k p3 -k p4 -k test_top
flock /tmp/claude-gpu.lock env PYTHONPATH=/home/ubuntu/Dev/Envs/isaaclab/lib/python3.12/site-packages OMP_NUM_THREADS=1 .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_unified_payload.py -v -k p5 -k p6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-physical-parcel/bin/python -B -m unittest discover -s test/bag_model -p test_physical_parcel.py -v
git diff --exit-code 8184257e -- src/low_order_parcel/lumped_contact_proxy.py
git diff --check
```

[最终 CPU 5 项 63.363 s](../test/runs/2026-11-21/bag/unified-payload-proxy-r4/cpu.log) · [最终 GPU 2 项 65.555 s](../test/runs/2026-11-21/bag/unified-payload-proxy-r4/cuda.log) · [整包 20 项 8.829 s](../test/runs/2026-11-21/bag/unified-payload-proxy-r3/physical-parcel.log) · [完整验收](../docs/PAYLOAD_TEMPLATE_REPORT.md)

## Stage 2.5 · 同袋分岔与侧推回归

最近验证为 2026-11-21 15:42（台北），机器 `ubun`。CPU 新增和原模板检查 7 项通过，用时 73.472 s。GPU 两项通过，用时 66.806 s。原 Stage 2 七项用时 228.137 s，原整包 20 项用时 8.737 s，均通过。旧刚体 CUDA Graph 回合 8,192 路全部达标，用时 1.087 s。

同袋测试中，硬书顶抓零捕获且零张力，侧抓悬停超过 3 s。衣物与枕头顶抓双侧捕获，离台 133.4 和 133.6 mm。轻包正撞侧壁后平移 50.32 mm，最大净进给 0.389 mm，零捕获。2 ms 异质 8,192 路新版批次用时 3.637 s，采样显存 1,812 MiB，8,003 路满足原悬停要求。GPU 压测和旧刚体 GPU 校验均使用共享锁。

沿用上节 CPU 与 GPU 命令及其环境。CPU 模板命令中的 `-k p2` 和 `-k p3` 同时包含了新增分岔和侧推检查。旧代理独立命令如下。

```sh
PYTHONPATH=/home/ubuntu/Dev/Envs/isaaclab/lib/python3.12/site-packages OMP_NUM_THREADS=1 .venv-physical-parcel/bin/python -B -m unittest discover -s test/low_order_parcel -p test_stage2_proxy.py -v
flock /tmp/claude-gpu.lock env PYTHONPATH=/home/ubuntu/Dev/Envs/isaaclab/lib/python3.12/site-packages OMP_NUM_THREADS=1 .venv-physical-parcel/bin/python -B test/low_order_parcel/benchmark_stage2_cuda.py --worlds 8192 --output test/runs/2026-11-21/bag/unified-payload-proxy-r5/stage2-cuda.json
```

[Stage 2.5 验收](../docs/PAYLOAD_TEMPLATE_REPORT.md#stage-25-同袋异物结果) · [CPU](../test/runs/2026-11-21/bag/unified-payload-proxy-r5/cpu.log) · [GPU](../test/runs/2026-11-21/bag/unified-payload-proxy-r5/cuda.log) · [旧代理](../test/runs/2026-11-21/bag/unified-payload-proxy-r5/stage2-proxy.log) · [整包](../test/runs/2026-11-21/bag/unified-payload-proxy-r5/physical-parcel.log)

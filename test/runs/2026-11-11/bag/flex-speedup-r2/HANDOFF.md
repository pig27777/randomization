# flex 提速第 2 轮（F8–F13）交接

交接时间：2026-11-11 21:35 CST，机器 `ubun`。截止：2026-11-13 晚。

**21:40 用户要求停止：F8、F13 两个后台队列已全部终止，GPU 上无本任务进程。** 下文“进行中”的运行都已中断，接手时需要重新排队启动；已完成的结果只有 `f8/base_rep1_*` 和 `f13/bendable_only04.json`。重启前先删掉或改名中断留下的残缺文件（目前没有：rep2 和 compressible_only04 中断时尚未写文件），队列脚本会拒绝覆盖已有输出，需要把已完成的配置从脚本里去掉。
原任务全文在同目录的 [PROMPT.md](PROMPT.md)，计划与标准在 [test/bag_model/REPORT.md](../../../../bag_model/REPORT.md) 的“flex 提速第 2 轮计划（F8–F12）”一节（不得修改其中的标准）。

## 1. 已确认的情况

- 开始前已检查 `git log`、`test/runs/2026-11-10/`、`test/runs/2026-11-11/`、`log/progress.md` 和 `nvidia-smi`，没有其他会话在做 F8–F12。工作区里有别人未提交的改动（`contents/HANDOFF.md`、`log/*`、`paper/`、`test/bag_model/README.md`、`side_pinch_*`、`four_part_pack_demo.py` 等）：不要动、不要提交。
- **F9 受阻**：本机只有 MuJoCo/MJWarp 3.17（`/home/ubuntu/Dev/Envs/mjwarp`），全盘找不到 4.0 安装。按约束不装、不升级共享环境。需要 `envs/continuum-requirements.txt` 的独立环境（据记录在另一台机器）。F10 计划写的是“4.0 discrete 积分器下”加大步长，只能在 3.17 下做可做的部分，并写明版本差异。
- 第 3 轮 GPU 数据（`bag_sim/outputs/round3_flex/gpu_base_{64,256,1024}.jsonl`）里，穿透超标只出现在 **centre 输入**；empty_end、corner 三档都是 0 mm（这两类全部夹取失败，没进入提起和搬运）。centre 的 1024 环境三个种子分别是 15.52、17.47、5.26 mm。各输入结束时的 CPU 复核穿透都是 0，说明是过程中的瞬时穿透。
- CPU 基准（`base_behavior.jsonl`）：centre 3/3 成功、其余 6 例夹取失败，穿透 0；最大约束行 840、接触 280。批量入口沿用约束上限 1566 行、接触 522（CPU 最大值 × 1.5）。

## 2. F8 进展（前提项，定位前不计吞吐）

**探索得到的线索（未排队，不作结论）**：
- 同一输入的 8 个环境，240 步后状态已经不一致（`identical_worlds=false`）。
- MJWarp 的 `opt.deterministic`：
  - 开 `CONTACTS` 或 `CONTACTS+CONSTRAINT`，状态仍不一致。
  - 开 `ATOMICS`（及 `ALL`）报错：`WarpDeterminismError`，稀疏 H 需要约 23.5 亿个原子更新槽，超过 2^31 上限。也就是说，这个模型在 3.17 上做不到完全确定。
- 由此得到的假设（待证）：浮点原子加法的顺序不确定，使各环境轨迹分叉。穿透有重尾，最大值随环境数（样本数）增大而增大，属于极值效应，不是环境之间的耦合 bug。要证实，需要看“超过 1 mm 的环境比例”在 64/256/1024 档是否大致不变，以及峰值出现在哪个阶段、是哪种穿透。
- 读过 MJWarp 3.17 的 flex 碰撞代码：盒子与三角形是 `mjraw_BoxTriangle` 的移植；候选、宽相溢出都会置溢出位（第 3 轮全部为 0）。没有发现静默丢弃接触的路径，但没有逐行核对。

**正式运行（排 GPU 锁；21:40 已中断）**：队列脚本 `f8_queue.sh`，输出在 `f8/`，日志 `f8/queue.log`。
- centre 种子 1（第 3 轮 1024 档最差的输入）：1 环境重复 3 次，然后 64、256、1024 环境各一次。
- 已完成 `base_rep1`（1 环境）：成功，穿透 0 mm，47.3 ms/步，平均迭代 5.75，墙钟 504 s。rep2 中断，未完成。
- 预计墙钟：1 环境约 8 分钟 ×3，64/256/1024 约 14/20/38 分钟（按第 3 轮推算）。
- 每次运行写 `<tag>_n<N>.jsonl` 和逐环境的 `centre_s1_n<N>_<tag>.npz`，内容包括：
  - `outcome`；
  - `peak_mm[:,0]`：膜顶点进入盒体的深度；
  - `peak_mm[:,1]`：盒角出袋到袋面的距离；
  - `when`：两种峰值的时刻、阶段号，以及出袋的是哪个角；
  - `state`。

**F8 下一步**：
1. 等 F8 队列跑完，比较三档的逐环境穿透分布：p50/p90/p99、超过 1 mm 的比例、峰值所处阶段和类型。
2. 3 次 1 环境重复是否逐位相同（`state` 是否相同；rep1 的 `identical_worlds` 只对 1 个环境有意义）。
3. 定位到原因后再找修复，例如：碰撞半径和接触余量（float32）、接触 `solref`/`solimp`（见 `mj_loose._contact_solref`，按顶点质量和步长算）、求解迭代或容差、指尖速度等。目标是 1024 环境最大穿透 ≤1 mm；修复要改模型参数时，同时用 CPU F3 做行为复核。
4. 若确认是重尾而且修不到 ≤1 mm，如实写“未通过”，不改 1 mm 标准；需要改标准的话，写理由留给用户决定。

## 3. F13 进展（薄膜 + v1.3 软内容物）

- 已查明步长冲突：v1.3 的 bushing 力在 `Sim.step()` 里由 CPU 写入（`mj_step1` → `Bushings.apply` → `mj_step2`），步长 `BUSHING_DT = 0.4 ms`。袋子和内容物在同一个 MjModel 里，只能共用一个步长，所以合在一起只能按 0.4 ms 走（还要再核对一次内容物线是否在更大步长下验证过；目前没有看到）。
- GPU 侧：内容物线的 v1.3 GPU 候选 `contents/segment/mjwarp_v13.py`（停在 G3，94.44% < 95%，HANDOFF 明确写了不能当作已验收后端）。它的 `Batch(specs)` 自己构建内容物场景，接不进 flex 袋。要合并，需要内容物线提供一个能接受外部 MjModel（含 flex 袋）的 GPU 步进接口，也就是把 bushing 和捏取 kernel 与场景构建解耦。只写需求，**不改 `contents/`**。另外 `bag_sim/soft_content_backend.py` 指向的 `outputs/soft_joint_20261108/` 在本机不存在。
- 正式 CPU 测量（排 GPU 锁，单线程；21:40 已中断）：队列 `f13_cpu_queue.sh`，输出在 `f13/`，入口 `test/bag_model/flex_soft_cost.py`。
  - 已完成 `bendable_only04`（易弯单独静置 2 s）：0.305 ms/步，其中 bushing 0.091 ms；约束 216 行，接触 72，迭代 1.07。
  - 未完成（中断）：`compressible_only04`，然后是 `hard08`、`hard04`、`bendable04`、`compressible04`。袋内配置是一次 F3 centre 种子 0 的完整侧夹（evaluate），软内容物每次可能 20–40 分钟。
  - 易压锚点（350×250×120 mm）装不进第 3 轮基准袋，`compressible04` 改用 `fill=0.75` 的放大袋，报告里要注明。
- 参考（第 3 轮，未排队，不作吞吐）：`bag_sim/outputs/round3_flex/soft_reference.jsonl` 是易弯在 fill 0.75 袋中的 R3，0.4 ms，74.6 ms/步，约束 2682 行，接触 894，迭代 7.3，最终掉落，穿透 5.08 mm。
- 合并后 GPU 吞吐无法实测（缺可接入的后端）。可以做一个只算原生物理的下限：flex 袋 + 软内容物在 MJWarp 上用 `mjw.step` 不加 bushing 力，测单步耗时和约束行数，并明确标注“不含 bushing、行为无效、只是成本下限”。是否做由你判断。

## 4. F12 进展（空袋端聚拢捏取）

- 新增 `bag_sim/film_gather_pinch.py::FilmGatherPinch`（继承 F7 的 `FilmPinch`）：
  - 捏住条件：两指尖都接触上层膜，且聚拢量 ≥ `GATHER_MIN`（8 mm，取自用户给四部分演示的初值；不是验收标准，正式值待真机结构检查，报告里要写明）。
  - 承载判定去掉把膜压向桌面的竖直分量（框架侧反力 z>0）。
- 聚拢量的量法改过一次：
  - 最初用指间带内上层膜的“弧长减弦长”。在 35 mm 网格下，指间带内只有 1–2 个顶点，量出来恒为 0，已弃用。
  - 现在的量法：压完时记下两指尖下方的上层膜材料点（三角形重心坐标），看合拢结束时二者水平距离缩短了多少。
- `test/bag_model/film_grasp.py` 加了可选参数 `binding_cls`。弹簧启用的判定用 `press_eligible`（存在时），F7 原行为不变；结果里多一个 `gather_mm` 字段。
- 入口 `test/bag_model/film_gather_grid.py`：F7 同一网格的空袋端 108 次，加 3 次低承载对照，28 进程，正式运行要排锁。
- **探索结果（不作结论）**：空袋端 w60/d12/F20/系数 1/种子 0，聚拢量只有 **0.31 mm**，捏取失败。探查脚本在 scratchpad（已不重要）。接触记录显示：
  - 指尖与膜的接触力只有 0.02–0.27 N；
  - 合拢后段左指不再接触上层膜；
  - 夹持力读数很小（膜很轻，指尖在膜上打滑）。

  这可能是 flex 薄膜在这个协议下的真实行为（平铺膜被夹不起来），也可能是量法或接触参数的问题，需要再查一层，例如合拢方向的摩擦、指尖与膜的接触刚度。如果查明膜确实聚不起来，F12 就如实记“不通过”。

## 5. 新增 / 改动的文件（都未提交）

| 文件 | 说明 |
|---|---|
| `test/bag_model/flex_round2.py` | F8/F10/F11 的 GPU 批量入口：流程照抄 `flex_batch.py`，步长、迭代、确定性、环境数可设，逐环境记穿透、阶段和类型；`--max-steps` 只用于入口检查 |
| `test/bag_model/flex_soft_cost.py` | F13 CPU 单步成本，包装计时 bushing |
| `bag_sim/film_gather_pinch.py` | F12 规则 |
| `test/bag_model/film_gather_grid.py` | F12 网格 |
| `test/bag_model/film_grasp.py` | 只加 `binding_cls`、`press_eligible` 钩子和 `gather_mm` 输出 |
| `test/runs/2026-11-11/bag/flex-speedup-r2/` | 本轮运行目录：`f8_queue.sh`、`f13_cpu_queue.sh`、`hostname.txt`、`f8/`、`f13/` |

还没做：`test/bag_model/README.md` 入口表（该文件里有别人未提交的改动，只追加行，不动别人的内容）、`REPORT.md` 结果节、`test/runs/2026-11-11/README.md` 索引、`log/progress.md`、`reports/flex-speedup-round2-test.md`。

## 6. 剩余工作建议顺序

1. F8 收尾：等队列；分析分布；定位原因；修复；1024 环境复测穿透 ≤1 mm。必要时加跑 centre 种子 0、2。
2. F10（3.17 下）：在 F8 修复后的配置上，步长 0.8 → 1.6 → 3.2 ms。`flex_round2.py --dt` 已支持，10 ms 检查间隔自动换算。CPU F3 行为复核用 `round3_flex.py` 的 evaluate，但它的步长固定 0.8 ms，需要加参数或另写小入口，而且不能覆盖 `bag_sim/outputs/` 里的旧文件。
3. F11：4096 环境；整步 CUDA graph（`flex_round2` 已经是整步 graph，主机每 2×检查间隔同步一次）；迭代上限和线搜索下调；缓存与提前结束保留。标准：溢出 0，与 CPU 类别一致率 ≥90%。注意显存（第 3 轮 1024 档约 8.2 GB）。
4. F12 收尾：见第 4 节，查清后正式跑 108+3 次。
5. 叠加通过项，做完整的 F3 行为检查和 F7 顶面抓重跑；然后 F13 的软内容物配置。
6. 写结果：REPORT 新增“F8–F13 结果”小节，加当日 README、progress 3–5 行、任务检查报告。最终回复按三档给结论（≥10 次/s 且行为通过 / 1–10 次/s / <1 次/s 或穿透超标），并列出受阻项（F9 和 F13 的 GPU 吞吐）。

## 7. 约定提醒

- 正式运行一律 `flock /tmp/claude-gpu.lock <命令>`；不排队的探索运行不作结论，耗时也不作吞吐数据。速度只和 `ubun` 上的数据比。
- `contents/` 只读；冻结 v1.3 从 `bag_sim/frozen_content.py` 校验后导入。
- 不改验收标准、基准和捏取阈值；不提交、不推送；不覆盖 `bag_sim/outputs/` 里的已有证据。
- 解释器统一用 `/home/ubuntu/Dev/Envs/mjwarp/bin/python -B`，在仓库根目录运行；CPU 测速加 `OMP_NUM_THREADS=1`。

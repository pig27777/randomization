# 内容物 v1.3 GPU 后端检查

最近测试：2026-11-08（Asia/Taipei）。**结论：硬物R3探索排除本例读数错误，确认首接触/合拢停止与保持增益链路有实际夹紧差异，按用户要求停止。G1已通过；已批准的R1y四次正式补跑及G3–G5未进入。GPU平台指定MJWarp，但当前后端尚未供训练；CPU v1.3保留参考。**

以下11月6日结果保留为历史；本次续检见文末，未将探索结果改算正式验收。

## 范围与方法

- 新代码：`contents/segment/warp_v13.py`（GPU 算子）、`mjwarp_v13.py`（异 κ 分桶的批量执行器，协议 v2.2 向量化）、`validate_warp_v13.py`（G1–G5 入口）。
- 复用 S2 已通过的压密 weld kernel 和连接力数学；v1.2 的 `warp_*` 文件不动。
- 结果在 `contents/segment/results/warp_backend_v1_3/`；正式运行均持 `flock /tmp/claude-gpu.lock`。

## 结果

| 项 | 命令 | 结果 |
|---|---|---|
| G1 | `validate_warp_v13.py g1 --run-id g1_formal` | 通过：2,700 多个同状态样本，连续量差 ≤8.9e-16，离散输出全部相同；主机低通与弹簧差 0 |
| G2 | `validate_warp_v13.py g2 --run-id g2_formal --workers 5` | **未通过**：易弯 R1y 刚度 0.50115 → 0.48363（−3.50%）。其余软物 R1/R2 与 R4 ≤1.52%，离散读数在分辨率内；无仿真失败 |
| G2 动态层（记录） | 同上 | 软物 R3 在线内；硬物 R3 保持期夹持力 18.92 → 7.42 N |
| 定位（探索） | 同一 GPU 路径重跑易弯 R1y 两次 | −1.10%、−0.46%；连同上一轮 −0.57%，GPU 运行间极差约 3%，CPU 确定 |
| G3–G5 | — | 未运行 |

## 11月6日遗留（历史）

- 需要用户和总体判断会话决定：κ 响应层的 GPU 判定是否改为多次运行取中位数或均值，或采用其他判法。
- 硬物 GPU/CPU 夹持力差异在 G3 会直接受到检验。
- G2 运行期间追加了 G3/G4 代码（G2 用到的函数未变），G2 元数据中该文件 SHA 与当前文件不一致。
- 详见[实验报告](../test/content_model/REPORT.md)“v1.3 GPU 后端”。

## 11月8日审阅后续检

用户已批准κ响应超线项正式补至5次，中位差≤2%且每次≤5%；探索不计入。Isaac Lab不再作为内容物候选。本轮按顺序先诊断硬物R3，遇真实物理差异即停，不继续补R1y和G3–G5。冻结归档/本构/阈值不改。

| 范围 / 方法 | 命令（项目根目录） | 结果 |
|---|---|---|
| GPU短试、只验证观察入口 | `flock /tmp/claude-gpu.lock /home/ubuntu/Dev/Envs/mjwarp/bin/python -B contents/segment/diagnose_r3_v13.py --backend gpu --max-time .02 --output contents/segment/results/warp_backend_v1_3/r3_diagnostic_20261108/smoke` | exit0，明确`incomplete`，20ms探索；未当成R3通过 |
| 原R3控制器的完整CPU/GPU观察 | 同一持锁入口，`--output contents/segment/results/warp_backend_v1_3/r3_diagnostic_20261108/closed_loop` | CPU/GPU各532个状态，无确认仿真失败；保持期力18.924050/7.420248N。观察墙钟0.967/53.489s，不是吞吐 |
| 接触独立解码、逐步比较（只读） | `.../mjwarp/bin/python -B contents/segment/analyze_r3_v13.py contents/segment/results/warp_backend_v1_3/r3_diagnostic_20261108/closed_loop` | 两端椭圆锥、保持期64接触；35,760个GPU导出接触与原生6维力解码全部匹配，最大差0。首活动接触分歧t=0.3804s，CPU32、GPU0 |
| 冻结完整性与运行源码 | SHA256核对v1.3归档及model/mj_adapter/protocols/metrics/pinch/top_grasp，核对G2依赖和观察器前后指纹 | 六项冻结源码、原归档一致；原G2依赖一致；运行期间未改依赖。只新增诊断/离线分析源码 |

**定位**：CPU/GPU判定停止于0.3816/0.3812s；共同挤压5.551/0µm，使原`Closer.hold`保持增益5.551e-6/1e-8m/(N·s)，相差约555倍。瞬态回落后CPU继续补力，GPU几乎不补；搬运前总接触压入102.369/40.293µm，净开口相差约62µm。硬物本体仍刚性，此量是指垫接触压入。未发现力读取、接触数或金字塔解码错误；首接触数值来源尚未分离，不能直接归因于float32。按用户要求到此停止，没有修改控制器、未重跑“修好读数后”的正式硬物R3，也未进入后续正式门槛。

**限制/下一步**：本轮只有一次完整探索对照，不作新G2统计或训练性能结论。R1y补4次正式、G3任务类别、G4随机稳定、G5吞吐/峰值显存/时间拆分/solver_niter性能分布及所有优化均未执行，当前不能宣称GPU能用于训练。建议先审阅零共同挤压时的保持增益边界并隔离首接触差异，再恢复既定顺序。I-37已记入ideas，v1.4再处理；HANDOFF已明确候选入口及“尚未就绪”，未把它交成已验收软GPU接口。未安装/升级包、未改共享库及其他线代码。

证据：[诊断指纹汇总](../contents/segment/results/warp_backend_v1_3/r3_diagnostic_20261108/closed_loop/summary.json)、[离线分析](../contents/segment/results/warp_backend_v1_3/r3_diagnostic_20261108/closed_loop/analysis.json)、[逐时刻比较](../contents/segment/results/warp_backend_v1_3/r3_diagnostic_20261108/closed_loop/comparison.csv)。原始逐接触数据`cpu.json`/`gpu.json`及短试、日志保留在上述目录，无覆盖旧正式结果。

结束静态检查通过：9项相关源码AST、6个相关JSON严格解析、6项冻结CPU源码与原归档SHA、诊断前后源码及分析输入哈希、6条新增结果链接；无额外物理运行，无本线实验残留进程。[审计记录](../contents/segment/results/warp_backend_v1_3/r3_diagnostic_20261108/audit.json)。

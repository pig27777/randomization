# Policy V3 C2 r2 · 50 µs MuJoCo CPU Flex

## 本轮计划

主机 `ubun`，MuJoCo 4.0.1、PyTorch 2.15.0+cpu。复用 r1 的 450 个 C1 回合、既有 GroupSplit，以及独立闭环种子和试探后积分检查点。有效训练／验证／测试为 265／56／55，正样本为 87／17／12。三组模型只使用原有效标签。V0_ONLY 保留初始几何和公开 Try，去力对照屏蔽所有力通道；Blind 屏蔽全部历史。

在冻结权重上检查三个匹配 RGB 状态。对同一候选，三态最大分数差须至少为 0.05，Top-1 须涉及至少两个候选。通过后评测 Left／Center／Right 各 15 个独立状态，每个策略每状态作一次决定。同一检查点和候选共用物理结果。缓存缺失用 22 个 MuJoCo CPU 进程执行，OpenMP、OpenBLAS、MKL 各进程分别限 1 线程。沿用原碰撞掩码、有效性审计和成功判据。留出 Top-1 沿用 r1 每划分六个状态的全候选物理分支。离线失败时先记录梯度和嵌入诊断；闭环按实际结果报告，图中呈现有效分母。

训练与评测分别运行 `bash test/runs/2026-11-26/cpu-flex-policy-v3-c2-fix-r2/run-r2.sh train` 和同一路径的 `evaluate`。输入为 [r1 原始 C1 包](../cpu-flex-policy-v3-c1c2-r1/c1_raw_packets.tar.gz)、[既有划分](../cpu-flex-policy-v3-c1c2-r1/c1_dataset_summary.json) 和 [r1 物理分支包](../cpu-flex-policy-v3-c1c2-r1/physical_eval_packets.tar.gz)。

## 冻结配置

三层残差因果 1D TCN 的宽度为 24，按公开 Try 的压入／扫掠时窗进行 masked mean/max pooling，输出 32 维。历史仅保留有效训练中的 17 个非零通道，用训练均值、标准差（下限 1e-6）与按动作拟合的平均轨迹做残差标准化。去力版余 13 个通道。编码由实际标准化残差与零残差的 TCN 表示差得到。候选连续特征采用训练集 Z-score，另保留公共候选身份和标准化 x 的平方。候选隐向量在同一公共池内居中，再经 FiLM 和线性读出。

有效训练集中，Center 的 g5、g10、g12 缺少有效标签。候选支持头从有效记录中的实际候选索引和已知公开提议分布估计相对支持。推理分数为条件成功概率乘相对支持（同池最大支持归一为 1）。成功监督只使用原始有效标签，候选 BCE 按训练频数的平方根倒数加权。相对支持为池内比较量。加权 BCE 与候选支持损失共用 FiLM 表示，支持项权重 1.5。三组采用 AdamW，编码器／评分头学习率 0.001／0.0003，权重衰减 0.01、头部 dropout 0.2，最多 100 epoch，耐心 18。最小验证 BCE 选权重，以 AUROC 处理平局。验证、测试 BCE 均用未加权的最终评分。

初版 FiLM 最大跨态分数差 0.075136，三个峰值仍为 g9。分窗、模板残差、残差卷积和候选居中等配置增加历史敏感度，部分高达 0.300，候选排序仍锁定。预定四组候选支持权重试验中，按顺序保留首个通过门禁的 24 通道／1.5 组合。其编码器与 FiLM 梯度范数为 0.669 和 0.262，平均逐维嵌入跨态方差为 0.180。匹配三例用于配置门禁；C1 测试标签和 45 个闭环结果与选择隔离。各轮验证分数、梯度和门禁结果见 [调参记录](offline_tuning.json)。

## 结果

### C2 离线评分

| 评分器 | 最佳 epoch | 验证 BCE | 验证 AUROC | 验证 Top-1 | 验证全池成功/有效/组 | 测试 BCE | 测试 AUROC | 测试 Top-1 | 测试全池成功/有效/组 | 最大跨态分数差 | 匹配峰值 L/C/R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Blind | 100 | 0.397 | 0.860 | 4/5 | 3/4 / 6 | 0.305 | 0.899 | 4/5 | 3/4 / 6 | 0.000000 | g5 / g5 / g5 |
| Informed w/o Effort | 27 | 0.370 | 0.903 | 5/5 | 4/5 / 6 | 0.322 | 0.909 | 5/5 | 4/4 / 6 | 0.429189 | g12 / g12 / g12 |
| Informed (Ours) | 27 | 0.404 | 0.900 | 5/5 | 5/6 / 6 | 0.337 | 0.888 | 5/5 | 5/5 / 6 | 0.275187 | g12 / g11 / g12 |

Top-1 在至少有一正一负的状态组内，从有物理标签的候选中比较最高评分。全池指标先选分数最高的候选，再记录实际标签和有效分母。每个划分有六个沿用 r1 的状态组。三个 320 × 240 初始 RGB 逐像素相同，V0、公开 Try 和候选坐标相同。最终权重保存后重载并再次通过门禁。

### 45 状态闭环

| 策略 | Left 成功/有效 | Center 成功/有效 | Right 成功/有效 | 总成功/有效 | 合并有效成功率 | 95% CI | 状态等权 | 全样本未知界限 | 无效/尝试 | 无效率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Best Fixed | 12/13 | 0/0 | 12/15 | 24/28 | 85.7% | [72.0%, 96.6%] | NA | [53.3%, 91.1%] | 17/45 | 37.8% |
| Blind | 14/14 | 0/0 | 13/15 | 27/29 | 93.1% | [82.8%, 100.0%] | NA | [60.0%, 95.6%] | 16/45 | 35.6% |
| Informed w/o Effort | 15/15 | 0/0 | 11/14 | 26/29 | 89.7% | [78.6%, 100.0%] | NA | [57.8%, 93.3%] | 16/45 | 35.6% |
| Informed (Ours) | 15/15 | 4/6 | 11/14 | 30/35 | 85.7% | [74.3%, 96.9%] | 81.7% | [66.7%, 88.9%] | 10/45 | 22.2% |

双方同时有效 28/45。Informed − Blind 为 -3.6 pp，配对 Bootstrap 95% CI [-11.1, +0.0] pp。状态等权差为 NA，95% CI NA。全 45 状态的配对未知结果界限为 [-28.9, +28.9] pp。20,000 次 Bootstrap 按隐藏状态分层，以共同初始态为配对单位。某一策略在某类状态没有任何有效回合时，其状态等权成功率留空。

r2 的 Blind 重训后仍在 45 个状态选 g5。下表列出两轮在相同种子上的分状态结果。

| 策略 | Left 成功/有效 | Right 成功/有效 | Center 成功/有效 | Center 无效数 |
| --- | --- | --- | --- | --- |
| r1 Blind | 14/14 | 13/15 | 0/0 | 15 |
| r1 Informed | 1/15 | 5/7 | 4/14 | 1 |
| r2 Blind | 14/14 | 13/15 | 0/0 | 15 |
| r2 Informed | 15/15 | 11/14 | 4/6 | 9 |

| Center 公开试探 | Informed 成功/有效 | 候选计数 |
| --- | --- | --- |
| Try_Left | 0/0 | g12×8 |
| Try_Scan | 4/6 | g11×7 |

| 无效原因 | Best Fixed | Blind | Informed w/o Effort | Informed (Ours) |
| --- | --- | --- | --- | --- |
| film_edge_ratio | 2 | 16 | 16 | 9 |
| film_inset_core | 0 | 0 | 0 | 1 |
| jaw_inset_core | 15 | 0 | 0 | 0 |

### 行为适应度

| 策略 | Left 选择数 | Center 选择数 | Right 选择数 | 选姿平均 x L/C/R (mm) | Left–Center TV | Left–Right TV | Center–Right TV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Best Fixed | g10×15 | g10×15 | g10×15 | 3.0 / 3.0 / 3.0 | 0.000 | 0.000 | 0.000 |
| Blind | g5×15 | g5×15 | g5×15 | 0.0 / 0.0 / 0.0 | 0.000 | 0.000 | 0.000 |
| Informed w/o Effort | g12×15 | g12×15 | g12×15 | -3.0 / -3.0 / -3.0 | 0.000 | 0.000 | 0.000 |
| Informed (Ours) | g12×15 | g11×7, g12×8 | g12×15 | -3.0 / 10.1 / -3.0 | 0.467 | 0.000 | 0.467 |

闭环共 45 个独立初始态，复用 90 个物理分支，新增 52 个物理分支。另复用 132 个留出排序分支，排序状态的原始候选标签从 C1 包读取。Best Fixed 沿用 r1 的训练划分状态等权下界规则，其中无效记为未知，三组 BCE 训练均严格排除无效样本。

### RSS 2027 Section 4

We trained a FiLM-conditioned causal temporal convolutional scorer on 265 physics-valid trials from the archived 50 µs MuJoCo CPU Flex collection. The scorer used initial geometry, the public probe command, and proprioception recorded during the probe. The score combined success conditional on a valid run with relative candidate support estimated from valid samples and the known randomized proposal. On three parcel states with pixel-identical initial RGB and the same command, the largest score change for a fixed grasp was 0.275. The selected grasps were g12, g11, g12 for Left, Center, and Right, respectively. The held-out test AUROC was 0.888 for Informed and 0.899 for Blind. We evaluated 45 separate initializations, 15 per hidden state, with the policies branching from identical post-probe physics checkpoints within each initialization. Across the Left, Center, and Right states, Informed achieved 15/15, 4/6, 11/14 successes per valid run and Blind achieved 14/14, 0/0, 13/15. Blind had 15 invalid Center runs. Informed's pairwise total-variation distance between hidden-state grasp distributions was 0.00–0.47. Across 28 jointly valid pairs, the paired success-rate difference (Informed minus Blind) was -3.6 pp (95% stratified paired bootstrap CI [-11.1, +0.0] pp). Treating invalid outcomes as unknown gave a full-cohort difference bounded by [-28.9, +28.9] pp. The full-cohort difference remains unresolved because these bounds include zero.

[矢量 PDF](fig_c2_score_shift_and_success.pdf) · [450 dpi PNG](fig_c2_score_shift_and_success.png) · [离线指标](c2_metrics.json) · [闭环指标](closed_loop_eval.json) · [新增物理分支](new_physical_branches.tar.gz) · [冻结核验](frozen-check.log) · [原单测](policy-v3-regression.log)

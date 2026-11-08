# 文献

按用途分组。⚠ 表示二手信息或尚未核实。

## 物流软包
- **Manyar, Ye, Mayya, Wang, Gupta. Real-to-Sim Parameter Learning for Deformable Packages… IDETC 2025**（Amazon/USC）。基于 Warp 的粒子系统，袋子是两层粒子网格组成的壳，三角形单元加弯曲单元；内容物是一个 160 mm 的刚性圆盘；吸盘用弹簧近似；用动作捕捉加贝叶斯优化标定；单场景实时。只做到刚性内容物。https://www.amazon.science/publications/real-to-sim-parameter-learning-for-deformable-packages-using-high-fidelity-simulators-for-robotic-manipulation
- 同组：Simulation-assisted learning for bin packing of deformable packages（MuJoCo，把仿真当作先验）。

## 平台和求解器
- **VBD**：Chen, Liu, Yang, Yuksel. SIGGRAPH 2024, TOG 43(4)。
- **AVBD**：Giles, Diaz, Yuksel. SIGGRAPH 2025, TOG 44(4)。`rigid_contact_k_start` 是这里接触罚刚度的起始值。https://graphics.cs.utah.edu/research/projects/avbd/
- **Newton 的耦合求解**：`SolverCoupled`（proxy / ADMM，从 1.4.0 起）；Isaac Lab 的 `coupled_mjwarp_vbd_manager`。https://isaac-sim.github.io/IsaacLab/main/source/concepts/coupled_solvers.html
- **C-IPC**：Li, Kaufman, Jiang. SIGGRAPH 2021。**StiffGIPC / libuipc**：Huang 等，arXiv 2411.06224，TOG 2025。

## 内容物力学
- **van Wyk 1946** 纤维集合体压缩：P ∝ E·(φ³ − φ₀³)；Toll 考虑取向后指数更高；文献中的指数为 3–15.5。原模型不含迟滞（Stanković 2008 拟合过迟滞）。
- **Matan, Williams, Witten, Nagel. Crumpling a Thin Sheet. PRL 88, 076101 (2002)**：恒载下对数蠕变（长达 3 周）；迟滞依赖历史最大载荷，预压“训练”后才可重复。arXiv cond-mat/0111095
- **泡沫**：hyperfoam（Ogden → Hill → Storakers）；Mullins 损伤因子 η（3 个参数）；加 Prony 级数的粘弹性；Polymers 2024 用超弹性层和弹塑性层交替建模 PE 包装泡沫。
- **四面体粘弹性**：Zeng, Scovazzi 等，IJNME 2017（Prony 级数，内部变量）；Hirsekorn 2011（Prony 级数蠕变的半解析积分，长步长下稳定）；GOOP 2008（嵌入式四面体有限元中的塑性）。
- **散件**：DEM 加质点–弹簧膜的颗粒阻塞夹爪（Götz 等，arXiv 2109.03356）；MPM 配 Drucker–Prager。
- ⚠ Hahn 等 SIGGRAPH Asia 2019：粘弹性参数的 real2sim 辨识（会议列表信息有混淆，需核实）。

## 机器人和数据驱动
- **PhysTwin**（ICCV 2025，arXiv 2503.17973）：从视频重建质点–弹簧模型，对象含毛绒玩具和快递包裹，基于 Warp。
- **DefGraspSim**（arXiv 2203.11274）：Isaac Gym 有限元，34 个物体，6800 次抓取，有内容物移位指标。
- 衣物：C-IPC 叠衣（Frontiers 2022）；MPM 翻衣（arXiv 2509.15953）；GarmentDynamics / RGBench（arXiv 2511.06434）。⚠ JKU 2026 年海报：揉团布料任务仿真成功率 0%，实物 94%。
- 袋子基准：DeformableRavens（arXiv 2012.03385）；Bag All You Need（arXiv 2210.09997）。⚠ 原记“arXiv 2403.12685：现有引擎都无法准确仿真塑料袋”编号错误：2403.12685 实为 Hannus 等的可变形物体模仿学习论文（2026-11-04 对 arXiv 元数据核对），“无法准确仿真塑料袋”的原始出处待查。AutoBag（下文）提到仿真训练的策略因视觉和动力学差距迁移困难，可能是相近的说法，未核对原句。

## 测量标准
- **Kawabata KES-FB3** 织物压缩：LC 线性度，WC 压缩功，WC′ 回复功，RC 厚度回复率。⚠ 公式以 Kato Tech 手册为准。
- ⚠ 泡沫：ASTM D3574 的压痕力挠度（IFD），未核实。

## RSS 2027 相关工作候选（2026-11-04 检索）

按 [论文计划](../paper/rss2027_plan.md) §5 的主题整理。✓ 表示标题、作者、年份已与 arXiv 元数据（export.arxiv.org API）或出版方页面核对；发表处注明来源。**写进论文前仍要读原文核对引用的具体说法**，这里的一句话概括来自摘要。

### 检索中发现的错误

- 论文计划 §5 把 **DROPO** 记为 arXiv 2303.04136，编号错误。DROPO 的正确出处：Tiboni, Arndt, Kyrki. *DROPO: Sim-to-Real Transfer with Offline Domain Randomization*. arXiv 2201.08434；Robotics and Autonomous Systems 166:104432 (2023)。2303.04136 实为同一作者的另一篇：Tiboni, Protopapa, Tommasi, Averta. *Domain Randomization for Robust, Affordable and Effective Closed-loop Control of Soft Robots*. IROS 2023（arXiv 注明）。后者对本文更相关（见下文）。两篇都可引用，需要总体判断会话改正计划里的编号。
- 上文“袋子基准”中 arXiv 2403.12685 的编号错误，已在原条目标注。

### 1. 参数随机化与物理一致参数化

- ✓ **XHugWBC**：Xue 等. *Scalable and General Whole-Body Control for Cross-Humanoid Locomotion*. arXiv 2602.05791 (2026)。本文模板 + κ + 重参数化的直接来源。
- ✓ **Rucker, Wensing. Smooth Parameterization of Rigid-Body Inertia**. IEEE RA-L 7(2):2771–2778 (2022), DOI 10.1109/LRA.2022.3144517。伪惯性矩阵的 log-Cholesky 参数化，把 R¹⁰ 光滑映射到全部物理一致的惯量。XHugWBC 那种“θ 空间任意扰动都合法”的做法的数学出处；本文的逐项映射可以和它对照着写。
- ✓ **Wensing, Kim, Slotine. Linear Matrix Inequalities for Physically-Consistent Inertial Parameter Identification**. arXiv 1701.04395；RA-L / IROS 2017（arXiv 注明）。伪惯性矩阵 J ≻ 0 的约束出处。
- ✓ **URMA**：Bohlinger 等. *One Policy to Run Them All*. arXiv 2409.06366；CoRL 2024, PMLR 270:3356–3378（Lund 大学记录）。16 种腿式机器人一个策略，训练时随机化形态和动力学。
- ✓ **GenLoco**：Feng 等. arXiv 2209.05309。四足的形态随机化，零样本迁移到多种四足。⚠ 发表处：项目页 BibTeX 写 “Proceedings of the 6th CoRL”（即 CoRL 2022, PMLR 205），页眉写 CoRL 2023，两者矛盾，引用前查 PMLR 205 目录。
- ✓ **Tobin 等. Domain Randomization and Generative Models for Robotic Grasping**. arXiv 1710.06425（arXiv 注明投稿 IROS 2018）。用数百万个程序生成的“不真实”物体训练抓取，在真实物体上成功率约 80%。与本文“物体分布本身就是随机化对象”同一思路，只是随机的是几何，不是力学响应。
- ✓ **EGAD!**：Morrison, Corke, Leitner. arXiv 2003.01314；RA-L 2020（arXiv 注明）。用进化算法生成 2000 多个按形状复杂度和抓取难度分布的物体，另给 49 个可 3D 打印的评估物体。可借鉴它“按难度轴覆盖、给可复现评估集”的写法，用于本文的 30 件评估物品。

### 2. 仿真到真实与域随机化

- ✓ **Tobin 等. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World**. arXiv 1703.06907；IROS 2017。视觉域随机化的开端。
- ✓ **Peng 等. Sim-to-Real Transfer of Robotic Control with Dynamics Randomization**. arXiv 1710.06537；ICRA 2018（项目页）。动力学参数随机化。
- ✓ **OpenAI 等. Solving Rubik's Cube with a Robot Hand**. arXiv 1910.07113 (2019)，只有 arXiv 版。自动域随机化（ADR）：随机范围随训练自动扩大。
- ✓ **BayRn**：Muratore, Eilers, Gienger, Peters. *Data-efficient Domain Randomization with Bayesian Optimization*. arXiv 2003.02471（v1 题为 *Bayesian Domain Randomization for Sim-to-Real Transfer*）；arXiv 注明 RA-L / ICRA，卷期未核对。用真机回报做贝叶斯优化，调整随机分布的参数。
- ✓ **SimOpt**：Chebotar 等. *Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience*. arXiv 1810.05687；ICRA 2019（NVIDIA 博客）。用少量真机轨迹调整随机分布。
- ✓ **Active Domain Randomization**：Mehta, Diaz, Golemo, Pal, Paull. arXiv 1904.04762；CoRL 2019, PMLR 100:1162–1176。在给定范围内主动挑选信息量大的参数。
- ✓ **DROPO**：见“检索中发现的错误”。用离线真机数据拟合参数分布。
- ✓ **Chen, Hu, Jin, Li, Wang. Understanding Domain Randomization for Sim-to-real Transfer**. arXiv 2110.03239；ICLR 2022（spotlight）。理论上证明温和条件下不需要真实样本也能迁移，并指出带记忆（依赖历史）的策略重要。可用来支撑本文“不逐件测量、只给范围”的做法，也提示策略输入里的夹紧过程读数（力、开口）有理论依据。
- ✓ **Muratore 等. Robot Learning from Randomized Simulations: A Review**. Frontiers in Robotics and AI 9:799893 (2022)，arXiv 2111.00956。综述，可以和已有的 Zhao 等综述（arXiv 2009.13303, IEEE SSCI 2020，已核对）一起引。

本文和这一组的区别：以上方法要么在固定范围里随机物理参数，要么用真机数据调范围（BayRn、SimOpt、DROPO）。本文随机的是“夹爪层面的响应量”，范围只由夹爪能力和常识界限给出，不用真机数据。

### 3. 柔性 / 易碎物体抓取与随机化

- ✓ **Matas, James, Davison. Sim-to-Real Reinforcement Learning for Deformable Object Manipulation**. CoRL 2018, PMLR 87:734–743；arXiv 1806.07851。布料任务全在仿真里加域随机化训练，真机零样本。可变形物体 sim-to-real 的早期代表。
- ✓ **DefGraspSim**：Huang 等. arXiv 2203.11274；RA-L 2022 可变形物体专刊（arXiv 注明）。Isaac Gym 有限元，平行夹爪抓三维可变形物体，指标包括最小防滑夹持力、挤压应力、形变。本文的“过度挤压”指标和夹持力上限可以参照它的定义。
- ✓ **DefGraspNets**：Huang 等. arXiv 2303.16138；ICRA 2023（arXiv 注明）。用图网络代替有限元预测应力和形变场，比 DefGraspSim 快约 1500 倍。说明高精度有限元太慢、需要替代，可作为本文降阶模型的动机之一。
- ✓ **Stress-guided RL**：Ikemura 等. *Sim-to-Real Gentle Manipulation of Deformable and Fragile Objects with Stress-Guided Reinforcement Learning*. arXiv 2510.25405（arXiv 注明“under review”，引用时查是否已发表）。训练时随机化杨氏模量、泊松比和摩擦，真机零样本。**和本文最接近的对比对象**：它用高精度软体仿真，本文主张降阶模型加响应空间随机化就够。
- ✓ **SimWeaver**：Hu 等. *Zero-Shot RGB Sim-to-Real for Deformable Manipulation*. arXiv 2606.15338 (2026)。
- ✓ **GenDOM**：Kuroki 等. *Generalizable One-shot Deformable Object Manipulation with Parameter-Aware Policy*. arXiv 2309.09051；ICRA 2024（arXiv 注明）。策略以可变形物体的物理参数为条件，测试时用一次真机演示估计参数。和 H4（真实物品的响应是否落在训练分布里）相关；也是“策略要不要显式输入响应参数”的一个参照。
- ✓ **Lach 等. Towards Transferring Tactile-based Continuous Force Control Policies from Simulation to Robot**. arXiv 2311.07245；IROS 2024, pp. 725–732（DFKI 记录）。平行夹爪的夹持力控制，MuJoCo 中训练，加域随机化和课程，真机零样本。同为 MuJoCo + 平行夹爪 + 夹持力，可作为“夹紧阶段”的相关工作。
- ✓ **DeformGen**：Lin 等. *Dynamics-Based Topology Augmentation for Deformable Manipulation Policy Learning*. arXiv 2606.25939 (2026)，只读了标题和作者，内容待读。

### 4. 降阶模型的仿真到真实

- ✓ **SoMo / SoMoGym**：Graule, McCarthy, Teeple, Werfel, Wood. *SoMoGym: A Toolkit for Developing and Evaluating Controllers and Reinforcement Learning Algorithms for Soft Robots*. IEEE RA-L 7(2):4071–4078 (2022)（哈佛实验室页面）。把连续体软机器人近似成“刚性段 + 带弹簧的关节”，跑在 PyBullet 上，用于 RL。**这是本文“分段刚体 + 弹性关节”最直接的先例**，区别是它建模的是机器人本身，本文建模的是被抓物体，参数也按响应定义。原始 SoMo 论文（Graule 等，IROS 2021 左右）待查。
- ✓ **Tiboni, Protopapa, Tommasi, Averta. Domain Randomization for Robust, Affordable and Effective Closed-loop Control of Soft Robots**. arXiv 2303.04136；IROS 2023（arXiv 注明）。软体机器人有限元仿真中的域随机化，并用它从数据推断随机分布。
- ✓ **Goury, Duriez. Fast, Generic, and Reliable Control and Simulation of Soft Robots Using Model Order Reduction**. IEEE T-RO 34(6):1565–1576 (2018), DOI 10.1109/TRO.2018.2861900。有限元模型降阶（POD + 超降阶），提速 100 倍以上。代表“从有限元降阶”的路线，与本文“直接用集总刚体模型”的路线对比。
- 已有：Schaff 等软体爬行机器人（TTIC 2023，论文计划 §5 已核对）。
- ✓ **Blanco-Mulero 等. Benchmarking the Sim-to-Real Gap in Cloth Manipulation**. IEEE RA-L 9(3):2981–2988 (2024), DOI 10.1109/LRA.2024.3360814；arXiv 2310.09543。用真实数据比较 MuJoCo、Bullet、Flex、SOFA 四个仿真器的布料仿真差距、耗时和稳定性。可用来说明“高精度仿真器之间、仿真与真实之间的差距本来就大”，支撑本文不追求形变精度的主张。

### 5. 可变形物体仿真与基准

- ✓ **SoftGym**：Lin, Wang, Olkin, Held. arXiv 2011.07215；CoRL 2020（arXiv 注明）。基于 FleX。
- ✓ **DaXBench**：Chen 等. arXiv 2210.13066；ICLR 2023 oral（arXiv 注明）。可微物理，基于 JAX。
- ✓ **DeformableRavens**：Seita 等. *Learning to Rearrange Deformable Cables, Fabrics, and Bags with Goal-Conditioned Transporter Networks*. arXiv 2012.03385；ICRA 2021（arXiv 注明 v3 为 ICRA 版）。含袋子任务。
- ✓ **Isaac Lab**：NVIDIA. *Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning*. arXiv 2511.04831 (2025)。
- ✓ **MuJoCo Playground**：Zakka 等. arXiv 2502.08844 (2025)。基于 MJX 的 GPU 训练与 sim-to-real 框架；项目页写获 RSS 2025 Outstanding Demo Paper。本文用 MJWarp 批量训练时可引它和 MuJoCo 原始论文（Todorov 等，IROS 2012，待核对）。
- 综述（✓ 均已核对出版方信息）：Yin, Varava, Kragic. *Modeling, learning, perception, and control methods for deformable object manipulation*. Science Robotics (2021), DOI 10.1126/scirobotics.abd8803；Sanchez 等. *Robotic manipulation and sensing of deformable objects in domestic and industrial applications: a survey*. IJRR 37(7):688–716 (2018)；Gu 等. arXiv 2312.10419 (2023)。

### 6. 袋子 / 包裹操作

- ✓ **AutoBag**：Chen 等. *Learning to Open Plastic Bags and Insert Objects*. arXiv 2210.17217；ICRA 2023（arXiv 注明）。
- ✓ **SLIP**：Chen 等. *Bagging by Learning to Singulate Layers Using Interactive Perception*. arXiv 2303.16898；IROS 2023（arXiv 注明）。
- ✓ **Bag All You Need**：Bahety 等. arXiv 2210.09997；IROS 2023, pp. 960–967（DBLP，经检索结果转述）。含异构物体装袋的仿真基准。
- ✓ **ShakingBot**：Gu, Zhang, He, Yu. arXiv 2304.04558；Robotica 42(3):775–791 (2024)。
- ✓ **Zhou 等. Bimanual Deformable Bag Manipulation Using a Structure-of-Interest Based Neural Dynamics Model**. arXiv 2401.11432（v1 题名用 “Latent Dynamics Model”）；IEEE/ASME T-Mech, DOI 10.1109/TMECH.2024.3485471（理大机构库记 30(5):3254–3265, 2025）。
- ✓ **Coinbot**：Gonnochenko 等. *Intelligent Robotic Coin Bag Manipulation Using Deep Reinforcement Learning And Machine Teaching*. arXiv 2012.01356 (2020)，发表处未查到。用 MuJoCo composite 模拟装满硬币、质心会移动的袋子，抓取会因接触不足或质心突变而掉落。**和本文“松装 / 活动质量”最接近的已有工作**。
- ✓ **Wang 等. Vision Based Picking System for Automatic Express Package Dispatching**. arXiv 1902.08951；IEEE RCAR 2019（arXiv 注明）。快递软包用两指夹爪，在深度图上采样对向抓取；避开会碰到袋内物品的抓取。物流软包平行夹爪抓取的早期工作。
- ✓ **ARMBench**：Mitash 等. arXiv 2303.16382；ICRA 2023（arXiv 注明）。Amazon 仓库实拍的 23.5 万次拣选、19 万种物品，含软包；任务是感知（分割、识别、缺陷检测），不涉及力学。可用来说明真实物流物品的多样性。
- 已有：Manyar 等 IDETC 2025 与同组装箱工作（见上文“物流软包”）。

### 检索后的判断（供总体判断会话参考）

1. 没有找到同时满足“降阶物体模型 + 按响应定义的随机化 + 平行夹爪真机零样本 + 物流软包”的工作。最接近的三条线：Stress-guided RL（高精度软体仿真 + 材料参数随机化）、SoMoGym（刚性段 + 弹簧关节，但对象是机器人）、Coinbot / Manyar（软包或装袋物体，但没有大规模随机化）。
2. 专门研究“松装时内容物在袋内滑动、影响抓取”的论文没有找到。这一点可以作为本文 H3 的新颖性，但写之前要再做一轮针对性检索（关键词：pouch contents、object-in-bag、sloshing contents grasp）。
3. 审稿人可能会问：为什么不用 BayRn / SimOpt / DROPO 那样用真机数据调范围？论文需要一句话说明，例如评估物品不可见，以及随机范围由夹爪能力给出就够。

### 待查

- SoMo 原始论文（Graule 等）的题名和发表处。
- MuJoCo（Todorov, Erez, Tassa, IROS 2012）和 MuJoCo Warp 的引用方式。
- 松装 / 袋内物体滑动的针对性检索（见“检索后的判断”第 2 条）。

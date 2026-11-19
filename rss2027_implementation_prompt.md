# System Role & Scope
你是一名专攻「GPU 并行物理仿真、柔性接触代理建模与 Zero-Shot Sim-to-Real 强化学习」的机器人研发工程师。
请严格遵循以下技术规格，基于 Stage 1 产出的「局部微切片（Micro-Patch）」离线仿真数据，依次实现：
1. **双层低阶物理模型（宏观多体 + 微观解析接触代理）**；
2. **Stage 1 微切片曲线 → 低阶参数包络的自动拟合与建库管线**；
3. **参数化柔顺原语 + 主动微提拉验证的非对称 Actor-Critic 训练接口**；
4. **物理包络 + 硬件 Datasheet 鸿沟 + 对抗角点的三层域随机化（DR）模块**。

**不可违反的全局约束**：
- **零 Real2Sim**：严禁依赖任何真机前置数据做 System ID 或微调；所有物理先验仅来自材料手册与 Stage 1 微切片仿真。
- **GPU 无分支向量化**：低阶接触代理的所有状态切换必须使用掩码（Mask）张量运算，禁止 Python 层 `if-else` 环境循环，所有物理量严格采用 **SI 国际单位制**（m, kg, s, N, N/m）。
- **拒绝两个极端**：禁止用黑盒神经网络 World Model 替代动力学；禁止用“距离达标即吸附”的二值规则替代聚拢与渐进滑移。

---

## Module 1: 核心低阶模型架构（Dual-Layer Lumped-Parameter Proxy）

整个软包装抓取仿真模型拆分为**宏观承载层**与**微观接触代理层**，目标稳定步长 $\Delta t \in [1.0, 2.0]\,\text{ms}$。

### 1.1 第一层：宏观承载骨架（Macro Multibody + Slack Tether）
- **包裹主体**：采用刚体引擎中的单刚体或 2–3 段带弹性铰链的分段刚体，负责表达包裹总质量 $m$、惯性张量、内容物质心偏心距 $\mathbf{r}_{\text{com}}$、桌面支承摩擦以及提拉后的重力倾覆与摆动。
- **单向非线性松弛柔索（Unilateral Slack Tether）**：连接「指尖捕获锚点 $\mathbf{p}_{\text{pinch}}$」与「包裹膜受力挂载点 $\mathbf{p}_{\text{bag}}$」。
  - 令当前锚点间距 $d_t = \|\mathbf{p}_{\text{pinch}} - \mathbf{p}_{\text{bag}}\|_2$，有效松弛长度为 $L_{\text{slack}, t}$。
  - 柔索张力标量严格单向（拉直前为零，不承受压力）：
    $$\delta_t = \max\!\left(0,\; d_t - L_{\text{slack}, t}\right)$$
    $$T_t = c_t \cdot \max\!\left(0,\; k \cdot \delta_t + c_d \cdot \dot{\delta}_t\right)$$
    其中 $c_t \in \{0, 1\}$ 为捕获激活标志，$k$ 为等效膜拉伸刚度，$c_d$ 为等效阻尼。
  - 张力沿单位方向向量 $\hat{\mathbf{u}}_t = (\mathbf{p}_{\text{pinch}} - \mathbf{p}_{\text{bag}}) / (d_t + \epsilon)$ 同时作用于包裹（产生提拉力与偏心力矩）和夹爪（产生腕部/指尖力反馈）。

### 1.2 第二层：微观解析接触代理（Micro Contact Interface Proxy）
为每个环境中的每个指尖接触区维护固定维度的连续隐状态张量 `contact_state`，形状为 `(N_envs, N_fingers, 5)`，包含：
1. `s` ($s_t \in [0, s_{\max}]$)：当前指尖下方已聚拢的局部膜余量（单位：m）；
2. `c` ($c_t \in \{0, 1\}$)：当前是否处于有效摩擦自锁捕获态；
3. `slip_cum`：累计滑移抽离长度（单位：m，直接累加到柔索有效松弛长度 $L_{\text{slack}, t}$ 上，产生真实的拉长与张力衰减）；
4. `contact_hold_time`：持续稳定法向接触时长（用于过滤高频抖动）；
5. `peak_tension`：当前捕获周期内的峰值张力记录。

**离散时间无分支步进方程**：
1. **余料聚拢项（Gathering）**：
   仅当法向压力达标（$F_n \ge F_{\min}$）、持续接触稳定（`contact_hold_time` $\ge \tau_{\text{gate}}$）且指尖存在单向内收聚拢速度（$v_{\text{gather}} > 0$）时累积：
   $$\dot{s}_{\text{gather}} = \alpha \cdot \max(0,\, v_{\text{gather}}) \cdot \mathbb{I}(F_n \ge F_{\min}) \cdot \mathbb{I}(\text{hold} \ge \tau_{\text{gate}}) \cdot \max\!\left(0,\; 1 - \frac{\sum_i s_{t,i}}{S_{\text{bag}}}\right)$$
2. **渐进滑移项（Progressive Slip）**：
   当处于捕获态且柔索张力 $T_t$ 超过静摩擦自锁上限 $\mu_{\text{eff}} F_n$ 时，不瞬间脱手，而是触发连续余量抽离：
   $$\dot{s}_{\text{slip}} = c_t \cdot \beta \cdot \max\!\left(0,\; T_t - \mu_{\text{eff}} F_n\right)$$
3. **状态更新与松弛联动**：
   $$s_{t+1} = \mathrm{clip}\!\left(s_t + \Delta t (\dot{s}_{\text{gather}} - \dot{s}_{\text{slip}}),\; 0,\; s_{\max}\right)$$
   $$L_{\text{slack}, t+1} = L_{\text{slack}, t} + \Delta t \cdot \dot{s}_{\text{slip}}$$
4. **迟滞捕获与释放判据（Hysteresis Lock/Release）**：
   - 激活条件（$0 \to 1$）：$s_{t+1} \ge s_{\text{crit}}$ 且 $F_n \ge F_{\text{lock}}$；
   - 断开条件（$1 \to 0$）：$s_{t+1} \le \epsilon_s$（余料抽空脱手）或 $F_n < F_{\text{release}}$（夹爪主动张开），其中 $F_{\text{release}} < F_{\text{lock}}$ 防止边界抖动。

### 1.3 强制防作弊物理不变量（Anti-Exploit Constraints）
代码必须内建以下三条不变量，并提供独立单元测试验证：
- **全局余料守恒**：任意时刻 $\sum_i s_{t,i} \le S_{\text{bag}}$，紧绷高填充包裹绝不能凭空聚拢出余料；
- **禁止高频微抖刷分**：`v_gather` 必须投影至严格指向夹持中心的内收方向，且受 `contact_hold_time` 门控与速度上限钳位；
- **滑移能量单调耗散**：滑移项仅允许增大 $L_{\text{slack}}$ 与消耗 $s_t$，严禁反向注入弹性势能导致力脉冲爆炸。

---

## Module 2: 如何根据 Stage 1 微切片数据搭建低阶模型

Stage 1 离线仿真针对 4 类典型材质（PE 薄膜、牛皮纸、气泡膜、软纸板）× 多档初始松弛度 × 多档下压力，输出标准离线数据集 `stage1_micropatch_data.parquet`（或 JSON）。每条轨迹包含两段实验曲线：
- **Curve A（下压聚拢扫描）**：给定法向力 $F_n$ 与指尖内收位移 $\Delta x(t)$，记录指尖褶皱区收集到的等效膜长度 $s_{\text{patch}}(t)$ 与切向阻力 $F_{\text{tan}}(t)$；
- **Curve B（拉拔滑移扫描）**：在捕获状态下施加递增拉拔力 $T(t)$，记录指尖与膜片的相对滑移位移 $u_{\text{slip}}(t)$ 与脱落时刻。

**按以下 4 步自动化脚本完成低阶模型构建与参数映射**：
1. **逐工况提取低阶标量参数**：
   - $F_{\min}$：从 Curve A 中找出首次产生有效褶皱堆积（$s_{\text{patch}} > 0$ 而非纯刚体打滑）的临界法向力；
   - $\alpha$（聚拢效率）：在 Curve A 未饱和线性段做最小二乘斜率拟合 $\alpha = \mathrm{d}s_{\text{patch}} / \mathrm{d}\Delta x$；
   - $s_{\text{crit}}$：从 Curve B 找出能维持非零有效拉拔载荷所需的最小初始聚拢长度；
   - $\mu_{\text{eff}}$（含几何互锁的等效静摩擦）：由 Curve B 起始滑移临界拉力 $T_{\text{crit}}$ 计算 $\mu_{\text{eff}} = T_{\text{crit}} / F_n$；
   - $\beta$（渐进滑移速率）：在 Curve B 超过 $T_{\text{crit}}$ 后的滑移段拟合 $\beta = \dot{u}_{\text{slip}} / (T - \mu_{\text{eff}} F_n)$；
   - $k$（等效膜刚度）：从 Curve B 拉直后、滑移前的线性弹性段斜率提取。
2. **构建「材料族参数包络 + 相关性矩阵」**：
   - 严禁只取单一均值点！按材质族汇总每个参数的实测区间 $[\theta_{\min}^{(m)}, \theta_{\max}^{(m)}]$，并计算跨参数经验协方差矩阵 $\Sigma_{\text{mat}}$（显式保留物理相关性：例如高弯曲刚度材质对应更高的 $F_{\min}$、更高的 $k$ 与更低的 $\alpha$）。
   - 导出配置文件 `proxy_material_envelopes.yaml`。
3. **单环境波形对齐校验**：
   - 将 `proxy_material_envelopes.yaml` 载入 Module 1 单环境原型，复放与 Stage 1 相同的动作指令，自动比对三种规范波形的误差：
     1. 空夹/低余料无张力曲线；
     2. 真捕获张力爬升与悬停曲线；
     3. 虚夹持/过载渐进滑脱曲线（重点核对张力建立滞后时间与临界滑移载荷误差）。
4. **架构冻结门禁**：
   - 仅当不修改方程结构、仅替换材质标量参数即可复现 4 类材质波形，且通过随机极值动作防作弊扫描后，方可将 Module 1 向量化并冻结进入下游训练。

---

## Module 3: 后续 Policy 的训练接口契约（Asymmetric RL + 状态机）

### 3.1 压缩动作空间：三阶段柔顺原语状态机
禁止直接输出高频 6-DoF 关节增量。封装标准 Gym/IsaacLab 风格环境 `SoftParcelGraspEnv`，单次抓取包含最多 $K_{\max}=3$ 轮尝试，每轮由以下状态机构成：
- **Phase A（接触与聚拢原语）**：Policy 输出 5 维连续归一化动作 $a_A \in [-1, 1]^5$，映射为物理参数交由底层阻抗控制器执行：
  1. `target_press_force` $F_{\text{press}}$（下压目标法向力）；
  2. `gather_stroke` $\Delta x_{\text{gather}}$（双指向内聚拢行程）；
  3. `pinch_force` $F_{\text{pinch}}$（闭合夹持法向力）；
  4. `offset_xy` $(\Delta x_0, \Delta y_0)$（相对包裹中心落点偏置）。
- **Phase B（主动微提拉验证，固定探测动作）**：夹爪闭合后自动执行小幅抬升（$\Delta z_{\text{micro}} = 0.025\,\text{m}$，持续 $0.4\,\text{s}$），期间按 $50\,\text{Hz}$ 采集长度为 $H=20$ 的历史观测缓冲区 `micro_lift_buffer`。
- **Phase C（闭环决策与补偿）**：Policy 读取 `micro_lift_buffer` 输出动作 $a_C$：
  - 离散/门控决策 `mode` $\in \{0:\text{继续正常提拉},\; 1:\text{增力慢速防滑提拉},\; 2:\text{判定虚夹持并松开重抓}\}$；
  - 连续修正量 $(\Delta F_{\text{pinch}},\, \Delta v_{\text{lift}})$（当 `mode == 1` 时生效）；若 `mode == 2` 则重置回 Phase A 进行下一轮修正聚拢。

### 3.2 非对称观测空间（Asymmetric Actor-Critic）
- **Actor 观测 `obs_actor`（纯真机可部署、强制去量纲化）**：
  形状 `(N_envs, H, D_actor)`，输入轻量级 **1D-TCN 或 GRU 历史编码器**，严禁依赖脆弱的绝对力阈值：
  1. 归一化张力变化斜率：$\dot{F}_z / (F_{\text{pinch,cmd}} + \epsilon)$；
  2. 微提拉窗口力保持率：$F_z(t) / (\max_{\tau \le t} F_z(\tau) + \epsilon)$；
  3. 相对力负载比：$F_z(t) / (F_{\text{pinch,cmd}} + \epsilon)$ 与切向/法向力比值 $F_{xy}(t) / (F_z(t) + \epsilon)$；
  4. 夹爪开合度 $w(t)$、开合速度 $\dot{w}(t)$ 及相对闭合比 $w(t) / w_{\max}$；
  5. 末端相对高度与速度 $(z_{\text{rel}}, v_z)$、当前阶段 one-hot、剩余重试次数 `retries_left`。
- **Critic 特权观测 `obs_critic`（仅仿真训练期使用）**：
  拼接 `obs_actor` 的编码特征 + 仿真内部精确特权向量 `priv_state`：
  - 接触代理隐状态：$[s_L, s_R, S_{\text{bag}}, c_t, \dot{s}_{\text{slip}}, T_t, L_{\text{slack}}]$；
  - 真实物理参数：$[\alpha, \mu_{\text{eff}}, \beta, k, s_{\text{crit}}, m, \mathbf{r}_{\text{com}}]$；
  - 真实硬件扰动参数：`[sensor_gain, sensor_bias, ctrl_delay_ms]`。

### 3.3 奖励函数设计（明确鼓励主动验证与失败恢复）
总奖励 $R = R_{\text{term}} + R_{\text{gather}} + R_{\text{verify}} - P_{\text{safety}}$：
- **终态成功奖励 $R_{\text{term}}$**：成功将包裹提离桌面达标高度并稳定悬停 $3.0\,\text{s}$ 无脱落，给予 $+10.0$；若在正常提拉中途发生掉落，给予 $-4.0$（严惩盲目硬拉导致真机摔包）。
- **聚拢过程引导 $R_{\text{gather}}$**：按 $\min(s_t, s_{\text{crit}}) / s_{\text{crit}}$ 给予有界塑形奖励（封顶防刷分）。
- **主动验证与恢复奖励 $R_{\text{verify}}$**：当 Phase B 处于虚夹持或即将滑脱状态（$T_t > 0.85 \mu_{\text{eff}} F_n$ 或 $s_t < s_{\text{crit}}$）时：
  - 正确触发 `mode == 1` 增力稳住并最终成功提拉，额外奖励 $+2.0$；
  - 正确触发 `mode == 2` 主动松开重抓并在后续轮次成功提拉，额外奖励 $+1.5$。
- **安全与效率惩罚 $P_{\text{safety}}$**：过度下压/夹持力惩罚（防止压溃或撕裂包装）、滑移能量耗散惩罚、每次重抓的时间/步数惩罚（$-0.3/\text{retry}$）。

---

## Module 4: 三层域随机化（Domain Randomization）完整清单

为保证**零 Real2Sim 下真实世界成为训练分布的严格子集**，按以下三层结构实现 `DomainRandomizer` 类：

| 随机化层级 | 具体参数 | 采样规则与范围策略 | 物理/工程目的 |
| :--- | :--- | :--- | :--- |
| **Tier 1: 接触与材料机制（核心）** | 全局可用余料 $S_{\text{bag}}$、聚拢效率 $\alpha$、等效摩擦 $\mu_{\text{eff}}$、滑移率 $\beta$、膜刚度 $k$、捕获阈值 $s_{\text{crit}}$、最小压触力 $F_{\min}$ | 基于 Stage 1 导出的 `proxy_material_envelopes.yaml` 向外扩展 **1.5x–2.0x**；采用**高斯混合 / 截断协方差采样**保留跨参数物理相关性，剔除非物理无解组合 | 覆盖从 25% 松垮空袋到 95% 紧绷满袋、从高滑 PE 膜到高挺牛皮纸的连续物理流形 |
| **Tier 2: 宏观载荷与几何** | 包裹总质量 $m$、3D 质心偏心距 $\mathbf{r}_{\text{com}}$、包裹长宽厚与分段铰链刚度、桌面摩擦、初始摆放倾角与高度误差 | 质量对数均匀采样（如 $0.05\text{--}2.5\,\text{kg}$）；质心偏心覆盖最大尺寸的 $\pm 35\%$；倾角 $\pm 15^\circ$ | 覆盖内部不规则硬物/衣物偏载引发的提拉倾斜、单侧受力过载与摆动剥离 |
| **Tier 3: 硬件 Datasheet 鸿沟（极关键）** | 力传感全局增益缩放（**0.7x–1.3x**）、回合零点漂移、高斯白噪声、低通滤波截止频率抖动、传感与控制延迟（**10–50 ms**）、夹爪电流-力映射误差、伺服刚度与死区 | 按目标机械臂/夹爪/力传感手册最坏指标的 **2x–3x** 保守注入；每回合独立采样 | 彻底阻断策略依赖绝对力值或无延迟力尖峰，防止零微调部署真机时控制震荡 |

**课程与对抗边界采样策略**：
1. **两阶段难度推进**：Stage A（窄物理包络 + 零硬件延迟噪声）训练至成功率 $\ge 85\%$ 后，线性过渡至 Stage B（全宽物理 DR + 完整 Tier 3 硬件鸿沟）。
2. **固定极端角点监控集（Corner Suite）**：固化 6 组极端参数角点（例如「极滑 PE + 90% 紧绷 + 重载大偏心 + 50 ms 延迟 + 0.7x 力增益」），仅当所有角点均有可行解且策略成功率同步达标时判定收敛。
3. **失败边界重加权**：训练后半段按回合失败率周期性提高高难区域的采样概率，防止策略只在简单均值工况上刷分。

---

## Execution & Verification Protocol
请按以下顺序分步交付代码，每一步必须附带**可直接运行的单元测试**：
1. 先输出 `ContactProxyConfig` 与无分支 `LumpedContactProxy` 类（含 Module 1 全部方程、防作弊单元测试、三种规范波形绘图脚本）。
2. 再输出 `stage1_to_proxy_calibrator.py`（读取 Stage 1 曲线并生成 `proxy_material_envelopes.yaml` 与相关性矩阵）。
3. 再输出 `DomainRandomizer`（含 Tier 1–3 采样与 6 组 Corner Suite 生成器）。
4. 最后输出 `SoftParcelGraspEnv` 与非对称 Actor-Critic（TCN/GRU Actor + Privileged Critic）的训练接口及维度自检脚本。

请先确认接口与物理单位无歧义，然后等待我指令从第几步开始生成代码。

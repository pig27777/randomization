# RSS 2027 Stage 1 首轮任务检查

最近检查时间 2026-11-19 21:38（北京时间）。机器 ubun，MuJoCo 4.0.1。完整实验配置与证据见[阶段报告](../test/bag_model/RSS2027_STAGE1_REPORT.md)。

| 范围 | 检查方法 | 结果 |
| --- | --- | --- |
| 单指原生接触 | 持共享锁运行 `rub_contact_e1.py`，碰撞开关各一次 | 旧刚体反力读数失败。改为从约束雅可比恢复反力后，正负对照均通过，最大反力差 $6.94\times10^{-18}$ N。 |
| 随机几何 | 持锁运行 `rss2027_geometry.py`，种子 20261119 | 128 个样本通过填充率、变化范围和余量非负检查。 |
| PE 微切片 | 持锁运行 `rss2027_micropatch.py`，比对余料、支撑、步长和摩擦 | 强支撑后穿透小于 7 μm，平膜未聚拢。预褶两档步长均出现正拉力，零指垫摩擦对照的正拉力为零。 |
| 整袋及 GPU | 统一 YAM 入口 2 ms，RTX 5090 静置与合指各 10 步 | CPU 装配与 GPU 短步进完成，GPU 左指接触力差最高 1.046 N。见[GPU 报告](rss2027-stage1-yam-gpu-probe-test.md)。 |
| MuJoCo 查看器 | `view_rss2027_fold.py --check` 核对预褶和仓库整袋轨迹，启动原生窗口后切换到侧抓褶皱近景 | 预褶首、中、末帧与归档节点在 1 pm 容差内一致。整袋顶抓 252 帧、侧抓 500 帧时间匹配。侧抓窗口以 `--start-time 7.42 --fold-edge 94 95` 打开。 |
| 褶皱视频 | `render_rss2027_fold_video.py --edge 94 95 --start .32 --end 8.20 --highlight-time 7.42`，FFmpeg 全量解码，ffprobe及两帧目检 | 1280×512，198 帧，25 fps，7.92 s。修订视角后，可同时看见整袋和夹口附近的膜面弯折。 |
| 既有袋模型单元检查 | `.venv-physical-parcel/bin/python -B test/bag_model/test_physical_parcel.py` | 16 项通过，测试夹具产生预期的 flex 诊断提示。 |
| 文件检查 | Python 源码语法检查、CSV 行尾规范化和 `git diff --cached --check` | 5 个 Python 文件通过解析，12 份 CSV 统一为 LF 行尾，暂存区格式检查通过。 |
| PE 下压力复测 | 持共享锁用 `rss2027_micropatch.py --dt .00005 --pad-contact-time .001` 与 `--pad-contact-time .0001` | 两组均完成，峰值实际下压力 4.86 / 12.13 mN，未达到 0.4–0.6 N 验收带。 |
| GPU 接触差异 | 同状态逐点比对 CPU/GPU 合指首步 | 两个左指浅触点解释首步左指竖向反力差的约 92%。其后的轨迹分岔仍待验证。[同状态检查报告](rss2027-stage1-yam-contact-parity-test.md) |

还需验证平膜起皱、材料曲线、脱手终点和闭环 GPU 基线。关键运行文件经仓库归档，其余探索轨迹和冗余快照保留在本机运行目录。

## 空袋整袋修复检查

本轮在 ubun、MuJoCo 4.0.1 上使用仓库 YAM、142 节点封口双膜片和 PE 预设，按用户要求改为整袋试验。正式动力学持 `/tmp/claude-gpu.lock`。

| 范围 | 检查方法 | 结果 |
| --- | --- | --- |
| 原试片反力 | `normal_probe.py` 和 `normal_dwell.py` | 静态触点力与广义反力吻合，原接触设置在恒位移足够时能传递超过 0.5 N。 |
| 空袋与力门槛 | 检查编译模型、指尖几何和连续 10 ms 门槛；运行 20 ms 入口；`.venv-physical-parcel/bin/python -B test/bag_model/test_physical_parcel.py` | 袋膜 142 节点、零内容物、零袋膜附着，保留 YAM 关节耦合。单元测试 17 项通过。 |
| 空袋静置 | `settle_probe.py` 与 `collision_probe.py` | 默认容差三档步长约 1 mm 自接触尖峰。原生凸碰撞收紧到 1 nm 后，两档步长的 0.30 s 静置均通过，峰值 0.150 mm。 |
| 空袋完整抓取尝试 | `record_physical_parcel.py` 读取 `rss2027-stage1-empty-yam-r2/config.json` | 2.2242 s 下压力门槛达标，直接碰桌反力零。总历时 4.3194 s，起抬后双指失联超过 50 ms，抓取失败。首次接触峰值 14.43 N。 |
| 空袋接近限速复测 | 同一入口读取 `rss2027-stage1-empty-yam-r3/config.json` | 3.4584 s 法向力验收通过，接近峰值降至 0.646 N、原生穿透峰值 0.150 mm、零告警、直接碰桌反力零。试验达法向门槛即停止。 |

本轮归档了完整尝试和独立加载复测，限速版的闭合与提拉尚待验证。后续记录整袋的连续参数样本、捕获与滑移临界点及完整 GPU 回合。

## 整袋夹入诊断、同参 PE 与内容物装配

最近检查时间 2026-11-20 02:33（北京时间）。ubun，MuJoCo 4.0.1，GPU 在线入口使用 RTX 5090 与 MuJoCo Warp 4.0.1。正式动力学均持项目共享锁。夹口膜量及依赖该量的事件见坐标复核行。

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| R4 限速空袋 | `record_physical_parcel.py` 读取 R4 配置，`rss2027_yam_capture_audit.py` 对齐 MJB、`trajectory.npz` 和力轨迹 | 0.5 N 压力带通过。闭合与起抬 105 帧均无双侧内表面接触候选及夹口内膜三角面，起抬拉力零。归为空夹。回放只重建几何，20 ms 采样间隙不做接触推断。 |
| 空袋增加闭合保持 | `record_physical_parcel.py` 读取 `rss2027-stage1-empty-yam-close-hold-r1/config.json`，再回放 120 帧 | 起抬夹口降到 0.581 mm，膜仍未进入夹口，桌面支撑至少 0.09328 N。旧整袋侧抓正对照能识别双侧内触点、夹口内膜面及 4.668 N 正拉力。 |
| GPU 在线反馈 | `record_physical_parcel.py --engine warp`，同模型依次检查 20 ms 入口及 0.5 N 力控 | 入口通过。加载在 9.859 s 仿真、约 1200 s 墙钟停于预算，未满足连续 10 ms，最大采样显存 964 MiB，零容量溢出。未测得完整 GPU 回合。 |
| 同参 PE 第一批 | `flock /tmp/claude-gpu.lock bash test/runs/2026-11-19/bag/rss2027-stage1-pe-matched-r1/run_cases.sh`，枚举平坦余料、预褶、零摩擦，分别复核半网格和半步长 | 五组数值检查通过。平坦余料和预褶出现双侧承载及完全脱开，零摩擦无有效拉力。半步长峰值差约 1.9%，半网格 +216%，网格验收失败，起滑力仍为候选。首批含有效捕获的墙钟字段有计数变量覆盖错误，已修复。 |
| 同参 PE 按面积分配质量 | `flock /tmp/claude-gpu.lock bash test/runs/2026-11-19/bag/rss2027-stage1-pe-matched-r2/run_cases.sh` | 5 组完成。平坦有余料峰值 0.4091 N，粗/细网格预褶 0.4854/0.5268 N（差 8.5%）。半网格起滑候选相差 32.7%，整体定量网格门槛待过。半步长的峰值、时长、起滑候选和脱开差分别小于 1%。 |
| 切向接触复核 | `rss2027_coupon_sliding.py` 分析预褶粗、半网格、半步长轨迹 | 触点估计分别为 0.0942、0.3375、0.0996 N，网格加密后仍未收敛。保留采样近似和候选定义。 |
| 同 6.5 N 夹持力复测 | `flock /tmp/claude-gpu.lock bash test/runs/2026-11-19/bag/rss2027-stage1-pe-clamp-force-r1/run_cases.sh`，补做相同触点速度复核 | 粗细网格均达标 60 ms，起拉时夹口材料 9.786/6.356 mm，峰值 0.4295/0.6564 N，仍超 10%。 |
| 材料随机化与位置入口 | `.venv-physical-parcel/bin/python -B test/bag_model/test_physical_parcel.py`，核对旧准备态指纹 | 19 项通过。显式厚度覆盖保留 PE 碰撞皮肤；同为 30% 解析填充率的内容物尺寸与偏置可装配，参考袋形一致，实体三角面无相交。旧居中准备态兼容。 |
| 同 30% 填充率顶部抓取 | `flock /tmp/claude-gpu.lock bash test/runs/2026-11-19/bag/rss2027-stage1-content-pair-r1/run_cases.sh`，再逐帧回放 | 两组下压力达标分别需 10.088/8.465 s，起抬时均无夹口内膜面或正拉力。 |
| 几何与源文件检查 | 比对 R4 MJB 数值皮肤与指垫摩擦、匹配粗细试片几何、验证夹口内材料线裁剪，解析 Python 和 `git diff --check` | 摩擦 0.8、数值皮肤半径 0.25 mm 一致；加密后仍保持 10 mm 展开褶宽和 8 mm 余料；夹口材料长度的已知几何值核对通过。 |

原始值、配置和后续批次见[Stage 1 实验报告](../test/bag_model/RSS2027_STAGE1_REPORT.md)。

## 整袋侧抓、细网格与 GPU 分段计时

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| 整袋侧抓正对照 | 持锁执行 `rss2027-stage1-side-grasp-r1/run.sh`，回放核查夹入及导出提拉力—位移 | 8.50 s 试验覆盖 0.20 s 离台悬停，质心升高 146.57–158.59 mm，峰值拉力 5.049 N、原生穿透 0.484 mm、零告警。 |
| 侧抓褶皱录像 | `render_rss2027_fold_video.py --edge 94 95 --start .32 --end 8.42 --highlight-time 7.42`；FFmpeg 完整解码并目检 | 1280×512、203 帧、25 fps，膜边角度 5.41°→106.49°。[视频](../test/runs/2026-11-19/bag/rss2027-stage1-side-grasp-r1/yam-side-fold.mp4)。 |
| 同参 PE 整袋侧抓 | 持锁执行 `rss2027-stage1-side-grasp-matched-pe-r1/run.sh`、`half-step-r1/run.sh` 与 `fine-step-r1/run.sh`，检查包覆并归档峰值接触 | 100 μs 露角停止，50 μs 悬停但穿透超限，25 μs 通过。25 μs 悬停质心升高 143.18–155.90 mm，膜—膜穿透峰值 0.642 mm，独立膜—物几何无相交。匹配材质视频完整解码并目检。 |
| 单轴网格与 1.25 mm 两轴网格 | 执行 `rss2027-stage1-pe-mesh-axes-r1/run_cases.sh` 和 `rss2027-stage1-pe-quarter-grid-r1/run.sh`，复用切向速度检查 | 单轴均达到夹持力，闭合轴加密使起拉膜长减少约 3 mm。定开口预褶的 2.5→1.25 mm 峰值、持时、脱开差小于 5%，触点起滑仍差 63%。 |
| GPU 耗时分解 | 持锁执行 `rss2027_yam_warp_profile.py`，用 R4 两个状态帧分整图与两段图各 64 步 | `step2` 18.46–33.01 ms/步，完整反馈约 3.23–3.62 ms/步，零溢出。这些数据用于瓶颈定位，完整回合仍待测。 |
| 3–5 s 基线 CPU 候选 | 执行 `rss2027-stage1-pickup-baseline-cpu-r1/run.sh` 和 `cpu-r2/run.sh` | 首个候选 6.50 s 仍在闭合；第二个以 200 μs 完成 4.226 s 抓取悬停，墙钟 82.15 s，穿透 0.697 mm，零告警。 |
| 基线 GPU 200 μs | `rss2027-stage1-pickup-baseline-warp-r2/run.sh`，对停止态以 CPU 重算包覆几何 | 2.7604 s 提拉失败，墙钟 469.11 s、进程显存最高采样 966 MiB、零溢出。独立确认一个角点露出。 |
| 基线 100 μs | CPU 和 GPU 各执行 `rss2027-stage1-pickup-baseline-*-r3/run.sh`，并以 CPU 几何重算 GPU 最终状态 | CPU 通过 4.2248 s，墙钟 151.49 s，穿透 0.367 mm。GPU 在 2.7595 s 露角，墙钟 887.12 s、采样显存 966 MiB、零溢出，原生穿透峰值 1.067 mm。 |
| 牵引分组和穿透峰值记录 | 使用已存持载片验证分组，25 μs 同参整袋实际记录峰值 | 两侧摩擦锥分类通过包含关系核对，同态峰值定位到闭合期膜面 3/151，并用原三角面投影复核。 |
| 原生触点整体滑移 | 持锁执行 `rss2027-stage1-pe-traction-r1/run_cases.sh`，对比 2.5/1.25 mm 原生摩擦锥和有载切向速度 | 两档候选力 0.33756/0.29977 N，力差 11.2%；时刻 0.547/0.529 s，差 3.3%。力仍未过 10%。 |
| 平坦稳压聚拢 | 持锁执行 `rss2027-stage1-pe-gather-force-r1/run.sh`、`gather-force-r2/run_cases.sh` 与 `pe-gather-fine-grid-r1/run.sh`，在原始压力窗比对共同行程 | 5 mm 半步长差 0.15%；2.5→1.25 mm 共同行程斜率 0.67217→0.72519 m/m（差 7.89%），通过该工况的细网格复核。 |
| 共享平台兼容 | `.venv-physical-parcel/bin/python -B test/bag_model/test_physical_parcel.py`，检查新增字段默认值和 Python 源码 | 19 项通过。 |
| CPU/Warp 内容物接触法向 | 持锁执行 `rss2027_payload_contact_parity.py`，用 4 个同态姿态读取 Warp 约束，再给膜 0.01 m/s 法向投影对照 | 闭合末帧与停止帧的膜面 191 法向差为 89.18°、87.73°，设备约束投影误差小于 1 nm/s，零容量溢出。 |
| 改变 PE 聚拢条件 | 持锁执行 `rss2027-stage1-pe-gather-conditions-r1/run_cases.sh` 并按原始压力窗拟合 | 4 mm 余料组斜率 0.713，0.8 N 下压力组斜率 0.709，均取得合格压力窗。各只运行一次，新增两组的独立网格复核待做。 |
| 短回合慢速提拉 | 持锁执行 `rss2027-stage1-pickup-baseline-*-r4/run.sh`，核对离台并回放 GPU 停止态 | 0.18 m/s CPU 通过 4.7704 s，墙钟 91.60 s。GPU 在 2.7804 s 露角，墙钟 477.94 s、966 MiB、零溢出，穿透峰值 0.926 mm。 |
| 整袋夹口材料 | `rss2027_yam_capture_audit.py` 回放 R4 空袋与同参 PE 侧抓，另用已知面积验证裁剪 | 平面裁剪的变形/参考面积为 4/24 m²。空袋 105 帧为零，整袋 PE 起抬 3.98 mm、悬停 8.74–10.19 mm，使用 20 mm 指宽和矩形夹口等效。 |
| 整袋材料装配与入口 | 以同一 142 节点资产检查四种 `parcel_world`，运行 `rss2027-stage1-whole-bag-materials-r1/run.sh smoke`，按单步位置核对峰值。复跑 `.venv-physical-parcel/bin/python -B test/bag_model/test_physical_parcel.py` | 19 项回归通过。PE、气泡膜的 200 μs 入口通过；牛皮纸 200 μs 与纸板 100 μs 在静置早期的穿透峰值 1.416/1.390 mm 超限。 |
| PE 同参整袋短动作与 15 mm 偏置 | 持锁运行材料对照 `run.sh pe` 和偏置 CPU `run.sh`，分别检查夹入数据和离台门槛 | PE 200 μs 成功夹入，3.0202 s 露角停止。偏置的原 PE 类整袋 CPU 通过 4.7804 s，墙钟 87.62 s，穿透 0.526 mm，GPU 对照已启动。 |
| 偏置在线 GPU、约 30% 填充率 CPU | 持锁执行 `rss2027-stage1-pickup-baseline-warp-r5/run.sh` 和 `cpu-r6/run.sh`，以两端轨迹复核几何与悬停 | GPU 偏置 2.8805 s 露角，穿透 0.676 mm、966 MiB、零溢出。约 30% CPU 在 4.7626 s 通过离台、双侧承载和穿透门槛。 |
| 气泡膜动作、纸材 0.5 ms 入口 | 持锁执行材料首轮 `run.sh bubble` 与 `run.sh smoke <材料复测目录>` | 气泡膜 3.0188 s 失触，起抬有 2.63 mm 等效膜长但无正拉力。两种纸材入口峰值 0.855/0.748 mm，均通过，完整回合已启动。 |
| 同参 PE 起拉复核 | 对存档运行 `rss2027-stage1-pe-aperture-width-r1/measure.py`，再持锁执行 `rss2027-stage1-pe-low-slack-r1/run_cases.sh` | 捕获帧整指宽等效长度 5.33/3.04 mm，起滑前候选刚度 77.52/33.58 N/m，超载各 9 ms。低余料或低压力三组均无正拉力，其中两组有合格收拢窗口。 |
| PE 中等余料和较高压力 | 持锁执行 `rss2027-stage1-pe-capture-boundary-r1/run_cases.sh`，解析 1 ms 原始采样 | 4 mm／0.5 N 连续承载 0.318 s，峰值 0.0286 N。8 mm／0.8 N 承载 0.545 s，峰值 0.446 N。两组均脱开、数值检查通过。 |
| 牛皮纸 0.5 ms 完整动作 | 持锁执行材料首轮 `run.sh kraft <材料复测目录>`，回放夹口与峰值触点 | 完成 6.187 s 及离台悬停，闭合期原生膜—膜峰值 1.483 mm 超限。夹口回放未见同帧双侧内表面触点。 |
| 慢速拉拔 | 持锁执行 `rss2027-stage1-pe-slow-pull-r1/run.sh`，对比旧 25 mm/s 轨迹的原生滑动与行程 | 慢速数值通过，整体滑移候选力 0.063 N，原速 0.338 N；慢速脱开行程 13.60 mm，原速 14.33 mm。双侧正超载均仅 9 ms。 |
| 约 30% 整袋在线 GPU | 持锁执行 `rss2027-stage1-pickup-baseline-warp-r6/run.sh`，独立回放停止态 | 3.1805 s 露出角点 0，墙钟 560.24 s、966 MiB、零溢出。CPU 同时间完整包覆。[几何复核](../test/runs/2026-11-20/bag/rss2027-stage1-pickup-baseline-warp-r6/geometry-recheck.json)。 |
| 纸板完整动作与膜参数入口 | 持锁执行材料首轮 `run.sh paperboard <复测目录>`，逐步峰值投影核对；对组装 MJB 检查 `solref` 和优先级 | 纸板离台，提拉期膜—膜穿透 1.719 mm 超限，悬停一侧闭合轴力曾为零。新 0.25 ms 只改变膜的 `solref`，指、盒、台保持 0.5 ms。19 项既有回归通过。 |
| 4 mm 临界细网格 | 持锁执行 `rss2027-stage1-pe-capture-fine-r1/run.sh`，比对前 0.10 s 原位移和力 | 1.25 mm 档无正拉力，粗档的边界未重现。峰值穿透 0.269 mm，原始合格压力窗 0.612 s。 |
| Warp 凸碰撞与法向 | 持锁运行两档容差与隔离副本的同态测试，`check_normal_depth.py` 计算原生法向投影 | 改容差不消除差异。保留 GJK/EPA 法向后最大角差 0.004°，距离误差 1.848 μm，零溢出。 |
| 纸板膜接触复测 | 持锁执行材料脚本，膜参数设为 0.25 ms | 20 ms 入口通过。动作完成，但同一对膜三角面仍有 1.242 mm 穿透且一侧悬停力曾归零。 |
| 隔离版完整 GPU 基线 | 持锁执行 `rss2027-stage1-warp-film-normal-r1/run_trial.sh`，用 `analyze_trial.py`、独立停止态几何和夹口回放检查 | 4.76965 s 模拟，539.12 s 墙钟，966 MiB，穿透 0.469 mm。10 个悬停采样均离台、双侧有载、无几何外露，原生峰值与零告警均过。 |
| GPU 通过回合录像 | `render_rss2027_fold_video.py --edge 94 95 --start .32 --end 4.75 --highlight-time 4.68`，FFmpeg 完整解码并目检 | 1280×512、112 帧、25 fps。膜边 5.41°→72.50°，终态双视角可核查。[视频](../test/runs/2026-11-20/bag/rss2027-stage1-warp-film-normal-r1/yam-warp-fold.mp4)。 |
| 8 mm 名义细网格 | 持锁运行 `rss2027-stage1-pe-nominal-capture-fine-r1/run.sh`，比较收拢、起拉和 2.5 mm 记录 | 1.25 mm 试片能收拢，0.40 mm 开口处保留膜为零，首 0.15 s 无正拉力。1.10 mm 开口已排队。 |
| 受控拉力双侧承载 | 持锁运行 `rss2027-stage1-pe-force-pull-r1/run.sh` 并按原生反力与相对位移抽查连续窗 | 0.270–0.281 N 持续 0.1229 s 时合计夹力 6.50–6.72 N，未出现整体滑移。原生穿透 0.250 mm、零警告。 |
| 隔离版约 30% 填充率完整动作 | 持锁执行 `rss2027-stage1-warp-film-normal-30pct-r1/run.sh`，校验悬停、夹口和停止态包覆 | RTX 5090 通过 4.75966 s 完整模拟，墙钟 579.86 s，966 MiB，峰值穿透 0.813 mm。悬停支撑为零，停止态无露角或相交。 |
| 纸板上缘侧抓 | 持锁运行 16 mm 进给位置对照，复算双指触点并检查实际支撑与原生峰值 | 6.0969 s 完成，穿透峰值 0.685 mm，悬停 10 帧离台，最低单指闭合轴力 6.98 N。夹口回放仍缺同时双侧内法向。 |
| 细网格保留开口与 1 N 预褶 | 持锁执行 `rss2027-stage1-pe-residual-gap-r1/run.sh` 和低夹力对照 | 1.10 mm 开口在起拉时保留 4.095 mm，但无正拉力。预褶从宽开口施加 1 N 目标，保持时夹力仍为零，未进入目标拉力带。 |
| 通过场景的设备短窗 | 持锁执行 `rss2027-stage1-yam-batch-reference-r1/run.sh`，逐步检查状态、容量与原生峰值 | 9 个阶段／世界数组合通过，最大穿透 0.253 mm。64 世界最高 2,277.72 环境步/s、采样显存 1,798 MiB。 |
| 牛皮纸与气泡膜上缘进给 | 持锁执行各一次 16 mm 高度对照，复查原始峰值和夹口 | 牛皮纸离台但膜—膜峰值 1.931 mm 超限。气泡膜在起抬前跌离原支撑，夹口无膜、无正拉力。 |
| 已入夹后的 PE 夹力反馈 | 持锁执行 `rss2027-stage1-pe-gather-grip-r1/run.sh`，检查控制入口、保持及起抬 | 细档起抬保留 3.408 mm，合力 0.825 N、无正拉力。夹力和捕获判据未过，穿透峰值 0.294 mm。 |
| 完整整袋受控松夹 | 持锁运行 `rss2027-stage1-whole-bag-release-r1/run.sh`，以 2 ms 样本复算材料量与原生载荷 | 200 μs CPU 整袋通过悬停和自由脱手。滑移候选在 9.03 mm 开口和 3.407 N，约 42 ms 后两指失触。全程穿透 0.601 mm、零告警。共享默认路径回归 19 项通过。 |
| 整袋松夹半步长与纸板 | 持锁运行同共享脚本，追加原生活动约束记录并执行 `rss2027_release_traction.py` | PE 类 100 μs 完成脱手，候选力差 29.5%。纸板完成脱手，允许数值预算内的整袋承载，原生记录没有持续双侧整体滑移。 |
| 牛皮纸上缘半步长与膜顺应性 | 持锁运行两轮同输入对照，按原模型盒体/膜面投影复核峰值 | 100 μs 仍有 1.902 mm 膜—膜穿透。膜时间常数 0.25 ms 后最深接触转为右支架—膜 1.628 mm，几何复核一致。 |
| 50 μs 整袋松夹 | 持锁运行 `rss2027-stage1-whole-bag-release-r3/run.sh`，比对 2 ms 样本的开口、拉力以及单双侧活动约束 | 脱开开口 10.595 mm；三档在 10% 内。候选拉力 2.993 N，三档仍未收敛。原生悬停 10 帧均双侧内载；松夹有持续单侧滑动，无持续双侧滑动。 |
| 气泡膜 110 mm 起始净空 | 持锁运行 `rss2027-stage1-bubble-clear-start-r1/run.sh`，使用末帧位姿及速度重算接触 | 1 mm 数值入口通过。20.2 ms 时右指合力 1.446 N，复算触及膜面 6、146，未过指尖净空门槛。135 mm 工况排队。 |
| 气泡膜 135 mm 净空与筛查 | 持锁运行 `rss2027-stage1-bubble-clear-start-r2/run.sh`，回放闭合与提拉的 81 帧 | 静置指尖反力为零，完整侧抓 3.3772 s、峰值 0.177 mm，内容物留在高台、无夹入和正拉力。 |
| 牛皮纸 10 mm 高度 | 持锁执行材料首轮 `run.sh kraft <本轮目录>`，核对原生最大穿透和首个提拉观测 | 峰值 1.591 mm 回到膜—膜，4.5017 s 双指失触，未通过离台。16 mm 高度净空复测已启动。 |
| PE 入夹后卸压 | 持锁运行 `rss2027-stage1-pe-gather-relief-r1/run.sh`，回放夹力和局部节点 | 峰值 0.177 mm、零告警。起抬竖向压触由 0.812 降到 0.103 N，内侧合力由 0.825 降到 0.095 N，未建立正拉力。经定位成对隆起，偏移对照排队。 |
| 整袋夹口坐标复核 | `.venv-physical-parcel/bin/python -B test/runs/2026-11-20/bag/rss2027-stage1-aperture-axis-r1/replay.py` | 29 条轨迹、31 份曲线、7,801 个观测。正开口指体 y 与闭合方向误差最大 $2.77\times10^{-12}$。时序、反力和触点一致。三档整袋 PE 脱手仍在 10.45–10.59 mm，纸板只有 22.06 mm 的完全脱手通过。[复算索引](../test/runs/2026-11-20/bag/rss2027-stage1-aperture-axis-r1/index.json)。 |
| 试片 0.625 mm 偏移 | 持锁执行 `rss2027-stage1-pe-offcenter-gather-r1/run.sh`，先比对初始节点和拓扑，再检查夹口力、材料及提拉 | 775 个节点的参考和初始位置按预定偏移。细网格夹力达标 89.4 ms，双侧正拉力达标 192.9 ms，峰值 0.0812 N。0.25 s 内未脱手，网格和步长复核尚待。 |
| 袋模型回归 | `.venv-physical-parcel/bin/python -B test/bag_model/test_physical_parcel.py` | 19 项通过，保留 `crossing_triangle` 测试夹具的预期告警。 |

# Isaac Lab 第 1 轮任务检查

最近检查：2026-11-08 19:02（Asia/Taipei）。

**结论**：独立安装及 Newton/Isaac Sim 两种启动检查通过。三类冻结描述质量/质心检查通过，CPU 参考正式 54 次、sim_fail=0。L1 正式九次通过；当前 L2 适配静置失败，L4 参数能力已做初测，整轮 L1–L6 尚未完成，当前不宜迁移内容物训练。

**11月8日续检**：按用户允许将5%初态边应变保留为诊断后继续L5。13项相关检查通过，圆角指尖裸硬盒完整侧夹1次成功、动作墙钟5.853s。易压物+牛皮纸袋入口在准备≤50ms确认相邻段重叠/越限，并出现VBD逐体接触缓冲318>64；当前实现未通过，按要求停止。180次L5矩阵、加速行为对照、L6和L2′/L3′均未进入，尚不能选定GPU训练平台。详细续检记录在文末；原失败历史保留。

| 范围 | 命令或检查方法 | 结果 |
|---|---|---|
| 约束与输入 | 阅读 `AGENTS.md`、Isaac Lab 计划、描述层、任务 v2、R1–R3 测法、袋子说明与牛皮纸 profile | 已核对，不改标准和其他线代码 |
| 主机 | `nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv`；`df -h /home/ubuntu/Dev/Envs`；`python3 --version`；`free -h` | RTX 5090 / 32607 MiB、驱动 580.178.04、约 1.6 TiB 空闲、Python 3.12.3 |
| 官方依赖 | 阅读官方 v3.0.0 安装文档、`pyproject.toml` 与 `uv.lock`，来源见实验报告 | 文档支持 Newton 无 Isaac Sim；MuJoCo 3.12 与 CPU 3.17 差异已记录 |
| 冻结输入 | 标准库 `hashlib.sha256`、`tarfile` 只读比对 `v1.json` | 压缩包、22 个包内源码与换算表 SHA256 全部一致 |
| 牛皮纸网格 | 已有 `mjwarp/bin/python -B`，`numpy.load(..., allow_pickle=False)` | 成功：413 顶点、792 三角形 |
| 安装与导入 | 官方 v3.0.0 冻结锁导出 PEP 751，保留 URL/哈希；分段大包整包校验；固定 workspace editable；`./isaaclab/run.sh -c ...` 导入 | 292 包；Newton 1.6.1、Warp 1.17、torch 2.11+cu130、NumPy 2.3.1、Numba 0.64、ONNX GPU 1.26 导入通过；源码 clean |
| 依赖检查 | `uv pip check --python /home/ubuntu/Dev/Envs/isaaclab/bin/python` | exit 1：仅 fsspec/urllib3 两项官方根配置明确覆盖的元数据冲突，未记录为全通过 |
| Newton 入口 | `OMNI_KIT_ACCEPT_EULA=yes flock /tmp/claude-gpu.lock timeout 300 ./isaaclab/run.sh isaaclab/cloth_benchmark.py --num_envs 1 --warmup 2 --steps 5 --output isaaclab/results/install/newton-smoke.json` | exit 0，官方指定 preset；输出有限；短检查不计正式吞吐 |
| Isaac Sim 入口 | `OMNI_KIT_ACCEPT_EULA=yes flock /tmp/claude-gpu.lock timeout 300 ./isaaclab/run.sh isaaclab/check_isaacsim.py --output isaaclab/results/install/isaacsim-smoke.json` | exit 0，无窗口启动 11.88 s，USD stage 存在，完成 10 次更新并关闭；未测 GUI、传感器或机器人任务 |
| CPU 参考 | `mjwarp/bin/python -B isaaclab/cpu_reference.py --batch`，内部逐配置 flock | 三类×6测法×3次，共54次、sim_fail=0；不是 Isaac Lab 对照通过 |
| 报告静态检查 | JSON 解析、Markdown 相对链接与 L1–L6 表格检查 | 见文末复核 |

原始预检：[preflight.json](../isaaclab/results/preflight.json)。计划、清单和测试状态：[第 1 轮实验报告](../test/isaaclab/REPORT.md)。

**失败与限制**：工作区无 Git 元数据，`git status --short` 返回 128，无法提供 Git diff。首次环境探查附带 `import scipy` 返回 `ModuleNotFoundError`，随后核对描述层形状实现无需 SciPy；实际所需 NumPy/MuJoCo 可用，未修复或更改共享环境。安装后空间含缓存 35.37 GiB；尚无正式 GPU 吞吐或 CPU/GPU 一致性结论。冻结 v1 不代表 v1.1/v1.2 或顶面抓已通过。

**历史预检复核（安装前）**（2026-11-05T21:17:59+08:00）：预检 JSON 解析、冻结输入标记、4 个本地文档链接、L1–L6 未运行状态、原 10%/10 倍线及第 2 轮未运行记录检查通过；独立环境仍不存在。无动力学测试。

**安装失败与修复记录（2026-11-06）**：大 wheel 长请求反复重置，改为 16 MiB 分段续传并验整包 SHA256；初次普通 requirements 多索引误选 CUDA torchaudio，哈希拒绝，改用保留锁定 URL 的 pylock 后成功，未放宽校验。原锁 NumPy 2.5.1 无法被 Numba 导入，独立兼容清单改为 2.3.1，并将 CPU ONNX 替换成机器人组件要求的 GPU 发行包。正式 L1 首次遗漏本进程 EULA 环境变量，启动前 exit 1（`n1-r1-eula.log`）；补齐后重跑，未改全局配置。

安装清单：[manifest.json](../isaaclab/results/install/manifest.json)；完整逐包记录：[installed-packages.json](../isaaclab/results/install/installed-packages.json)；CPU 汇总：[summary.json](../isaaclab/results/cpu_reference/summary.json)。详细日志统一位于 `isaaclab/results/install/` 与 `isaaclab/results/l1/`。原预检与失败日志保留，不以先前未安装时的静态复核替代本次启动验证。

**安装后正式检查（2026-11-06）**：

| 范围 | 执行入口 / 方法 | 结果与限制 |
|---|---|---|
| L1 | `flock` 下 `cloth_benchmark.py`，1/64/256 环境，各3种子，100步预热+1000步同步计时 | 9/9 exit 0；速度中位36.46/1212.85/4020.11环境步/s，进程测量期显存采样峰值662/726/1016 MiB；不是完整抓取吞吐 |
| L2 原生闭环 | `segment_adapter.py --loop-mode native`，三类描述，各一次；广度顺序警告修为深度优先后复核 | 硬物固定闭环可映射；易弯/易压37/57条D6闭环被跳过，exit 2 |
| L2 替代入口 | `segment_adapter.py --loop-mode proxy --rest`，先小入口；逐次配置见实验报告 | 默认缓存溢出、微小代理惯量被自动抬升均已记录并修正配置。最终代理1e-5 kg/1e-9惯量、njmax=nconmax=4096，易弯/易压在t=0.1472/0.1336s非有限；L2未通过，未把“图可构建”当作成功 |
| L4 参数能力 | `parameter_probe.py --num-envs 64 --seed <20261105/6/7>`；五个隔离测具，Newton和MJWarp双读回及运动响应 | 初批3/3有限，五项参数确实异值且生效；测具完整边长惯量修正后的复核在共享锁队列中，初批保留为`fixtures-small-inertia-*`，尚不宣称新版复核完成 |
| L4 异构/分桶 | `structure_probe.py`，三类混合3次；每类两个环境分桶各1次 | 混合3/3构建被拒，明确要求homogeneous worlds；分桶3/3构建成功，未测分桶动力学 |

**保留的失败**：代理入口5维向量曾用错误API `wp.vec`，已改`wp.types.vector`；首次非有限输出未传播失败且包含NaN，源脚本已修为严格JSON/null、首个非有限时间和exit 3，确认复现有效。旧原始输出仅保留为`.invalid-json`，不用作通过证据。部分动力学日志有线搜索达到20次的提示，未通过提高上限或放宽判据来消除；原20%越限/非有限等sim_fail标准保持。

**未完成范围**：GPU R1–R3 ≤10%对照、内容物训练级L4、运行中/reset后随机化、L5袋装v2侧夹/顶面抓、L6 256/1024完整抓取吞吐。未测能力不作否定或肯定结论。下一步闭环方法仍待实现验证；本次交付为安装后阶段记录，不是11月14日终验。

**本阶段静态复核（2026-11-06 14:00）**：7个本线Python入口AST解析、99个JSON严格解析、14个本地文档链接、`bash -n isaaclab/run.sh`通过；原10%和10倍标准仍存在。已知无效原始NaN文件仅以`.invalid-json`保留。官方源码Git工作区仍clean；环境当前含缓存约35.55 GiB。[复核记录](../isaaclab/results/install/static-check.json)。最后的完整箱体惯量复核等待其他线释放共享锁，未纳入已完成结果。

## L5/L6 续检（2026-11-08）

范围：原牛皮纸413节点网格、保面板/袋舌的150节点网格、硬盒及原易压锚点的GPU耦合入口；代码限定在`isaaclab/`。用户已批准保留>5%边应变诊断继续行为验收，未改原行为/失败标准。下列动力学/单元检查全部持`/tmp/claude-gpu.lock`，只读输入使用`RANDOM_L5_INPUT_ROOT=.../isaaclab/results/loose_package/reference_inputs`。

| 执行命令或检查方法 | 结果 / 关键数据 |
|---|---|
| `./isaaclab/run.sh isaaclab/package_assets.py --prepare --output isaaclab/results/loose_package/assets_sealed_behavior_20261108` | 6档×细/粗完整资产生成，缩放与旧完整清单一致；封口0.9mm。细网格应变4.42–6.94%、粗网格24.03–24.49%，仅诊断，未声称行为通过 |
| `./isaaclab/run.sh -m unittest discover -s isaaclab/tests -p test_package_pinch.py` | 最后13/13通过，13.276s，`unit-formal-rounded-pad.log`；覆盖点连接反力/自由转动、5ms解除、10mm聚料、原生驱动/承重、封口着色/初态、越限锁存、有向包围盒、完整缓存与带袋坐标变换 |
| `package_grasp.py --no-cloth --grasp side --output .../bare_side_rounded_gpu.json` | success，7.655s模拟/5.853s动作墙钟；含250mm提起、500mm搬运、放下松开。仅1次，不是256/1024吞吐 |
| `package_grasp.py --scale .35132402626147194 --mass-scale 10 --grasp top --w .020 --force 20 --limit 15 --cache .../hard_loose_canonical_mass10_cache.npz --output .../hard_loose_canonical_w20_f20_framefix.json` | 旧方角版抓取失败，无连接，9.525s模拟/328.977s动作。缓存末盒高188mm且强加载暂态，不能用来评价开口好坏或可用抓取速度 |
| `package_grasp.py --category compressible --scale 1.7320508075688772 --grasp top --limit .005 --cache .../compressible_loose_rounded_baseline_cache.npz --output .../compressible_loose_rounded_entry.json` | **sim_fail/exit3**，准备期≤0.050s，动作0次；重叠2.55314倍段长、压密7.65941倍限位、弯曲1.81359倍限位。dt50µs；准备43.728s、含初始化46.237s |
| 原描述与失败NPZ只读重算、日志匹配；不再运行物理 | 原描述初态重叠≈0，保存末态最坏为未表示的段12–15边，重叠2.54063；VBD逐体刚体接触缓冲1000条溢出，最高318>64，非已设4096的逐体粒子接触容量。见`compressible_loose_rounded_diagnostic.json` |

**修复与保留历史**：缓存坐标旋转曾升为float64，写入float32后触发逐位检查报错；显式保持原dtype后带袋缓存检查通过，断言保留。先前仅驱动Newton姿态、未驱动原生关节，以及mocap缺少速度的问题已改原生D6位置/速度驱动。补齐与CPU参考相同的2mm圆角指尖；缓存指纹加入形状/弹簧及场景源码，保存准备轨迹。所有旧失败、短试、资产和日志保留，不计成正式通过。

**未完成及下一步**：L5当前候选未通过，按用户要求停止实验；180次正式矩阵和四项加速行为对照未运行，L6批量抓取接口/256/1024吞吐未完成，L2′/L3′未运行。建议下一轮先修VBD逐体刚体接触容量并检查无袋/带袋隔离静置；缓冲溢出使结果不完整，不能断言原本构或平台不可行。粗网格预应变、硬盒缓存加载暂态、布料塑性/记忆迁移、点连接与CPU等价性仍有限制。未新增安装，未修改共享库和其他线源码。

结果入口：[L5/L6报告续节](../test/isaaclab/REPORT.md#本次入口结果与停止原因2026-11-08-1900asiataipei)；[资产清单](../isaaclab/results/loose_package/assets_sealed_behavior_20261108/manifest.json)；[易压物失败及源码指纹](../isaaclab/results/loose_package/compressible_loose_rounded_entry.json)；[状态证据](../isaaclab/results/loose_package/compressible_loose_rounded_entry.state.npz)；[离线诊断](../isaaclab/results/loose_package/compressible_loose_rounded_diagnostic.json)。

本次结束静态核对通过：9个相关Python文件AST解析、6个新相关JSON严格解析、8个冻结输入SHA256、最新裸盒/易压结果与当前源码指纹一致、失败状态7字段有限、15条本轮结果链接有效；未追加动力学运行。无本线包裹实验残留进程。[审计记录](../isaaclab/results/loose_package/audit_sealed_behavior_20261108.json)。

# 随机袋内容物 / 完整YAM / GPU容量

机器ubun，RTX5090 32607MiB；2026-11-16启动、跨至11-17。基线ac0a963a，实际改动以各任务manifest及source副本为准。环境见项目envs/physical-parcel-requirements.txt；正式运行持`/tmp/claude-gpu.lock`。

| 场景 | 已测加载上限 | 数值有效执行上限 | 物理预算有效上限 | 环境步/s |
| --- | ---: | ---: | ---: | ---: |
| 纸袋胶囊真实角部闭合 | 32768 | 8192 | 8192 | 32716.5 |
| 9状态/7物理模板混合 | 34560 | 23040 | 0（本轮失败） | 15821.6（诊断） |

加载最近失败界分别40960、36864；执行最近失败界10240、27648。仅加载不建立动力学图，不能当可执行容量。9次真实抓取0成功，无承重保持状态。23项回归通过；全部计时档3次、100步≈10ms。源范围/接触优先级及几何覆盖限制见主题报告。

- [最终表与原始索引](summary.md)、[机器可读汇总](summary.json)、[主题报告](../../../../bag_model/PHYSICAL_PARCEL_REPORT.md)、[任务检查](../../../../../reports/physical-parcel-yam-gpu-test.md)。
- [范围/分布](random_ranges.json)、[18组参数、采样和拒绝](preparation/groups.json)、[9条调度](schedule.json)、[CPU/GPU完整尝试](trial_summary.json)。每条`trials/trial-N/`保存原配置、实际位置、trace、qpos/qvel/ctrl、来源指纹及接触回放审计。
- `gpu/group-*`及`gpu/trial-*-group-*`为每个真实抓取状态的8环境检查；重复物理组的不同抓取姿态使用独立trial路径。
- `gpu/capacity-trial-6-group-3-close-*`为纸袋胶囊角部闭合状态的主容量曲线；`gpu/capacity-close-*`为较早的PE可压缩诊断曲线，存在外露，不能计物理有效容量。所有窗口100步、dt=100µs、3次重复，固定实际驱动；没有成功承重保持窗口。
- `mixed/`为9个实际状态批次、7个独立物理模板同时驻留的混合容量。包含几何失败来源，数值容量与物理有效容量分别看待。
- `loaded.json`只证明模型完成加载；`samples.json`/`result.json`证明实际计时及检查。OOM日志保留；有限细化后只给已测上下界，不把其间未测规模推为通过。
- `parameters-*.npz`保存每环境实际质量、完整惯量/COM、摩擦及软参数；`actual-material.npz`保存膜和自定义力回读；`effective-friction-*.npz`保存接触实际系数和指尖法向载荷。原生priority会覆盖部分摩擦采样，见主题报告。
- [三类实际失败视频及解码核对](video/verification.json)：[纸袋](video/kraft.mp4)、[塑料薄膜袋](video/pe.mp4)、[气泡信封袋](video/bubble.mp4)。

## 复现

从本工作树根目录执行，输出必须选择新目录。下例直接复测已有真实闭合状态：

```sh
PYTHONPATH=test/bag_model:experience:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -u -B \
  test/bag_model/physical_parcel_yam_benchmark.py benchmark \
  --trial test/runs/2026-11-16/bag/yam-gpu-randomized-ubun-r2/trials/trial-6 \
  --phase close --worlds 512 --steps 100 \
  --output test/runs/2026-11-17/bag/yam-gpu-replay-ubun/close-512
```

完整重做准备与调度：

```sh
export PYTHONPATH=test/bag_model:experience:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN=test/runs/2026-11-17/bag/yam-gpu-replay-ubun
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -u -B \
  test/bag_model/physical_parcel_yam_benchmark.py prepare \
  --configs-from test/runs/2026-11-16/bag/yam-gpu-randomized-ubun-r1/preparation \
  --settle-seconds .5 --support-margin .04 --seed 20261116 --output "$RUN/preparation"
cp test/runs/2026-11-16/bag/yam-gpu-randomized-ubun-r2/schedule.json "$RUN/schedule.json"
flock /tmp/claude-gpu.lock .venv-physical-parcel/bin/python -u -B \
  test/bag_model/run_physical_parcel_gpu_suite.py --run "$RUN" --gpu-trial-indices 0 1 2
```

`commands.jsonl`保存实际子命令与机器路径。r2复用r1已保存的18组配置，故不能只凭相同seed的新准备推断参数完全相同；用`--configs-from`可重现本轮物理组。没有通过的准备/初态仍按原阈值拒绝。完整闭环端到端attempt/s与短窗口environment-step/s分开报告；CPU/GPU小批负载不同，不据此计算同配置加速比。

仅加载边界可独立复核（先按前述完整流程生成新run，或对本轮目录恢复已有记录）：

```sh
PYTHONPATH=test/bag_model:experience:. flock /tmp/claude-gpu.lock \
 .venv-physical-parcel/bin/python -u -B test/bag_model/run_physical_parcel_gpu_suite.py \
 --run test/runs/2026-11-16/bag/yam-gpu-randomized-ubun-r2 --load-only-capacity
```

[加载扫描](load-capacity-summary.json) · [完整环境](environment.json) · [源码与参数回读核对](integrity.json)。所有物理工作完成于2026-11-17，日志保留实际结束时间；未运行承重保持，结果不能当训练场景物理验收。

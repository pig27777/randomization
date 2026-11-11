# 2026-11-10 测试记录

## 内容物 v1.3 GPU：A0小试与既有G3 E4复核

- 对象/版本：冻结分段CPU v1.3为参考，MJWarp3.17/Warp1.19 GPU候选；模型/驱动源码指纹分别见各运行metadata/source_audit，未改冻结包。
- 机器：`ubun`，RTX5090；环境`/home/ubuntu/Dev/Envs/mjwarp`，MuJoCo3.17。A0于2026-11-10 01:35:31（UTC+8）开始，01:39左右结束；跨UTC日期但归本地启动日。
- 问题/标准：A0只核对2个不同κ单独与同批类别全同、无确认失败/溢出；G3 E4要求类别≥95%、硬物力峰中位差≤10%。不改变任务/阈值；未过即停。
- 状态：单元3/3、A0局部2/2通过；既有正式E4复核85/90=94.44%未过，硬物力峰差0.00014372%通过。停止，未继续E5全50/G4/G5及加速扫描。
- 报告：[主题报告](../../content_model/REPORT.md) · [任务检查](../../../reports/content-model-gpu-v1_3-test.md)。
- 数据：[A0小试](contents/v13-a0/smoke/summary.json) · [E4复核](contents/v13-a0/e4-audit/summary.json) · [5项反例及原始输入](contents/v13-a0/e4-audit/counterexamples.json) · [检查审计](contents/v13-a0/checks.json)。原G3元数据缺hostname，结果只用于类别核对，不与本机速度比较。

仓库根目录执行（结果目录拒绝覆盖；复跑需新目录）：

```sh
PYTHONPATH=contents/segment:test/content_model /home/ubuntu/Dev/Envs/mjwarp/bin/python -B -m unittest discover -s test/content_model/tests -p test_mjwarp_batching.py -t test/content_model -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 flock /tmp/claude-gpu.lock /home/ubuntu/Dev/Envs/mjwarp/bin/python -B -u test/content_model/validate_tasks_v13.py a0-smoke --directory test/runs/2026-11-10/contents/v13-a0/smoke > test/runs/2026-11-10/contents/v13-a0/smoke.log 2>&1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 flock /tmp/claude-gpu.lock /home/ubuntu/Dev/Envs/mjwarp/bin/python -B -u test/content_model/validate_tasks_v13.py e4-audit --directory test/runs/2026-11-10/contents/v13-a0/e4-audit --e4-reuse contents/segment/results/warp_backend_v1_3/resume_20261108/g3/E4_raw_gpu.json --cpu-reference-reuse contents/segment/results/warp_backend_v1_3/resume_20261108/g3/supplemental_cpu_references.json > test/runs/2026-11-10/contents/v13-a0/e4-audit.log 2>&1
```

最后一条为离线读取正式旧数据，不运行物理；退出码2是门槛未过。小试只包含4个环境次，不是完整E5，也不作训练吞吐结论。各自原始JSON、日志、上下文/源码审计保留，无覆盖旧记录。

## 袋子表示探索：松弛绳最小演示（slack-rope-demo-r1）

- 对象/版本：新的探索表示（3段铰接盒+6根只受拉的绳+mocap夹持点），不属于现有袋子或内容物版本；v1.3易弯锚点只作测速对照。脚本 `test/bag_model/slack_rope_demo.py`。
- 机器：`ubun`，CPU MuJoCo 3.17单线程；22:37:28–22:37:35。不排GPU队列，不作结论。
- 问题：同一个模板只改松弛量和夹持位置，能否得到满装/松装、居中/偏心的悬挂行为；MJWarp能否按世界设置绳长上限。
- 状态：8例完成且无数值异常。松弛控制离地时机符合预期；摆动几乎不衰减、偏心倾斜39–50°、易弯下垂由绳几何决定，这三点不合预期。MJWarp按世界设置`tendon_range`可行，`tendon_limited`不可。
- 报告：[运行报告](bag/slack-rope-demo-r1/REPORT.md) · [任务检查](../../../reports/slack-rope-demo-test.md)；数据：[summary](bag/slack-rope-demo-r1/summary.json) · [总览视频](bag/slack-rope-demo-r1/overview_2x4.webm)。

## 袋子表示探索：四部分软包最小演示（four-part-pack-demo-r1）

- 对象/版本：探索模板（4×2段内容物带压板、摩擦表面、8根松弛绳、手掌+双指和夹持判定规则），不属于现有袋子或内容物版本，不接入v1.3。脚本 `test/bag_model/four_part_pack_demo.py`。
- 机器：`ubun`，CPU MuJoCo 3.17单线程；23:44:30–23:46:55。运行时有其他负载，耗时只作量级。不排GPU队列，不作结论。
- 问题：同一模板能否产生“捏不到 / 只捏到膜 / 捏到内容物 / 超载松脱 / 偏心 / 捏袋边”这些行为；3倍弯曲刚度下的下垂；松弛时绳阻尼的影响；绳穿过内容物的影响。
- 状态：10例完成，无数值异常。第1、2、5、6、7例符合预期；第3例因压板行程等于聚拢阈值而捏不到；第4例走connect，没有用绳；松弛的绳的阻尼会拖动没捏到的内容物（67–108 mm）；摆动不衰减；第8例远端着桌。未调参。
- 报告：[运行报告](bag/four-part-pack-demo-r1/REPORT.md) · [任务检查](../../../reports/four-part-pack-demo-test.md)；数据：[summary](bag/four-part-pack-demo-r1/summary.json) · [总览视频](bag/four-part-pack-demo-r1/overview_2x4.webm)。

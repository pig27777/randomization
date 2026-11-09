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

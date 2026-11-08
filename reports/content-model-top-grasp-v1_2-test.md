# 内容物 v1.2 顶面抓检查

最近检查：2026-11-05，T1正式回归未通过，停止后续抓取验收、未打标签；独立论文400物体/800测量已完成。

## 范围与方法

基于 SHA256 已核对的冻结 v1 长方体，局部上层细分、质量/COM/惯量守恒、原生局部平移连接、R3/RJ硬物80mm范围。形状暂停并归档，原v1包/表/标签不改。T1正式比较三锚点R1y/z、R2、R3各一次，10%门槛不变；未通过时不进入T2–T6。论文数据独立使用冻结v1，R3/RJ各200个，记录有效/下限/失败和配对剔除。

## 探索检查（不作正式验收或速度数据）

- `python -B -m unittest discover -s tests -v`：27/27通过，16.096s。质量矩、薄三层正质量边界、原生表层恢复/力累积、硬物范围和原24项检查均通过。[日志](../contents/segment/results/top_grasp_v1_2/unit_smoke.log)。
- 易弯物R3单例无确认失败，但滑移约2.348mm，高于v1约0.738mm；仅确认入口和耗时，待持锁正式T1确认。[记录](../contents/segment/results/top_grasp_v1_2/surface_r3_smoke/summary.json)。
- 进程内提高表层剪切比至10的原因分离探索没有改善易弯物滑移（约7.053mm），没有采纳进活动源码；压缩性锚点亦保留原始数据。[记录](../contents/segment/results/top_grasp_v1_2/shear_probe/)。
- 冻结v1论文试跑6物体12次测量完成、无执行异常；包含原v1 R1限位失败及刚度下限，正式分布不据此作结论。[试跑](../contents/segment/results/top_grasp_v1_2/paper_smoke/summary.json)。

正式命令：持有 `flock /tmp/claude-gpu.lock` 后完整单元检查 → `validate_surface.py --reference` → 候选T1 → `response_distribution.py --count 200 --workers 8`。论文部分在T1失败时仍独立完成；T2–T6未运行。

[主验收报告](../test/content_model/REPORT.md)；详细结果在 `contents/segment/results/top_grasp_v1_2/`。未安装包、修改共享库、其他线代码或改动协议。

## 正式 T1 与停止边界

- `flock /tmp/claude-gpu.lock /bin/bash -c ...` 中顺序执行完整单元检查及基线/候选比较。27/27通过，29.430s；基线与候选各12次，均无确认仿真失败、无执行异常。
- 质量误差≤4.39e-15kg、COM误差≤2.98e-17m、惯量相对误差≤1.15e-15；硬物4读数完全一致（R1仍为下限），薄三层与非零COM均有单元覆盖。
- **T1失败**：仅7/12记录在10%内，易弯R1y +14.20%、R3滑移0.738058→2.348155mm；易压R1z −14.90%、R3滑移−11.35%。易压R2掉落/碰桌、承重比0，1.143只是无效诊断读数，不能作为下垂响应。
- **T2–T5未运行**，完整顶面抓执行器尚未实现，没有抓取成功区域、随机稳定率或单环境速度结论。T6硬物80mm范围由正式单元检查覆盖（200个R3与RJ配对），不构成整体v1.2通过。
- [汇总](../contents/segment/results/top_grasp_v1_2/acceptance.json)、[正式日志](../contents/segment/results/top_grasp_v1_2/unit_formal.log)、[基线](../contents/segment/results/top_grasp_v1_2/t1_reference_formal/summary.json)、[候选](../contents/segment/results/top_grasp_v1_2/t1_candidate_formal/summary.json)、[候选源码指纹](../contents/segment/results/top_grasp_v1_2/candidate_source.json)。候选全部源码与正式元数据一致；v1冻结包、标签及换算表保持不变。

本候选采用旋转随粗格的局部平移表层，尚无实测标定；需要分离接触片、粗格质量/碰撞分配及连接效应，不能只提高剪切刚度或改变测法/门槛。I-32已交总体判断。论文测量独立使用冻结v1，不受本候选影响。

## 论文分布正式结果与静态交付核对

- 持锁运行冻结v1的R3/RJ各200，硬/易弯/易压各67/67/66；400几何/质量记录两两配对（200对），800次测量完成，无执行异常。
- R3/RJ确认失败分别7/200、10/200，18次失败测量均有完整状态证据；这是R1y/R2工况，不能替代冻结v1静置+R3的1/200。
- R3两类软物R1精确测量32/67、33/66，R2有效59/67、53/66；RJ分别30/67、40/66及62/67、52/66。两类软物联合有效R3 57/133、RJ 58/133，不能概括R3全面优于RJ。
- 硬物R1全为下限标记，有限正下限64/67；R2端面读数包含刚体姿态，不能当材料弯曲。独立边际与双方共同有效的配对数据分别提供。
- [分布/配对数据](../contents/segment/results/top_grasp_v1_2/paper_analysis/distribution.json) · [400行CSV](../contents/segment/results/top_grasp_v1_2/paper_analysis/responses.csv) · [全部原始数据](../contents/segment/results/top_grasp_v1_2/paper_distribution_formal/summary.json)。导出命令：`python -B analyze_top_grasp.py distribution`。

最终结论：T1未通过；T2–T5未运行；T6范围单元通过；未打v1.2标签。论文数据独立完成，已在流水告知论文线；详细分布、失败和限制见主报告。未改阈值/协议，未安装包或改其他线代码。

交付静态核对（2026-11-05T22:22:59+08:00）：v1原包/表、22个冻结源码、27个候选源码全部匹配；400逐例记录与汇总一致、200配对一致、CSV400行、18次失败测量均有状态；本节与主报告22个本地链接均存在。[核对记录](../contents/segment/results/top_grasp_v1_2/delivery_check.json)。本线正式链退出码0并释放锁。

# 接触修复测试报告

最近测试：2026-11-03 16:21（Asia/Taipei）。Newton 1.7.0 / Warp 1.19，RTX 5090，解释器 `/home/ubuntu/Dev/Envs/newton_latest/bin/python`。

## 结论与范围

**平面压头的覆盖区积分接触，在 1 ms、20 次迭代、8 秒静置协议下，压板及两种手指均单次通过原验收。完整平台仍未通过**：压板粗细网格下陷变化 6.56%；膜片摩擦抖动、圆角夹取的受力/静止问题仍在。没有进入内容物新本构或装袋。

第 1 项已执行修复、对照和测量改进；这里只确认有限范围的进展，不能记作全部接触问题已解决。第 2 项设计见 [压密本构方案](../../contents/newton/compaction_model.md)，尚未实现。

## 实现

- [planar_contact.py](planar_contact.py)：每步裁剪软块顶面三角形与工具投影的交集，每片三个积分点，保留真实重心分力与力矩；ke/kd 以面积除以固定 `0.025² m²` 加权。替换重复的原生软接触记录，仍使用 VBD。仅支持本基准的平面竖直工具，默认不开启。
- [contact_fixes.py](contact_fixes.py)：摩擦零滑移处保留连续极限刚度，覆盖刚柔、软软路径，进程内启用、不改共享安装。单位检查通过，但没有解决膜片抖动，默认关闭。
- [benchmark.py](benchmark.py)：测力保存求解前的刚体历史，避免读取结束后已前移的历史；新增三项可选对照。新结果记录本地修正文件哈希；旧报告和汇总数据未覆盖。
- [grasp.py](../gripper_grasp/grasp.py)：逐子步积分接触力，记录总动量；末段以接触冲量/时间计算承重，并用重力、全部接触力和动量变化交叉检查。该冲量来自末次迭代本构评估，仍需动量残差检验，不能称为精确内部求解冲量。

## 原生对照与积分接触

无摩擦、ke 3000、kd 3，软块材料及原验收阈值不变。表中最大速度沿用原报告的末段采样最大速度均值。

| 运行 | 步长 ms / 迭代 / 仿真秒 | 支撑误差 | 残差 | 最大速度 mm/s | 原判据 |
|---|---:|---:|---:|---:|---:|---|
| [repair_native_plate](results/repair_native_plate.json) | 1 / 10 / 2 | 17.546% | 6.286% | 29.257 | 未通过 |
| [repair_native_finger](results/repair_native_finger.json) | 1 / 10 / 2 | 0.085% | 0.154% | 4.944 | 未通过 |
| [repair_native_finger_off](results/repair_native_finger_off.json) | 1 / 10 / 2 | 40.782% | 53.965% | 520.794 | 未通过 |
| [repair_area_iter20_plate](results/repair_area_iter20_plate.json) | 1 / 20 / 8 | 0.006% | 0.005% | 0.024 | 通过 |
| [repair_area_iter20_finger](results/repair_area_iter20_finger.json) | 1 / 20 / 8 | 0.021% | 0.039% | 0.273 | 通过 |
| [repair_area_iter20_finger_off](results/repair_area_iter20_finger_off.json) | 1 / 20 / 8 | 0.001% | 0.036% | 0.018 | 通过 |
| [repair_area_half_plate](results/repair_area_half_plate.json) | 0.5 / 10 / 8 | 0.024% | 0.009% | 0.051 | 通过 |
| [repair_area_half_settled_finger](results/repair_area_half_settled_finger.json) | 0.5 / 10 / 15 | 0.114% | 0.068% | 0.043 | 通过 |
| [repair_area_half_finger_off](results/repair_area_half_finger_off.json) | 0.5 / 10 / 8 | 0.004% | 0.068% | 0.040 | 通过 |
| [repair_area_fine_plate](results/repair_area_fine_plate.json) | 1 / 10 / 2 | 0.030% | 0.044% | 0.893 | 通过 |

原始 2 秒积分接触：压板通过，节点上/节点间手指仍未静止；节点间手指延长到 5 秒通过，节点上手指 8 秒通过。减半步长时，节点上手指 8 秒仍未通过（最大速度 1.64 mm/s、漂移 0.069 mm），15 秒才通过。**延长的是仿真静置时间，未放宽阈值，也不能宣称原 2 秒全部通过。** 早期失败均保留在 `results/repair_area*.json`。

步长减半后充分静置的下陷与 1 ms/20 次迭代相比，压板约 0.02%、节点上手指约 0.10%、节点间手指约 0.002%。但粗细网格压板仍为 3.939821 / 4.198167 mm（6.56%），超过 5%；细网格单次通过接触判据不等于网格收敛。

## 诊断与未解决项

- 形心补丁：[diagnose_centroid.py](diagnose_centroid.py) 记录 12 个子步；压板有 37 次形心条件切换、1751 次激活切换，节点间手指均为 0。结果在 `results/repair_centroid_audit_*.json`。不能据此认定全部压板回归只由条件切换造成；“共享接触三倍过冲”仍未证实。
- 摩擦修正：[repair_zero_membrane](results/repair_zero_membrane.json) 的支撑误差 0.936%、残差 0.597%、最大速度 4.59 mm/s，静止不通过。只跑一次，不能证明微小差异来自修正。
- 求解器内逐迭代重检测：`repair_iter_*` 三项均未通过，未作为默认修法。

夹取使用 0.25 ms、硬海绵、偏移 5 mm，全部沿用原侧夹动作。默认 epsilon 对照为直角指垫；两项小 epsilon 对照为 4 mm 圆角并启用零速修正，因此不能把两者差异仅归因于 epsilon。小 epsilon 的 10/40 次迭代彼此为同条件对照。

| 运行 | epsilon / 迭代 | 承重/重量 | 动量残差 | 下滑 mm/s | 结果 |
|---|---:|---:|---:|---:|---|
| [repair_impulse_control_firm_off5](../gripper_grasp/results/repair_impulse_control_firm_off5.json) | 0.01 / 10 | 0.997 | 0.35% | 0.739 | 未通过 |
| [repair_impulse_firm_off5](../gripper_grasp/results/repair_impulse_firm_off5.json) | 0.001 / 10 | 0.659 | 34.12% | 0.184 | 未通过 |
| [repair_impulse_iter40_firm_off5](../gripper_grasp/results/repair_impulse_iter40_firm_off5.json) | 0.001 / 40 | 0.934 | 6.62% | 0.202 | 未通过 |

小 epsilon 在逐子步积分后仍有明显受力不平衡；提高迭代有改善，但 40 次仍超 5%，静止和下滑也未通过。保留原瞬时承重字段供对照，新增的 `lift_force_over_weight` 使用积分值；未以整体重量倒推出“正确接触力”。

## 检查与复现

- `python -B -m unittest discover -s test/contact_transfer -p 'test_*.py' -v`：6 项通过。含原测量工具、零滑移力导数、实际接触调用和积分覆盖区面积/一二阶矩检查。最终日志：[repair_unit_tests.txt](results/repair_unit_tests.txt)。
- 静态检查：7 份 Python 语法、7 份 Markdown 格式/本地链接、3 份保存配置的新开关重放检查通过。首轮静态检查直接读取早期结果中尚不存在的新键而失败；改为按实际重放逻辑将缺失可选键视作关闭后通过，未改原结果。
- 上表各 JSON 保存解释器、配置和实际步长；`run_suite.command` 已支持三项新开关，可按保存配置重放。`--tag` 换新名字，保留本轮结果。
- 平压：`python -B test/contact_transfer/benchmark.py --cases plate finger finger_off --dt .001 --iterations 20 --planar-quadrature --seconds 8 --tag replay_area`。
- 半步节点上手指：同上改 `--cases finger --dt .0005 --iterations 10 --seconds 15`。
- 网格：`--cases plate --dt .001 --iterations 10 --seconds 2 --cells 24 20 8 --planar-quadrature`。
- 摩擦：`--cases membrane --dt .0001 --ke 30 --kd .03 --friction .5 --friction-zero-fix --seconds 2`。
- 夹取：`python -B test/gripper_grasp/grasp.py --material firm --offset-mm 5 --dt .00025 --friction-eps .001 --iterations 40 --pad-radius-mm 4 --friction-zero-fix --tag replay_impulse`。

首次新增单位检查曾因 Warp 不支持数组下标多目标赋值而失败；拆分赋值后通过。首次审计调用因 1.7 的方法多一个参数而失败，改为原样转发参数后完成。未删除断言、放宽验收或忽略失败。

## 限制

- 主要修复仅为平面竖直工具的软块顶面接触；任意姿态、工具侧面、圆角夹爪、袋子尚未适配，现有业务模型未切换到新接触。
- 尚未达到网格收敛，不能批准内容物本构或装袋整合；新材料仅交付设计。
- 除明确对照外每个配置各一次；没有完整重复性统计。部分 GPU 任务并行，耗时不作吞吐基准。
- 继承原基准每 50 ms 的几何/翻转采样限制，不能证明采样间绝无相交。没有实物标定。

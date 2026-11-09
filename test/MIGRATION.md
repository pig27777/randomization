# 代码迁移表（2026-11-09）

模型目录保留建模实现，独立测试入口迁到对应主题。本表用于把历史报告中的命令对应到当前工作区；历史结果、冻结包、原始记录与快照保持原内容。

| 原路径 | 当前路径 |
| --- | --- |
| `bag_sim/check_newton.py` | [test/bag_model/check_newton.py](../test/bag_model/check_newton.py) |
| `bag_sim/check_unified.py` | [test/bag_model/check_unified.py](../test/bag_model/check_unified.py) |
| `bag_sim/demo_contents.py` | [test/bag_model/demo_contents.py](../test/bag_model/demo_contents.py) |
| `bag_sim/demo_newton.py` | [test/bag_model/demo_newton.py](../test/bag_model/demo_newton.py) |
| `bag_sim/c1_newton.py` | [test/bag_model/c1_newton.py](../test/bag_model/c1_newton.py) |
| `bag_sim/loose_speed.py` | [test/bag_model/loose_speed.py](../test/bag_model/loose_speed.py) |
| `bag_sim/render_demo.py` | [test/bag_model/render_demo.py](../test/bag_model/render_demo.py) |
| `bag_sim/deadline_watch.py` | [test/bag_model/deadline_watch.py](../test/bag_model/deadline_watch.py) |
| `bag_sim/flex_diagnostics.py` | [test/bag_model/flex_diagnostics.py](../test/bag_model/flex_diagnostics.py) |
| `contents/segment/run_round1.py` | [test/content_model/run_round1.py](../test/content_model/run_round1.py) |
| `contents/segment/run_round2.py` | [test/content_model/run_round2.py](../test/content_model/run_round2.py) |
| `contents/segment/mjwarp_speed.py` | [test/content_model/mjwarp_speed.py](../test/content_model/mjwarp_speed.py) |
| `contents/segment/mjwarp_validate.py` | [test/content_model/mjwarp_validate.py](../test/content_model/mjwarp_validate.py) |
| `contents/segment/newton_check.py` | [test/content_model/newton_check.py](../test/content_model/newton_check.py) |
| `contents/segment/analyze_r3_v13.py` | [test/content_model/analyze_r3_v13.py](../test/content_model/analyze_r3_v13.py) |
| `contents/segment/analyze_top_grasp.py` | [test/content_model/analyze_top_grasp.py](../test/content_model/analyze_top_grasp.py) |
| `contents/segment/diagnose_r3_v13.py` | [test/content_model/diagnose_r3_v13.py](../test/content_model/diagnose_r3_v13.py) |
| `bag_sim/round3_flex.py` | [test/bag_model/round3_flex.py](../test/bag_model/round3_flex.py) |
| `bag_sim/round3_effect.py` | [test/bag_model/round3_effect.py](../test/bag_model/round3_effect.py) |
| `bag_sim/flex_batch.py` | [test/bag_model/flex_batch.py](../test/bag_model/flex_batch.py) |
| `bag_sim/effect_batch.py` | [test/bag_model/effect_batch.py](../test/bag_model/effect_batch.py) |
| `bag_sim/loose_mjwarp.py` | [test/bag_model/loose_mjwarp.py](../test/bag_model/loose_mjwarp.py) |
| `bag_sim/soft_joint_speed.py` | `test/bag_model/soft_joint_speed.py`（原本未跟踪，迁移后继续保留本机未提交状态） |
| `bag_sim/soft_flex_speed.py` | `test/bag_model/soft_flex_speed.py`（原本未跟踪，迁移后继续保留本机未提交状态） |
| `bag_sim/effect_control.py` | `test/bag_model/effect_control.py`（原本未跟踪，迁移后继续保留本机未提交状态） |
| `bag_sim/loose_protocols.py` | [test/bag_model/loose_protocols.py](../test/bag_model/loose_protocols.py) |
| `contents/segment/freeze_v1.py` | [test/content_model/freeze_v1.py](../test/content_model/freeze_v1.py) |
| `contents/segment/view_m2.py` | [test/content_model/view_m2.py](../test/content_model/view_m2.py) |
| `contents/segment/validate_pinch.py` | [test/content_model/validate_pinch.py](../test/content_model/validate_pinch.py) |
| `contents/segment/validate_warp.py` | [test/content_model/validate_warp.py](../test/content_model/validate_warp.py) |
| `contents/segment/validate_pinch_v14.py` | `test/content_model/validate_pinch_v14.py`（原本未跟踪，迁移后继续保留本机未提交状态） |
| `contents/segment/validate_tasks_v13.py` | [test/content_model/validate_tasks_v13.py](../test/content_model/validate_tasks_v13.py) |
| `contents/segment/validate_gpu_resume_v13.py` | [test/content_model/validate_gpu_resume_v13.py](../test/content_model/validate_gpu_resume_v13.py) |
| `contents/segment/stability_estimate.py` | [test/content_model/stability_estimate.py](../test/content_model/stability_estimate.py) |
| `contents/segment/validate_warp_v13.py` | [test/content_model/validate_warp_v13.py](../test/content_model/validate_warp_v13.py) |
| `contents/segment/benchmark_gpu_v13.py` | [test/content_model/benchmark_gpu_v13.py](../test/content_model/benchmark_gpu_v13.py) |
| `contents/segment/response_distribution.py` | [test/content_model/response_distribution.py](../test/content_model/response_distribution.py) |
| `contents/segment/validate_acceleration.py` | [test/content_model/validate_acceleration.py](../test/content_model/validate_acceleration.py) |
| `contents/segment/validate_model.py` | [test/content_model/validate_model.py](../test/content_model/validate_model.py) |
| `contents/segment/audit_gpu_v13.py` | `test/content_model/audit_gpu_v13.py`（原本未跟踪，迁移后继续保留本机未提交状态） |
| `contents/segment/diagnose_tasks_v13.py` | `test/content_model/diagnose_tasks_v13.py`（原本未跟踪，迁移后继续保留本机未提交状态） |
| `contents/segment/tests/test_gpu_model_batching.py` | `test/content_model/tests/test_gpu_model_batching.py`（原本未跟踪，迁移后继续保留本机未提交状态） |
| `contents/segment/tests/test_freeze_interfaces.py` | [test/content_model/tests/test_freeze_interfaces.py](../test/content_model/tests/test_freeze_interfaces.py) |
| `contents/segment/tests/test_pinch.py` | [test/content_model/tests/test_pinch.py](../test/content_model/tests/test_pinch.py) |
| `contents/segment/tests/test_pinch_v14.py` | `test/content_model/tests/test_pinch_v14.py`（原本未跟踪，迁移后继续保留本机未提交状态） |
| `contents/segment/tests/test_acceptance_entry.py` | [test/content_model/tests/test_acceptance_entry.py](../test/content_model/tests/test_acceptance_entry.py) |
| `contents/segment/tests/test_gpu_continuation.py` | `test/content_model/tests/test_gpu_continuation.py`（原本未跟踪，迁移后继续保留本机未提交状态） |
| `contents/segment/tests/test_bushing.py` | [test/content_model/tests/test_bushing.py](../test/content_model/tests/test_bushing.py) |
| `contents/segment/tests/test_gpu_contact_buffer.py` | `test/content_model/tests/test_gpu_contact_buffer.py`（原本未跟踪，迁移后继续保留本机未提交状态） |
| `contents/segment/tests/test_scene_extensions.py` | [test/content_model/tests/test_scene_extensions.py](../test/content_model/tests/test_scene_extensions.py) |
| `contents/segment/tests/test_sampling.py` | [test/content_model/tests/test_sampling.py](../test/content_model/tests/test_sampling.py) |
| `contents/newton/unified_tests.py` | [test/contact_transfer/unified_tests.py](../test/contact_transfer/unified_tests.py) |
| `contents/newton/ef_normal_probe.py` | [test/contact_transfer/ef_normal_probe.py](../test/contact_transfer/ef_normal_probe.py) |
| `contents/newton/i11_finger_off.py` | [test/contact_transfer/i11_finger_off.py](../test/contact_transfer/i11_finger_off.py) |
| `contents/newton/ef_node_probe.py` | [test/contact_transfer/ef_node_probe.py](../test/contact_transfer/ef_node_probe.py) |
| `contents/newton/rigid_tet_check.py` | [test/contact_transfer/rigid_tet_check.py](../test/contact_transfer/rigid_tet_check.py) |
| `contents/newton/ef_dedup_check.py` | [test/contact_transfer/ef_dedup_check.py](../test/contact_transfer/ef_dedup_check.py) |
| `contents/newton/render_newton.py` | [test/contact_transfer/render_newton.py](../test/contact_transfer/render_newton.py) |
| `contents/newton/ef_centroid_check.py` | [test/contact_transfer/ef_centroid_check.py](../test/contact_transfer/ef_centroid_check.py) |
| `contents/newton/ef_contact_probe.py` | [test/contact_transfer/ef_contact_probe.py](../test/contact_transfer/ef_contact_probe.py) |
| `contents/newton/mesh_check.py` | [test/contact_transfer/mesh_check.py](../test/contact_transfer/mesh_check.py) |
| `contents/particle/` | [contents/legacy/particle/](../contents/legacy/particle/) |
| `contents/rigid/` | [contents/legacy/rigid/](../contents/legacy/rigid/) |

## 模型与执行流程拆分

- `bag_sim/loose_setup.py` 保留可复用的场景构建；原 `loose_protocols.py` 的抓取执行与判定在 `test/bag_model/loose_protocols.py`。
- `bag_sim/film_pinch.py` 保留捏取组件；独立 F7 执行流程在 `test/bag_model/film_grasp.py`。
- `bag_sim/soft_content_backend.py` 校验并加载原 GPU 候选快照，模型不再通过测速脚本加载后端。
- 冻结归档继续按原标签与源码清单逐项校验；搬迁后的当前测试驱动单独记录指纹，不能当作原冻结源码。

模型和历史输入位置保持不变；新轮次使用 [按日期、对象管理的运行目录](runs/README.md)。不提供同名旧脚本副本，避免出现两个维护入口。

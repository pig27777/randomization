# Yam_B 纯仿真任务检查

最近检查 2026-11-25 20:12，北京时间，机器 `ubun`。 [主题报告](../test/yam/REPORT.md) · [参数与接口](../experience/YAM_PURE_SIM.md)

| 范围 | 方法与结果 |
| --- | --- |
| 资产、SDK、关节顺序与末端 | `test/yam/smoke.py asset` 与 `test/yam/check_sdk.py`。本地资产、指垫和 FK 雅可比通过。官方标准 YAM v1 的关节方向与限位一致，J2–J5 轴线和工具坐标存在版本差异。 |
| 空载、接触、夹爪与抓放 | 持锁运行 `test/yam/smoke.py tracking/contact/grasp`。均通过，原生碰撞实提实放。名义仿真 28 维和实际 actuator force 对齐。 |
| GPU 批量与随机化 | 持锁运行 `test/yam/smoke.py gpu`，1、16、64、256 路各一遍 nominal、DR。均通过。256 路 nominal 约 23,566 环境步/s，684 MiB，零数值故障。 |
| ABC 场景 | `test/yam/smoke.py abc` 通过基座、桌面、顶相机和腕相机回读。停驻 Yam_A 保留碰撞，Yam_B 完成抓放。 |
| PE 三路接入 | 持锁运行 `test/yam/bag_try.py`。原深程发现直触，高程未触膜。最终中位接触峰值 2.33 N，左右两路未触膜，三路验收失败。整袋夹持等待稳定的袋体前置条件。 |
| 旧平台回归 | `experience/tests/check_migration.py`。三个用例各 25 步状态差均为 0，33 个带清单文件和兼容入口检查通过。 |
| 静态与制品 | Python AST、JSON 严格解析及 `git diff --check`。录像用原有平台渲染，FFmpeg/ffprobe 检查可解码。 |

GPU 首轮发现约束行容量和测试输入不符，修复后通过。硬件通信、真机标定和策略训练不在本轮验证范围内。

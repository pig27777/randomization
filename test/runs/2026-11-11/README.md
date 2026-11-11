# 2026-11-11 测试记录

## 袋子表示探索：四部分软包水平捏侧面（side-pinch-demo-r2）

- 对象/版本：四部分探索模板（4×2段内容物＋被捏侧面压板替身、摩擦表面、20根只受拉的绳、水平手掌＋双指和夹持判定），不属于现有袋子或内容物版本，未接入v1.3（压板替身，理由见报告）。脚本 `test/bag_model/four_part_pack_demo.py --set side`（sha256 `dc594a8f…`）。
- 机器：`ubun`，CPU MuJoCo 3.17单线程；20:38:08–20:41:29（UTC+8，含r1顶面10例回归）。不排GPU队列，不作结论。
- 问题：先改r1的三处（绳阻尼置0、绳总激活＋压入超临界再连内容物、侧面可压入），再看6个水平捏侧面情况的行为是否说得通。规则初值不变，不调参。
- 性质：**无臂夹爪预检，不作结论**；本轮结论须在统一YAM平台上运行，本机缺YAM资产与`.venv`，受阻未运行。已写适配器`test/bag_model/side_pinch_adapter.py`（v1.3本体、20根绳、新规则、捏住阻尼和合拢消耗松弛量开关），静态检查8/8通过；共享平台的“可选接近方向”只写了改动说明，未落地，见报告YAM节。
- 状态：6例＋顶面10例完成，无数值异常。空手不拖动（r1为67–108 mm，现为0）。第4例捏不到，第2、3、5例吊起。两例硬物（1、6）绳超载、没离地：两连接点竖直分布，轮廓高度方向又没有松弛，被捏面被锁成竖直。吊起的情况被捏面只转到45–82°。第3、5例内容物连接的超载主要来自指尖与压板的内部对顶。
- 报告：[运行报告](bag/side-pinch-demo-r2/REPORT.md) · [任务检查](../../../reports/side-pinch-demo-test.md)；数据：[summary](bag/side-pinch-demo-r2/summary.json) · [顶面回归](bag/side-pinch-demo-r2/top-r1-cases/summary.json) · [总览视频](bag/side-pinch-demo-r2/overview_3x2.webm)。

- 规则修正后重跑（预检，21:01–21:04）：松弛按薄膜圈分配，捏住阻尼和合拢消耗松弛量默认打开。硬物第1例离地，第6例竖吊（远端着桌），被捏面朝上80–84°，夹持点到表面约20 mm（只按圈分配时约41 mm），摆动0.1–0.4 mm（只按圈分配时34.5 mm）。不开阻尼时第1例在摆入末端被绳拽停、超载。[报告节](bag/side-pinch-demo-r2/REPORT.md) · [数据](bag/side-pinch-demo-r2/loop-slack-defaults/summary.json) · [视频](bag/side-pinch-demo-r2/loop-slack-defaults/overview_3x2.webm)。
- 薄膜抗弯刚度对照（预检，21:07–21:09）：撤销“被捏面朝上”这一预期，改为报告倾角。第1例 k=0 时下垂80–84°（接近塑料袋尖顶吊着），k=0.81 N·m/rad（按托住30°估）时下垂31–32°（接近纸袋部分下垂）。松弛减半后夹持点到表面从约20 mm降到12–14 mm，倾角几乎不变。4次均未超载；悬挂时指尖与内容物无接触。[报告](bag/side-pinch-demo-r2/REPORT.md) · [数据](bag/side-pinch-demo-r2/film-bend/)。

```sh
MUJOCO_GL=egl OMP_NUM_THREADS=1 FFMPEG=/home/ubuntu/Dev/Envs/miniforge3/envs/robocasa/bin/ffmpeg /home/ubuntu/Dev/Envs/mjwarp/bin/python -B test/bag_model/four_part_pack_demo.py --set side --output test/runs/2026-11-11/bag/side-pinch-demo-r2
MUJOCO_GL=egl OMP_NUM_THREADS=1 /home/ubuntu/Dev/Envs/mjwarp/bin/python -B test/bag_model/four_part_pack_demo.py --set top --no-video --output test/runs/2026-11-11/bag/side-pinch-demo-r2/top-r1-cases
```

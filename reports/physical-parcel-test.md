# 双膜片物理包裹任务检查报告

最近整理时间为2026-11-15 17:01（Asia/Taipei），机器ubun。用户要求停止测试，本轮结束，整体物理验收尚未通过。

**后续复核更正** 用户指出画面穿透后，整面几何检查发现旧初始化有盒角外露和真实相交。下文支撑与GPU数据保留为旧构型记录，原几何防穿透通过结论撤回。初始化、反力测量刷新及录像的后续结果见 [YAM录像检查报告](physical-parcel-yam-video-test.md)。

## 范围与结果

- 复用原 `kraft_coarse_35.npz`，200顶点、396三角形，建立MuJoCo/MJWarp 4.0.1原型。源码入口为 [physical_parcel.py](../bag_sim/physical_parcel.py) 和 [physical_parcel_gpu.py](../bag_sim/physical_parcel_gpu.py)。
- `physical_parcel_checks.py` 已执行audit、support、contacts、pull、slack、compliance、conservation、capture；各轮输出在 [当日运行目录](../test/runs/2026-11-15/bag/)。
- `python -B -m unittest discover -s test/bag_model -p test_physical_parcel.py -v` 实际7/7通过。
- 底膜支撑误差0.0788%，稳定段穿透0.01674 mm，裸物—桌面接触0。膜节点外力提拉68.858 mm；松紧外力对照均完成。
- 六向探针16/18记录反力，box左/右为0，部分探针穿透超1 mm。捕获夹具目标例未记录指尖反力，承重、渐进滑移及脱落未验证。
- 默认25 μs的角动量相对漂移4.281%，超过1%标准；6.25 μs复核为0.7301%。线动量及耗散检查通过相应短窗。
- YAM统一平台仅完成2 ms入口检查，完整抓取未运行。B2/B3的GPU批次、完整材料比较及长期动力学未验证。压板限位参数后续修订未动态复验。
- GPU命令使用 `physical_parcel_benchmark.py --worlds ... --steps 50 --repeats 3 --initial-state ...`。8192环境三次完成，9032.73环境步/s、每批step907.904 ms、采样显存29400 MiB，原始overflow全0、无非有限值。9216在CUDA Graph执行实例创建时OOM。8192至9216之间的精确边界未测。
- 每次GPU计时仅覆盖1.25 ms仿真，单刚体、静置状态。512环境逐核profile中，稀疏分块Cholesky占79.52%。
- 旧1024/4096独立性检查误将偶然重复的float32质量值判为不独立，检查已修正，8192核对实际联合参数通过。早期原生弯曲残余力、显式大步长发散、YAM入口失败与修复记录均保留。

所有正式运行持共享GPU锁。当前材料采用未经实物标定的薄膜预设，永久塑性、记忆和撕裂尚未实现。用户要求停止后，只整理已有记录与文档。

完整方程、参数、功能支持、逐轮失败、命令和原始数据链接见 [主题报告](../test/bag_model/PHYSICAL_PARCEL_REPORT.md)。

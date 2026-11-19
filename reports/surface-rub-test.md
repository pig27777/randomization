# 当前整袋表面搓动与褶皱检查

最近检查：2026-11-19T21:44:41.681995+08:00，ubun，MuJoCo4.0.1，全部CPU，按用户要求不持GPU锁。

**结论：本工况能建立摩擦，但没有形成明显的搓动褶皱。** 仅适用于原142节点网格、350×250 mm袋、330×230×6 mm内容物及每指0.5/2 N、5/10 mm搓动。

| 范围 | 方法/命令 | 结果 |
| --- | --- | --- |
| 短入口 | surface_rub.py --smoke | 通过；2个物理步，仅入口检查 |
| 法向加载 | 完整整袋、共享YAM圆角指尖滑轨，实际速度进入接触；mj_contactForce与膜DOF Jᵀλ比对 | 7组有效对照均通过；误差最大1.99e-15 N；末30 ms全部力采样在原±20%带内 |
| 0.5 N/指搓动 | 按压0 mm、μ=.8搓5/10 mm、μ=0搓10 mm，各1次 | 4/4完成，无新折脊；有摩擦实际切向力约0.36–0.38 N/指 |
| 2 N/指搓动 | 仅按压、μ=.8搓10 mm、μ=0搓10 mm，各1次，5 ms力反馈滤波 | 3/3完成；摩擦约1.13 N/指；局部材料出现轻微压缩但未成折脊 |
| 几何/数值 | 原生contact及独立全膜—盒几何检查 | 峰值原生穿透0.1363 mm，几何深度0，无warning、全有限，辅助材料连接0；真实退出后接触力0 |
| 材料聚拢 | 相同材料节点/行长、弦高、同片二面角及主应变 | 节点位移约0.14–0.22 mm；邻点局部凸起最高0.178 mm，未过本轮0.5 mm可见折脊规则 |
| CPU慢放 | render_surface_rub.py及FFmpeg完整解码/ffprobe/抽帧 | 1280×580/25fps，5倍慢放，摩擦与零摩擦并排；无几何夸张 |
| 代码检查 | ast.parse surface_rub.py/render_surface_rub.py，git diff --check | 通过 |

完整复现参数与环境见[commands.json](../test/runs/2026-11-19/bag/flat-bag-surface-rub-r1/commands.json)；输出目录须选新目录。主要命令示例：

```sh
env PYTHONPATH=test/bag_model:experience:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 .venv-physical-parcel/bin/python -B -u test/bag_model/surface_rub.py \
 --config test/runs/2026-11-19/bag/folded-clothes-6mm-r1/config.json \
 --distance-mm 10 --press-force 2 --press-seconds 1.5 --normal-gain .01 --force-filter .005 \
 --output test/runs/2026-11-19/bag/flat-bag-surface-rub-replay
```

加载标定失败保留：400 ms下压只达到约0.085 N/指，原因是约12.5 mm膜—物间隙；延长控制后0.5 N有效。2 N原高增益及未滤波控制存在振荡，未作为恒压横搓结论；加5 ms滤波后通过更严格的压力带采样要求。未降低压力、穿透或成皱门槛。初版多层字幕缺字，改单层字幕后重新导出并目检，未改变轨迹。

精确滑移积分的探索性结果已标记未验证并排除；材料位移以实际节点位置为准。35 mm网格不能解析毫米级皱纹；本轮不证明真实袋子不起皱。材料、参考拓扑、自碰撞、膜—物摩擦未调参，未做半步长/网格/GPU或真实织物验收。原始contact和完整trace已gzip无损归档并校验原字节SHA；无效/过时加载标定的重复MJB与轨迹清理，完整力/几何trace、contact、配置、源码和失败结论保留。

[2 N对照视频](../test/runs/2026-11-19/bag/flat-bag-surface-rub-r1/surface-rub-2N.mp4) · [0.5 N对照视频](../test/runs/2026-11-19/bag/flat-bag-surface-rub-r1/surface-rub-05N.mp4) · [定量表](../test/runs/2026-11-19/bag/flat-bag-surface-rub-r1/results.csv) · [主题报告](../test/bag_model/PHYSICAL_PARCEL_REPORT.md)。

归档trace可用`gzip.open("trace.json.gz", "rt")`读取，contact同理；archive.json保存原字节SHA与压缩大小。

最终维护源码/文档的暂存差异检查通过；全量`git diff --cached --check`返回2，仅有两份实际运行model.xml中的6处行尾空格。为保留模型快照原字节，归档文件未重写；没有测试断言被删除或放宽。

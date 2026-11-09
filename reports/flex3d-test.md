# 3D flex与统一抓取平台检查

最近检查：2026-11-09，机器`ubun`，本仓库`.venv`/MuJoCo3.17.0。**统一平台已搭建并运行；当前3D flex候选没有通过可靠抓取。** 用户已将任务从平压改为完整机械臂多角度抓取，并指定后续模型共用此平台。

| 计划/用途 | 执行与结果 |
|---|---|
| T1接触基座 | 用户取消平压；短入口和中断记录保留，不计通过 |
| 原T2摩擦承重 | 原侧夹未做；改为完整YAM角度矩阵14次：4翻转、5抓取失败、5初始化未找到合格姿态，无成功 |
| T3 κ覆盖 | 未运行；不得从一个E/ν候选推广到整个参数域 |
| T4捏取规则迁移 | 已接入原生connect与5 ms/I-37，2项实际连接符号检查通过；R4系数1/2及悬挂角度验收未做 |
| T5运行成本 | 持锁CPU1/28完成：实时率0.1495；28进程0.18119 evaluate/s，全部合拢失败，成功抓取/s为0。存在其他CPU行为作业；未做分段v1.3、MJWarp、插值模型对照 |
| 后续统一平台 | 公共YAM、指尖、动作、判定、记录/视频、速度入口与模型适配器分开；原生flex与刚体入口各20 ms通过。使用约定已写入README及项目AGENTS |
| 用户要求枕头/翻转可见 | 枕头真实体积网格468顶点/1728四面体，0.29 kg；下压5.201 s翻转。整体+局部10倍慢放，翻转红色、末帧停留3 s，已查看末帧 |

验证方法：

```sh
.venv/bin/python -B test/flex3d/run_suite.py behavior --workers 2 --output test/flex3d/results/NEW_RUN
flock /tmp/claude-gpu.lock .venv/bin/python -B test/flex3d/benchmark.py --output test/flex3d/results/NEW_SPEED
.venv/bin/python -B -m unittest discover -s test/flex3d -p test_platform.py -v
node test/flex3d/test_viewer.js test/flex3d/results/20261109_ubun_pillow/bendable45/pillow-tetra-playback.html
```

长方体加密到20 mm仍翻转；dt1→0.5 ms均在约5.04 s采样检出，连接尚未开启。最终反演显示指尖下局部单元已压到原长约1.1%再越过零体积。材料大压缩失稳是主要解释，接触参数/控制影响尚未全部排除。无自定义材料补偿、无环境升级。

完整成功抓取链路尚未在当前模型上通过；刚体只做接口控制。真实浏览器不可用，回放仅做JS交互执行检查；视频CPU渲染和红色标记已实际查看。保留早期协议适配失败及无效参考坐标审计，不用其数据作通过结论。

复用入口：[统一平台说明](../test/flex3d/README.md)。原始矩阵、测速、细网格/半步长、枕头和视频的路径与源码版本见[本轮报告](../test/flex3d/REPORT.md)。后续先解决模型局部压缩与机械臂姿态/跟踪，再在同平台重测，不另建执行器或改变判定来比较模型。

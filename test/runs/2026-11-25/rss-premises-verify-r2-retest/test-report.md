# 10 g Newton 定向复测检查

机器 `ubun`。指定原生环境执行 Newton 回合。CPU Flex 使用项目的 `.venv-physical-parcel`，两者正式运行串行持共享 GPU 锁。输出一回合一目录，保留原始目标、状态与日志。

## 复现方法

从仓库根目录运行。该目录与首轮数据目录须同时可用。

```bash
D=test/runs/2026-11-25/rss-premises-verify-r2-retest
NATIVE=/home/ubuntu/workspace/czx/system0/sim/python.sh
"$D/run.sh" --output "$PWD/$D/repro-original" --cases 1B 1C-empty
"$D/run.sh" --output "$PWD/$D/repro-b-200" --cases 1B --light-through
"$D/run.sh" --output "$PWD/$D/repro-c-150" --cases 1C-150
"$D/run.sh" --output "$PWD/$D/repro-c-250" --cases 1C-250
PYTHONPATH="$PWD" "$NATIVE" -B "$D/diagnose_r1.py"
"$NATIVE" -B "$D/publish.py"
"$NATIVE" -B "$D/verify.py"
"$D/check-frozen.sh"
"$D/flex_cpu.sh" "$PWD/$D/repro-flex-cpu"
```

Flex 的五条命令在 [flex_cpu.sh](flex_cpu.sh)。已有回合目录保留，复现应使用新目录。几何测试构建封闭箱和单个探针三角面，验证 0、1、3 mm 三档仅第三档切入内缩 2 mm 盒。[边界实测](geometry-tests.log)

首个原始 500 g 重放在 1.0 s 以后收到 SIGTERM，进程退出码 143，未取得完整原始轨迹。后按用户轻载要求取消排队的 500／1,000 g 重跑，保留完整 600 g 对照，并启动 200／150／250 g。新回合逐位核对原目标与网格，不改变膜体和接触参数。

## 最终检查结果

2026-11-25 22:07 CST。五个 Newton 回合均通过源目标逐位对比、膜顶点及拓扑读回、质量和步长检查。两次 1B 的直接载荷中止帧均有原生载荷。三档轻载 1C 均完整跑完 4.6 s。CPU Flex 五项通过本地 MVP 判据，五份 50 µs 轨迹均可读。三套单页 PDF 和高分辨率 PNG 通过读回。Python AST、Bash 语法和冻结路径差异均通过。8 卡 GPU 作业无可用本地硬件。

[检查输出](validation.log) · [总报告](retest_verdict.md) · [冻结校验](frozen-check.log)

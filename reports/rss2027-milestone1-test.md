# RSS 2027 Milestone 1 材料包络检查

2026-11-20，北京时间，`ubun`。用户确认冻结 Stage 1 后，从现有归档生成参数配置。Stage 1 尚有聚拢及滑脱的网格验收缺口。[冻结阶段报告](../test/bag_model/RSS2027_STAGE1_REPORT.md)

```sh
.venv-physical-parcel/bin/python -B policy/rss2027/build_stage1_envelopes.py
OPENBLAS_NUM_THREADS=1 .venv-physical-parcel/bin/python -B policy/rss2027/test_stage1_envelopes.py -v
```

3 项测试通过。核对配置可逐字节重建、原始文件的 SHA-256、四个材料族、59 条整袋动作、14 条加载曲线与 128 个几何样本。核对 1.75 倍外扩、SI 单位、四个 9×9 协方差矩阵的对称性和正定性。配置采用 YAML 1.2 兼容的 JSON 表示。

已记录 `[N_envs, 2, 6]` 状态、左右指顺序、六个字段量纲与 1–2 ms 步长。实测同态 β 仅用于 PE 和牛皮纸的局部锚点；目前协方差为工程采样先验，经验协方差留空。脱手外载与摩擦系数分别记录，空间折形斜率和参考膜量斜率分别保留。全局余料上限随实际几何重算。防抖、双侧独立捕获、余料守恒和能量耗散留给 Milestone 2 的代理与测试。[配置](../policy/rss2027/proxy_material_envelopes.yaml)

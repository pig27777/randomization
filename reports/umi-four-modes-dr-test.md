# UMI 四种实物抓取模态的训练 DR 覆盖

2026-11-23，ubun，只读检查。审查 Route B 训练环境的 DomainRandomizer，在完整课程宽度（curriculum_alpha = 1）下按采样公式换算端点。更窄的课程不会补上缺口。长宽几何比较以袋的长轴对 X、短轴对 Y。训练半尺寸换为全尺寸后是长 14–38 cm、宽 10–30 cm、包裹总厚 2–16 cm。衣料厚度和空袋 PE 膜厚度另有物理含义。[训练入口](../src/isaac_yam_parcel/yam_parcel_base_env.py) · [实际采样器](../src/low_order_parcel/domain_randomizer.py) · [半尺寸分布](../configs/payload_template_X.yaml)

| 模态 | 平铺长 × 宽 | 实物厚度对照 | 自由膜颈与姿态覆盖 |
| --- | --- | --- | --- |
| A 松装，仅捏膜 | 39 × 29 cm，长越界 1 cm，宽覆盖 | 衣厚 0.25 cm；如以此近似整包厚度，低于 2 cm 下限。当前无独立衣厚采样。 | 10 cm 是加载后的真实间距，当前缺少按模态采样和验证的竖直自由膜颈区间。 |
| B 同袋，下压夹衣 | 同 A | 同 A | 近零自由膜颈需要膜和衣物同时被夹住。现有 PE 柔索正原长，缩短扭转参数不能证明直接抓衣。 |
| C 厚裙 | 39 × 29 cm，长越界 1 cm，宽覆盖 | 2 cm 在总厚区间下界；连续均匀采样精确取到边界的概率为零。 | 6 cm 是抓点到底面。它包含裙厚和裙子弯曲，不能当作膜颈长度。 |
| D 满装偏心 | 39 × 27 cm，长越界 1 cm，宽覆盖 | 未提供独立厚度，不能逐例判定 | 近零颈同 B。若“满装”映射到 fill_ratio = 1，当前上限 0.95 越界。质量权重可偏心，稳态倾角 35°–45° 仍需实际回放验证。 |

PE 名义单侧柔索 L0 为 6.5 cm。运行时按 L0 × (fill_ratio / 0.75)^(-1/3) 缩放，在 fill_ratio 为 0.25–0.95 时约 6.01–9.37 cm。原长到加载后竖直间距还受张力、伸长、锚点位置与姿态影响。stage3_dr_and_corners.yaml 的 neck_length_m = 2.2–5.5 cm 输入有限指垫扭转模型；该字段不是竖直膜颈或柔索原长的采样开关。A 与 B 在相同平铺尺寸下产生不同抓取，表明所需覆盖也包含接触模态。[PE 名义参数](../configs/proxy_material_envelopes.yaml) · [Stage 3 配置](../configs/stage3_dr_and_corners.yaml) · [扭转模型](../src/low_order_parcel/pad_torsion.py)

**待确认的字段建议**

- configs/payload_template_X.yaml 的 distributions.bag_half_extents：长轴上限 0.190 → 至少 0.195 m；若 A/B 的整包厚确为 2.5 mm，Z 半厚下限 0.010 → 至多 0.00125 m。宽轴现有范围足够。
- configs/stage3_dr_and_corners.yaml 的 fill_ratio 及 configs/payload_template_X.yaml 的 distributions.fill_ratio：若 D 等价于满体积，将上限 0.95 扩到 1.00，并检查零余量下的直接夹衣模态。Route B 训练用前者覆盖初始抽样。
- 需要单独覆盖 A/B/D 自由颈时，建议新增按抓取模态选择的自由膜颈标定目标或柔索原长分布，保留直接夹衣的零颈分支。将扭转长度设为 0 会触发已有的正长度校验；只扩大 PE 名义 L0 无法代表两种抓法。

以上字段均未修改。检查采用 YAML 中的 SI 单位、训练入口的字段覆盖顺序及解析边界，无需 GPU 抽样。

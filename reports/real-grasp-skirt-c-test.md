# UMI 厚裙 C 居中摩擦抓取直出

ubun，2026-11-23。按最新要求直接出回放，未执行四连图验收或全片几何门禁。两侧同用 2 ms 的 Yam_B 位置边界，250 g 的 39 × 29 cm PE 袋与厚裙。居中并垂直下压，参考开口 78 mm，末端 9 mm，指垫底面最低约 8.03 mm。手腕从提拉阶段开始俯转以保留 IK 可达性，指垫中点沿固定 XY 上升。

左侧 MuJoCo 4 以两块具有真实面速度的刚体指垫驱动，硅胶摩擦系数 1.3。PE 膜与可压缩四面体裙料均可发生接触。模型无抓取等式、无抓取随动。左侧片头静置段使用求解后静止的初始包裹状态，进入接触前继续步进。右侧沿用相同的半透明外观、冻结代理和真实记录的指垫轨迹，显示凹坑为按指垫轮廓和代理压陷生成的几何。两个后端的腾空表现应分别从视频和输出读数解释。

[共享轨迹](../test/runs/2026-11-23/bag/real-grasp-skirt-c-r2/q_ref_top_center.npz) · [轨迹 SHA-256 与工位](../test/runs/2026-11-23/bag/real-grasp-skirt-c-r2/q_ref_top_center.json) · [左物理日志](../test/runs/2026-11-23/bag/real-grasp-skirt-c-r2/native_preview.log) · [右代理日志](../test/runs/2026-11-23/bag/real-grasp-skirt-c-r2/proxy_probe.log)。

## 复现

```sh
D=test/runs/2026-11-23/bag/real-grasp-skirt-c-r2
P=.venv-physical-parcel/bin/python
"$P" -B "$D/plan_top.py" "$D/q_ref_top_center.npz"
"$P" -B "$D/native_contact.py" "$D/q_ref_top_center.npz" "$D/native_preview.npz" --seconds 9.6 --young 9000 --pre-settled --mass .25
"$P" -B "$D/proxy_contact.py" "$D/q_ref_top_center.npz" "$D/proxy_probe.npz" --mass .25
LP_NUM_THREADS=7 "$P" -B "$D/render_attempt.py" left
LP_NUM_THREADS=7 "$P" -B "$D/render_attempt.py" right
"$P" -B "$D/render_attempt.py" compose
```

## 回放读数

250 g 原生模型在闭合时记录到双指接触，在提拉初段短暂带动裙包。4.40 s 双指法向力约 5.20 N 和 4.77 N，包底仍接近台面。4.60 s 两指接触读数均为零，裙包回到台面。冻结代理在 8.00 s 的块体净空约 260 mm。整段以真实接触尝试为口径，保留原生滑脱。左侧闭合期原生接触日志中存在超过 3 mm 的数值接触重叠。按要求跳过全片穿模审计，因此此片是调试回放。

[250 g 直出分屏](../test/runs/2026-11-23/bag/real-grasp-skirt-c-r2/compare_real_grasp_skirt_c.mp4)。

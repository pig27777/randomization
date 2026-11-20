# Stage 2 YAM 抓取检查

2026-11-20，北京时间，机器 `ubun`。验证 2 ms 下完整 YAM 关节运动、PE 包裹代理双向载荷、稳定悬停与松夹释放，以及贴近包裹表面的 MuJoCo 窗口回放。

```sh
out=test/runs/2026-11-20/bag/rss2027-stage2-yam-proxy-r2
flock /tmp/claude-gpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-physical-parcel/bin/python -B -m experience.yam_platform.angles \
  --backend src.low_order_parcel.yam_adapter --dt .002 --pinch-coeff 0 --lift-height .15 --output "$out"
MUJOCO_GL=glfw .venv-physical-parcel/bin/python -B test/low_order_parcel/view_yam_proxy.py "$out"
```

- 40 ms 入口检查通过。首次整回合闭合前因包裹偏移导致余料进给不足。降低入口反力后通过平台抓取判定，悬停低速窗仍偏短。对准挂点并采用 40 mm/s 起抬后，长折边版本通过全部动作检查。
- 对最终近表面版本核对 960 帧轨迹和 XML。六个 YAM 关节均运动，记录全部有限且与时间轴逐帧对齐，夹持、抬升、悬停和释放均完成，MuJoCo 零警告。整个悬停阶段双侧均捕获，台面支撑始终为 0 N。
- 用保存的关节、盒体位姿和指尖世界坐标，沿盒面法向复核悬停后半段距离，范围为 20.99–23.26 mm。用 `mj_geomDistance` 复核各阶段 YAM 碰撞外形与盒体的最近距离，闭合阶段最大重叠 0.14 mm。原生窗口显示完整机械臂，验证切换近景并截取 1706 × 960 图片。

[工况报告](../test/low_order_parcel/YAM_PROXY_REPORT.md) · [结果](../test/runs/2026-11-20/bag/rss2027-stage2-yam-proxy-r1/near-surface/result.json) · [全景截图](../test/runs/2026-11-20/bag/rss2027-stage2-yam-proxy-r1/near-surface/yam-proxy-ui.png) · [近景截图](../test/runs/2026-11-20/bag/rss2027-stage2-yam-proxy-r1/near-surface/yam-proxy-close.png)

## 侧壁与内容物随机化回放

2026-11-20 20:09，北京时间，`ubun`。试验持共享锁。15° 侧向初始位姿未达逆解姿态门槛。25° 有余料工况实现离台 53.67 mm，低余料对照留在桌面。独立复跑三组随机样本，抽样清单、抓取判定和代理关键数据逐项一致，全部六阶段均录制。60% 组补录释放后 35 帧，末帧净空 0 mm、支撑力 2.62 N。

```sh
out=test/runs/2026-11-20/bag/rss2027-stage2-yam-content-video-r2
flock /tmp/claude-gpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-physical-parcel/bin/python -B test/low_order_parcel/run_yam_content_randomization.py --output "$out"
env LP_NUM_THREADS=4 .venv-physical-parcel/bin/python -B \
  test/low_order_parcel/render_yam_content_comparison.py "$out" --video
ffprobe -v error -select_streams v:0 -show_streams "$out/yam-content-randomization.mp4"
ffmpeg -v error -xerror -i "$out/yam-content-randomization.mp4" -f null -
```

检查代理的余料守恒、几何捕获和滑移耗散。录像为 1648 × 552、20 fps、338 帧、16.90 s，H.264 全片解码无错误。抽看 13.15 s 画面与原生 MuJoCo 60% 工况。三次回放里有一次离台，两次未夹住。60% 工况悬停阶段的连续低速窗为 16 ms，低于 0.20 s。

MP4 保存在本地，复现命令见上。[工况数据](../test/low_order_parcel/YAM_PROXY_REPORT.md#内容物随机化结果)

"""Label saved failure playback and copy only new assets; no extra video QA."""
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
DEST = HERE.parents[1] / 'contents-model-videos-r1'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
meta = json.loads((HERE / 'replay-metadata.json').read_text())
result = json.loads((HERE / 'render-only/result.json').read_text())
params = result['dimensions']['parameters_X']
size = '×'.join(f'{v*1000:g}' for v in params['size'])
mass = result['dimensions']['mass_kg']


def caption(name, text, y, size=20, color='white'):
    path = HERE / f'{name}.txt'
    path.write_text(text)
    return (f"drawtext=fontfile='{FONT}':textfile='{path}':fontcolor={color}:"
            f'fontsize={size}:x=14:y={y}')


settings = f"{size} mm｜{mass:.3f} kg｜E={params['young']:g} Pa｜ν={params['poisson']:g}｜屈服应变={params['yield_strain']:g}｜yaw/tilt 0°/0°"
variants = [
    ('failure-raw.mp4', 'continuum-failed-narrow.mp4', [
        caption('failure-title', '连续体 StVK/J2｜原生接触 · 辅助捏取=0｜失败：四面体翻转', 8, 25),
        caption('failure-settings', settings, 44, 18),
        caption('failure-event', f"接近阶段第7331步停止｜最后安全时刻 {meta['last_safe_time_s']:.6f} s｜第一失败步 {meta['first_failed_gpu_time_s']:.6f} s", 426, 19),
        caption('failure-note', '包含实际GPU第一失败步几何；没有补造后续抓取', 453, 18),
    ]),
    ('inversion-raw.mp4', 'continuum-tet-inversion.mp4', [
        caption('inversion-title', '四面体翻转诊断｜红色表示负有符号体积单元｜10倍慢放', 8, 25),
        caption('inversion-meaning', 'J=V/V₀：J>0 为原朝向，J=0 压扁，J<0 表示局部朝向反转', 44, 20),
        caption('inversion-event', f"实际 {meta['first_failed_gpu_time_s']:.6f} s｜单元ID 43 的 J={meta['minimum_J']:.8f}，GPU判失效并停止", 426, 20, 'yellow'),
        caption('inversion-note', '末0.8秒按10倍慢放；末帧停留3秒｜仅回放保存几何，插值不是新物理步进', 453, 18),
    ]),
]
for raw, name, labels in variants:
    output = DEST / name
    filters = ['drawbox=x=0:y=0:w=iw:h=76:color=black:t=fill',
               'drawbox=x=0:y=420:w=iw:h=60:color=black@0.9:t=fill', *labels]
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(HERE/raw),
                    '-vf', ','.join(filters), '-an', '-c:v', 'libx264', '-preset', 'fast',
                    '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart',
                    str(output)], check=True)
    shutil.copyfile(output, ROOT / 'docs/assets' / name)
    print(output, flush=True)

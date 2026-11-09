"""Label experimental SNH playback and package the three pillow video assets."""
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
DEST = HERE.parents[1] / 'contents-model-videos-r1'
SOURCE = HERE.parent / 'soft-5k-snh'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
result = json.loads((SOURCE / 'result.json').read_text())


def caption(name, text, y, size=20, color='white'):
    path = HERE / f'{name}.txt'
    path.write_text(text)
    return (f"drawtext=fontfile='{FONT}':textfile='{path}':fontcolor={color}:"
            f'fontsize={size}:x=14:y={y}')


filters = [
    'drawbox=x=0:y=0:w=iw:h=76:color=black:t=fill',
    'drawbox=x=0:y=405:w=iw:h=75:color=black@0.9:t=fill',
    caption('title', '实验性SNH软枕头｜240×160×50 mm｜实际0.29 kg｜E=5 kPa', 8, 26),
    caption('conditions', '576四面体 · 189节点｜外表面隐藏 · 完整线框｜原生接触 · 辅助捏取=0｜原速', 44, 20),
    caption('outcome', '真实结果：7.938893 s提起失夹，双指接触丢失超过50 ms；未改标成功', 411, 22, 'yellow'),
    caption('material', 'ν=0.3 · 屈服应变=0.2｜Jmin=0.432761 · 无翻转｜实验性本构，未经材料标定', 448, 19),
]
output = DEST / 'continuum-pillow-snh-tetra.mp4'
subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(HERE/'tetra-raw.mp4'),
                '-vf', ','.join(filters), '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
                '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', str(output)], check=True)
metadata = dict(source=str(SOURCE), authoritative_model=str(SOURCE/'model.mjb'),
                video=output.name, result=result, exterior_faces_visible=False, detail_cutaway=False,
                fps=20, slow_factor=1, physics_recomputed=False,
                additional_video_QA='not run: user requested no checks')
(DEST / 'continuum-pillow-snh-tetra.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
for name in ('continuum-pillow-snh-tetra.mp4', 'continuum-pillow-soft5k-tetra.mp4',
             'continuum-pillow-soft5k-failed.mp4'):
    shutil.copyfile(DEST/name, ROOT/'docs/assets'/name)
print(output, flush=True)

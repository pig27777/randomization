"""Label the actual large soft pillow failure, without extra video QA."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
DEST = HERE.parents[1] / 'contents-model-videos-r1'
SOURCE = HERE.parent / 'soft-5k'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
result = json.loads((SOURCE / 'result.json').read_text())
failure = json.loads((SOURCE / 'failure-analysis.json').read_text())


def caption(name, text, y, size=20, color='white'):
    path = HERE / f'{name}.txt'
    path.write_text(text)
    return (f"drawtext=fontfile='{FONT}':textfile='{path}':fontcolor={color}:"
            f'fontsize={size}:x=14:y={y}')


filters = [
    'drawbox=x=0:y=0:w=iw:h=76:color=black:t=fill',
    'drawbox=x=0:y=405:w=iw:h=75:color=black@0.9:t=fill',
    caption('title', '连续体大软枕头｜240×160×50 mm｜实际0.29 kg｜E=5 kPa', 8, 26),
    caption('conditions', 'ν=0.3｜屈服应变=0.2｜189节点 · 576四面体｜原生接触 · 辅助捏取=0｜yaw/tilt 0°/0°', 44, 19),
    caption('failure', '失败：闭合阶段四面体翻转，单元ID 247；GPU于4.371451 s停止', 411, 22, 'yellow'),
    caption('safe', '仅播放至4.370951 s的最后安全几何｜原结果sim_fail，未补造后续抓取', 448, 19),
]
output = DEST / 'continuum-pillow-soft5k-failed.mp4'
subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(HERE/'safe-raw.mp4'),
                '-vf', ','.join(filters), '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
                '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', str(output)], check=True)
metadata = dict(source=str(SOURCE), video=output.name, safe_geometry_only=True,
                result=result, failure_analysis=failure, additional_video_QA='not run: user requested no checks')
(DEST / 'continuum-pillow-soft5k-failed.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
print(output, flush=True)

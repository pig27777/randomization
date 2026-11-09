"""Label the real extended observation; no additional video checks."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
DEST = HERE.parent / 'contents-model-videos-r1'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

observation = json.loads((HERE / 'soft/observation.json').read_text())
start, end = observation['start_s'], observation['last_safe_time_s']
raw = HERE / 'continuum-soft-10s-raw.mp4'
output = DEST / 'continuum-user-rounded-10s-observe.mp4'


def label(name, text, y, *, size=21, color='white', enable=None):
    path = HERE / f'{name}.txt'
    path.write_text(text)
    value = (f"drawtext=fontfile='{FONT}':textfile='{path}':fontcolor={color}:"
             f'fontsize={size}:x=14:y={y}')
    return value + (f":enable='{enable}'" if enable else '')


filters = [
    'drawbox=x=0:y=0:w=iw:h=76:color=black:t=fill',
    'drawbox=x=0:y=420:w=iw:h=60:color=black@0.85:t=fill',
    label('title', '连续体 StVK/J2｜真实 10 秒回放｜rounded 用户软参数（未标定）', 8, size=25),
    label('conditions', '240×160×50 mm｜0.190346 kg｜yaw/tilt 40°/25°｜原生接触｜GPU Newton', 44, size=19),
    label('task', '原抓取任务｜5.414 s 提起阶段失夹，原结果保留', 426,
          enable=f'lt(t,{start:.9f})'),
    label('observe', '失夹后保持姿态观察｜保持终止控制目标，未改判抓取成功', 426,
          color='yellow', enable=f'gte(t,{start:.9f})'),
    "drawtext=fontfile='" + FONT + "':text='真实仿真时刻 "
    f'%{{expr\\:min(t\\,{end:.9f})}} s' + "':fontcolor=white:fontsize=17:x=14:y=453",
]
subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(raw), '-vf', ','.join(filters),
                '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                '-threads', '2', '-movflags', '+faststart', str(output)], check=True)
metadata = dict(video=output.name, recording_source=str(HERE / 'soft'), **observation,
                fps=20, width=1280, height=480, extra_video_checks='not run, user requested no checks')
(DEST / 'continuum-user-rounded-10s-observe.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
print(output, flush=True)

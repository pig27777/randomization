"""Label this actual native-contact success and package new video assets."""
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
DEST = HERE.parent / 'contents-model-videos-r1'
SOURCE = HERE / 'narrow-100k'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
r = json.loads((SOURCE / 'result.json').read_text())
if r['outcome'] != 'success' or r['config']['pinch_coeff'] != 0:
    raise ValueError('only an actual native-contact success may receive this label')


def label(name, text, y, size=20, color='white'):
    path = HERE / f'success-{name}.txt'
    path.write_text(text)
    return (f"drawtext=fontfile='{FONT}':textfile='{path}':fontcolor={color}:"
            f'fontsize={size}:x=14:y={y}')


filters = [
    '[0:v]split=2[l][r]',
    '[l]crop=640:480:0:0[lv]',
    '[r]crop=440:330:740:75,scale=640:480:flags=lanczos[rv]',
    '[lv][rv]hstack=inputs=2,drawbox=x=0:y=0:w=iw:h=76:color=black:t=fill,'
    'drawbox=x=0:y=405:w=iw:h=75:color=black@0.9:t=fill,' + ','.join([
        label('title', '连续体 StVK/J2｜原生接触 · 辅助捏取=0｜完整抓取成功', 8, 25, 'yellow'),
        label('conditions', '120×45×50 mm｜40.5 g｜E=100 kPa｜ν=0.1｜屈服应变=0.2｜摩擦=0.8｜yaw/tilt 0°/0°', 44, 19),
        label('result', '15.026 s 完成提起、搬运、放下与松开｜弹塑性框架中的弹性响应：本次塑性范数=0', 411, 20),
        label('limit', '单个小尺寸配置，未作材料标定｜机械臂跟踪峰值40.60 mm｜右侧近景显示放大1.45倍', 448, 19),
    ]) + '[out]',
]
output = DEST / 'continuum-native-success.mp4'
subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(HERE/'continuum-native-success-raw.mp4'),
                '-filter_complex', ';'.join(filters), '-map', '[out]', '-an', '-c:v', 'libx264',
                '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2',
                '-movflags', '+faststart', str(output)], check=True)
shutil.copyfile(output, ROOT / 'docs/assets/continuum-native-success.mp4')
metadata = dict(source=str(SOURCE), video=output.name, outcome=r['outcome'],
                sim_time_s=r['sim_time_s'], actual_X=r['dimensions']['parameters_X'],
                mass_kg=r['dimensions']['mass_kg'], config=r['config'], phases=r['phases'],
                model_metrics=r['model_metrics'], peaks=r['peaks'], gpu_health=r['dimensions']['gpu_health'],
                source_sha256=r['source_sha256'], right_display_crop=[740,75,440,330],
                right_display_scale=[640,480], additional_video_QA='not run: user requested no checks')
(DEST / 'continuum-native-success.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
print(output, flush=True)

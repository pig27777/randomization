"""Label the whole-tetrahedron pillow playback; no surface skin or extra QA."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
DEST = HERE.parents[1] / 'contents-model-videos-r1'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def caption(name, text, y, size=20, color='white'):
    path = HERE / f'tetra-{name}.txt'
    path.write_text(text)
    return (f"drawtext=fontfile='{FONT}':textfile='{path}':fontcolor={color}:"
            f'fontsize={size}:x=14:y={y}')


filters = [
    'drawbox=x=0:y=0:w=iw:h=76:color=black:t=fill',
    'drawbox=x=0:y=405:w=iw:h=75:color=black@0.9:t=fill',
    caption('title', '大软枕头四面体版｜240×160×50 mm｜实际0.29 kg｜E=5 kPa', 8, 26),
    caption('conditions', '576四面体 · 189节点｜外表面隐藏 · 完整线框｜原生接触 · 辅助捏取=0｜原速', 44, 20),
    caption('failure', '失败：约4.37 s闭合阶段四面体翻转（单元ID 247），GPU停止', 411, 22, 'yellow'),
    caption('safe', '仅回放最后安全几何至4.370951 s｜未补造后续动作，未作慢放', 448, 19),
]
output = DEST / 'continuum-pillow-soft5k-tetra.mp4'
subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(HERE/'tetra-raw.mp4'),
                '-vf', ','.join(filters), '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
                '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', str(output)], check=True)
metadata = dict(source=str(HERE.parent/'soft-5k'), safe_replay=str(HERE/'safe-render-only'),
                video=output.name, exterior_faces_visible=False, detail_cutaway=False,
                tetrahedra=576, fps=20, slow_factor=1, physics_recomputed=False,
                original_outcome='sim_fail', original_failure='inverted tetrahedron at step 8743',
                failure_time_s=4.371451377868652, safe_end_s=4.370951175689697,
                additional_video_QA='not run: user requested no checks')
(DEST / 'continuum-pillow-soft5k-tetra.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
print(output, flush=True)

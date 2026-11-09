"""Reproduce this run's CPU playback, honest labels, and video checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RAW = HERE / 'raw'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FPS = 20
SOURCES = {
    'segmented-pinch': ('segmented-v13-yam-r1/matrix/trial_000', '.venv',
        '分段内容物 v1.3｜易弯锚点｜辅助捏取=1',
        '240×160×50 mm｜0.290 kg｜yaw/tilt 0°/0°｜源动力学 CPU · MuJoCo 3.17',
        '15.945 s 完整搬运并松开｜含冻结 v1.3 辅助捏取规则'),
    'segmented-native': ('segmented-v13-yam-r1/matrix/trial_012', '.venv',
        '分段内容物 v1.3｜易弯锚点｜原生接触 · 辅助捏取=0',
        '240×160×50 mm｜0.290 kg｜yaw/tilt 0°/0°｜源动力学 CPU · MuJoCo 3.17',
        '9.586 s 提起阶段失夹｜双指接触丢失超过 50 ms，原处停止'),
    'continuum-hard-native': ('continuum-gpu-r1/grasp-behavior-refined/trial_000', '.venv-continuum',
        '连续体 StVK/J2｜hard · box 参数组｜原生接触 · 辅助捏取=0',
        '240×160×50 mm｜0.290 kg｜yaw/tilt 0°/0°｜源动力学 GPU Newton · MuJoCo 4',
        '8.244 s 提起阶段失夹｜双指接触丢失超过 50 ms，原处停止'),
    'continuum-user-rounded': ('continuum-gpu-r1/ui/203244-ddf55a34/trial', '.venv-continuum',
        '连续体 StVK/J2｜rounded｜用户自调软参数，未经材料标定',
        '240×160×50 mm｜0.190346 kg｜yaw/tilt 40°/25°｜源动力学 GPU Newton · MuJoCo 4',
        '5.414 s 提起阶段失夹｜compressible 是参数标签｜原生接触 · 辅助捏取=0'),
}


def run(command, **kwargs):
    subprocess.run(command, check=True, cwd=ROOT, **kwargs)


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,pix_fmt,nb_read_frames,r_frame_rate:format=duration',
        '-of', 'json', str(path)]))


def text_filter(path, x, y, size=21, color='white', enable=None):
    value = (f"drawtext=fontfile='{FONT}':textfile='{path}':fontcolor={color}:"
             f'fontsize={size}:x={x}:y={y}')
    return value + (f":enable='{enable}'" if enable else '')


def caption(name, suffix, text):
    path = HERE / 'captions' / f'{name}-{suffix}.txt'
    path.write_text(text)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RAW.mkdir(exist_ok=True)
    (HERE / 'captions').mkdir(exist_ok=True)
    (HERE / 'keyframes').mkdir(exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', LP_NUM_THREADS='2')
    manifest = {'fps': FPS, 'physics_recomputed': False,
                'render_source_sha256': sha(ROOT / 'test/flex3d/render_video.py'),
                'assembly_source_sha256': sha(Path(__file__)), 'sources': {}}
    for name, (relative, venv, title, condition, outcome) in SOURCES.items():
        source = HERE.parent / relative
        result = json.loads((source / 'result.json').read_text())
        raw = RAW / f'{name}.mp4'
        if not raw.exists():
            command = [str(ROOT / venv / 'bin/python'), '-B', 'test/flex3d/render_video.py',
                       str(source.relative_to(ROOT)), str(raw.relative_to(ROOT)), '--fps', str(FPS)]
            with (RAW / f'{name}.log').open('w') as log:
                run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        output = HERE / f'{name}.mp4'
        filters = ['drawbox=x=0:y=0:w=iw:h=76:color=black:t=fill',
                   'drawbox=x=0:y=420:w=iw:h=60:color=black@0.85:t=fill',
                   text_filter(caption(name, 'title', title), 14, 8, 25,
                               'yellow' if name == 'segmented-pinch' else 'white'),
                   text_filter(caption(name, 'condition', condition), 14, 44, 19),
                   text_filter(caption(name, 'outcome', outcome), 14, 426, 20),
                   "drawtext=fontfile='" + FONT + "':text='保存状态回放 · 仿真时刻 "
                   f'%{{expr\\:min(t\\,{result["sim_time_s"]:.9f})}} s' + "':"
                   'fontcolor=white:fontsize=17:x=14:y=453']
        if not output.exists():
            run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(raw),
                 '-vf', ','.join(filters), '-an', '-c:v', 'libx264', '-preset', 'fast',
                 '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', str(output)])
        manifest['sources'][name] = {
            'source': str(source.relative_to(ROOT)), 'video': output.name,
            'sim_time_s': result['sim_time_s'], 'outcome': result['outcome'],
            'reason': result['reason'], 'config': result['config'],
            'dimensions': result['dimensions'], 'source_sha256': result['source_sha256'],
            'files_sha256': {p: sha(source / p) for p in ['config.json', 'result.json', 'model.xml', 'trajectory.npz']},
            'render': json.loads((RAW / f'{name}_render.json').read_text()),
        }

    # Equal envelope/mass/yaw/tilt and native contact; materials and source CPU/GPU differ.
    names = ['segmented-native', 'continuum-hard-native']
    durations = [float(probe(RAW / f'{name}.mp4')['format']['duration']) for name in names]
    longest = max(durations)
    filters = []
    for i, (name, duration) in enumerate(zip(names, durations)):
        title = ('分段 v1.3 易弯锚点｜CPU｜dt=0.4 ms' if i == 0 else
                 '连续体 hard/box｜GPU Newton｜dt=0.5 ms')
        outcome = ('9.586 s 提起阶段失夹' if i == 0 else '8.244 s 提起阶段失夹')
        labels = ['drawbox=x=0:y=0:w=iw:h=76:color=black:t=fill',
                  'drawbox=x=0:y=398:w=iw:h=82:color=black@0.9:t=fill',
                  text_filter(caption(name, 'comparison-title', title), 12, 8, 21),
                  text_filter(caption(name, 'comparison-conditions', '240×160×50 mm · 0.290 kg · 0°/0° · 辅助捏取=0'), 12, 41, 18),
                  text_filter(caption(name, 'comparison-note', '材料未匹配；仅动作回放，不作等材料或速度结论'), 12, 405, 19),
                  text_filter(caption(name, 'comparison-outcome', outcome), 12, 433, 18)]
        if longest > duration:
            labels += [text_filter(caption(name, 'stopped', '已停止：此后保持末个保存画面'), 12, 460, 17,
                                   'yellow', f'gte(t,{duration:.6f})')]
        filters.append(f'[{i}:v]crop=640:480:640:0,setpts=PTS-STARTPTS,'
                       f'tpad=stop_mode=clone:stop_duration={longest-duration:.6f},' + ','.join(labels) + f'[v{i}]')
    filters.append('[v0][v1]hstack=inputs=2[out]')
    comparison = HERE / 'native-contact-comparison.mp4'
    if not comparison.exists():
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
        for name in names:
            command += ['-i', str(RAW / f'{name}.mp4')]
        run(command + ['-filter_complex', ';'.join(filters), '-map', '[out]', '-an',
                       '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                       '-threads', '2', '-movflags', '+faststart', str(comparison)])
    manifest['comparison'] = {'video': comparison.name, 'sources': names,
                              'padded_tail_s': dict(zip(names, [longest-d for d in durations])),
                              'material_matched': False, 'speed_comparison': False}
    videos = [HERE / f'{name}.mp4' for name in SOURCES] + [comparison]
    checks = {'ffprobe': {}, 'full_decode': {}, 'keyframes': {}, 'files_sha256': {}}
    for path in videos:
        data = probe(path)
        stream = data['streams'][0]
        assert (stream['width'], stream['height'], stream['r_frame_rate']) == (1280, 480, '20/1'), data
        run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path), '-f', 'null', '-'])
        checks['ffprobe'][path.name] = data
        checks['full_decode'][path.name] = 'passed'
        checks['files_sha256'][path.name] = sha(path)
        duration = float(data['format']['duration'])
        timestamp = 12.8 if path.stem == 'segmented-pinch' else max(0, duration - .15)
        frame = HERE / 'keyframes' / f'{path.stem}.png'
        if not frame.exists():
            run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', str(timestamp),
                 '-i', str(path), '-frames:v', '1', str(frame)])
        checks['keyframes'][path.name] = {'time_s': timestamp, 'path': str(frame.relative_to(HERE))}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (HERE / 'video-checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print(HERE / 'video-checks.json', flush=True)


if __name__ == '__main__':
    main()

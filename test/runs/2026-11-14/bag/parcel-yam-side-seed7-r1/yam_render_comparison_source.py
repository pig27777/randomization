"""Render saved variants and assemble synchronized, fixed-camera comparisons.

The run manifest supplies variants (id/trial/label) and named comparisons (lists
of variant ids). This only replays trajectories; it never steps the simulator.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate:format=duration',
        '-of', 'json', str(path)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    output = args.run / 'videos'
    output.mkdir(exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def render(variant):
        name = variant['id']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name):
            raise ValueError('variant ids must be safe lowercase filenames')
        trial = args.run / 'matrix' / variant['trial']
        command = [sys.executable, '-B', str(Path(__file__).with_name('render_video.py')),
                   str(trial), str(output / f'{name}.mp4'), '--fps', str(args.fps), '--view', 'detail']
        with (output / f'{name}.log').open('x') as log:
            subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        return name, probe(output / f'{name}.mp4')

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        metadata = dict(pool.map(render, manifest['variants']))
    variants = {v['id']: v for v in manifest['variants']}
    comparisons = {}
    for group, names in manifest['comparisons'].items():
        paths = [output / f'{name}.mp4' for name in names]
        durations = [float(metadata[name]['format']['duration']) for name in names]
        longest = max(durations)
        filters = []
        for i, (name, duration) in enumerate(zip(names, durations)):
            caption = output / f'{name}-caption.txt'
            caption.write_text(variants[name].get('label', name))
            font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
            font_arg = f"fontfile='{font}':" if font.exists() else ''
            filters.append(
                f'[{i}:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={longest-duration:.6f},'
                f"drawtext={font_arg}textfile='{caption.resolve()}':fontcolor=white:fontsize=18:"
                'x=14:y=h-54:box=1:boxcolor=black@0.65,'
                "drawtext=text='STOPPED - final frame held':fontcolor=white:fontsize=14:"
                f"x=14:y=h-26:box=1:boxcolor=black@0.65:enable='gte(t,{duration:.6f})'[v{i}]")
        filters.append(''.join(f'[v{i}]' for i in range(len(names))) + f'hstack=inputs={len(names)}[out]')
        path = output / f'{group}-comparison.mp4'
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
        for item in paths:
            command += ['-i', str(item)]
        command += ['-filter_complex', ';'.join(filters), '-map', '[out]', '-an', '-r', str(args.fps),
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                    '-threads', '2', '-movflags', '+faststart', str(path)]
        subprocess.run(command, check=True)
        comparisons[group] = dict(variants=names, duration_s=longest, metadata=probe(path),
                                  padded_tail_s=dict(zip(names, [longest-d for d in durations])))
    for path in output.glob('*.mp4'):
        subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path), '-f', 'null', '-'], check=True)
    report = dict(variants=metadata, comparisons=comparisons, full_decode=True,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  fixed_camera_distance=.67, fixed_camera_azimuth=-45,
                  fixed_camera_elevation=-30, fps=args.fps, physics_recomputed=False,
                  interpolated_saved_states=True)
    (output / 'video-checks.json').write_text(json.dumps(report, indent=2))
    print(output / 'video-checks.json', flush=True)


if __name__ == '__main__':
    main()

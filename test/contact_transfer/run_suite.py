"""Replay saved benchmark configurations without overwriting the originals."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
QUICK = ['guided_base_plate', 'guided_base_finger_off', 'wave_soft30_membrane',
         'wave_halfstep_membrane', 'mj0p1d20_plate', 'mj0p1d20_finger']


def command(result, prefix='replay'):
    c, e = result['config'], result['environment']
    tag = prefix + '_' + c['tag']
    if 'newton' in e:
        cmd = [e['executable'], '-B', str(HERE / 'benchmark.py'), '--cases', c['case'], '--tag', tag]
        for key, flag in [('dt', '--dt'), ('iterations', '--iterations'), ('ke', '--ke'), ('kd', '--kd'),
                          ('friction', '--friction'), ('seconds', '--seconds'), ('material_scale', '--material-scale'),
                          ('k_start', '--k-start'), ('drag', '--drag'), ('sdf_voxel', '--sdf-voxel'),
                          ('friction_epsilon', '--friction-epsilon')]:
            if key in c: cmd += [flag, str(c[key])]
        cmd += ['--cells', *map(str, c['cells'])]
        if not c.get('full_surface', True): cmd += ['--vertex-only']
        if not c.get('tile', True): cmd += ['--no-tile']
        if not c.get('guided', False): cmd += ['--free-tool']
        if c.get('drive', 'force') == 'position': cmd += ['--drive', 'position', '--tool-z-mm', str(c['tool_z_mm'])]
        if c.get('dense_s', 0): cmd += ['--dense-s', str(c['dense_s'])]
        if c.get('log_barrier', False): cmd += ['--log-barrier']
        if c.get('friction_zero_fix', False): cmd += ['--friction-zero-fix']
        if c.get('collision_iterations', False): cmd += ['--collision-iterations']
        if c.get('planar_quadrature', False): cmd += ['--planar-quadrature']
        if c['case'].startswith('membrane_'):
            cmd += ['--membrane', c['membrane'], '--membrane-ke', str(c['membrane_ke']), '--membrane-kd', str(c['membrane_kd'])]
        if c.get('checkpoints'): cmd += ['--checkpoints', *map(str, c['checkpoints'])]
        return cmd
    return [e['executable'], '-B', str(HERE / 'mujoco_benchmark.py'), '--cases', c['case'], '--tag', tag,
            '--dt', str(c['dt']), '--seconds', str(c['seconds']), '--drag', str(c['particle_drag_per_s'])]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--group', choices=['quick', 'all'], default='quick')
    ap.add_argument('--prefix', default='replay')
    ap.add_argument('--dry-run', action='store_true', help='print commands without running physics')
    args = ap.parse_args()
    paths = sorted((HERE / 'results').glob('*.json'))
    if args.group == 'quick': paths = [p for p in paths if p.stem in QUICK]
    else: paths = [p for p in paths if not p.stem.startswith(('replay_', 'baseline_', 'iter40_'))]
    if not paths: raise SystemExit('No saved configurations found')
    failed = []
    for path in paths:
        print('Replaying', path.stem, flush=True)
        cmd = command(json.loads(path.read_text()), args.prefix)
        if args.dry_run:
            import shlex
            print(shlex.join(cmd))
            continue
        status = subprocess.run(cmd, cwd=HERE).returncode
        replay = HERE / 'results' / (args.prefix + '_' + path.name)
        if status or not replay.is_file() or not json.loads(replay.read_text())['passed']:
            failed.append(path.stem)
    print(json.dumps(dict(runs=len(paths), failed=failed), indent=2))
    sys.exit(bool(failed))

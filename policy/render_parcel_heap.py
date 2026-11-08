"""Render initial matched exteriors and recorded heap states, without resimulation."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np

import parcel_heap as heap
from probe_feedback import save
from render_probe import write_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=heap.RUN/'contact_fix')
    parser.add_argument('--rows',type=Path)
    args = parser.parse_args()
    heap.install_backend(args.run)
    out = args.run/'frames'
    out.mkdir(exist_ok=True)
    references, comparisons, manifest = {}, [], []
    rows = [dict(case=c,layout='top') for c in heap.CASES]
    if args.rows:
        rows = [json.loads(line) for line in args.rows.read_text().splitlines()]
    for row in rows:
        if args.rows and not row.get('keyframes'):
            continue
        robot = heap.scene(row['case'],row['layout'],row.get('action','center'),row.get('jitter',0.))
        m,d = robot.sim.m,robot.sim.d
        frames = row.get('keyframes') or [dict(phase='initial',qpos=d.qpos.copy())]
        with mujoco.Renderer(m,600,800,max_geom=20000) as renderer:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [-.20,.05,.105]
            camera.distance=.65; camera.azimuth=55; camera.elevation=-55
            options = mujoco.MjvOption()
            options.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
            options.sitegroup[:] = 0
            options.geomgroup[3] = 0
            for frame in (frames[0],frames[-1]) if len(frames)>1 else frames:
                d.qpos[:] = frame['qpos']
                mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera=camera,scene_option=options)
                pixels = renderer.render()
                filename = '_'.join(str(x) for x in (row['case'],row['layout'],row.get('action'),
                                                    row.get('jitter'),frame['phase']) if x is not None)+'.png'
                write_png(out/filename,pixels)
                manifest.append(dict(file=filename,case=row['case'],layout=row['layout'],
                                     phase=frame['phase'],outcome=row.get('outcome','initial_only')))
                if frame['phase']=='initial':
                    family = 'envelope' if 'envelope' in row['case'] else 'bag'
                    if family in references:
                        diff = np.abs(pixels.astype(int)-references[family].astype(int))
                        comparisons.append(dict(case=row['case'],family=family,
                                                changed_pixels=int(np.any(diff,axis=2).sum()),
                                                max_channel_difference=int(diff.max())))
                    else:
                        references[family] = pixels.copy()
    save(out/(f'{args.rows.stem}_manifest.json' if args.rows else 'initial_manifest.json'),dict(frames=manifest,
                                                                                 comparisons=comparisons))
    print(json.dumps(comparisons))


if __name__=='__main__':
    main()

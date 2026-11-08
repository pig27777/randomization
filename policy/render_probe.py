"""Render recorded YAM test keyframes, without resimulating or editing images."""
import argparse
import json
import os
from pathlib import Path
import struct
import zlib

os.environ.setdefault('MUJOCO_GL','egl')
from probe_feedback import DEFAULT_RUN,load_backend,read_rows,save


def write_png(path,frame):
    def chunk(kind,data):
        return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
    h,w,_=frame.shape
    raw=b''.join(b'\x00'+row.tobytes() for row in frame)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!IIBBBBB',w,h,8,2,0,0,0))+
                     chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,default=DEFAULT_RUN)
    args=ap.parse_args()
    model=load_backend(args.run)
    from yam_probe_robot import Yam
    import mujoco
    row=next(r for r in read_rows(args.run/'test.jsonl') if r.get('keyframes'))
    spec=next(o for o in json.loads((args.run/'test_objects.json').read_text()) if o['id']==row['object_id'])
    robot=Yam(model.Kappa.from_dict(spec['kappa']),row['seed'])
    m,d=robot.sim.m,robot.sim.d
    # Color only: geometry and physical state come directly from the recorded trial.
    for g in robot.sim.obj_geoms:
        m.geom_rgba[g]=[.88,.48,.12,1.]
    destination=args.run/'frames'
    destination.mkdir(exist_ok=True)
    records=[]
    with mujoco.Renderer(m,height=600,width=800,max_geom=20000) as renderer:
        camera=mujoco.MjvCamera()
        options=mujoco.MjvOption()
        options.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=False
        options.geomgroup[3]=0
        camera.lookat[:]=[0.,-.08,.16]
        camera.distance=1.25
        camera.azimuth=45
        camera.elevation=-25
        for frame in row['keyframes']:
            if frame['phase'] not in ('approach','squeeze','carry','lower','open'):
                continue
            d.qpos[:]=frame['qpos']
            mujoco.mj_forward(m,d)
            renderer.update_scene(d,camera=camera,scene_option=options)
            path=destination/f"{frame['phase']}.png"
            write_png(path,renderer.render())
            records.append(dict(phase=frame['phase'],t=frame['t'],path=str(path.relative_to(args.run))))
    save(destination/'manifest.json',dict(object_id=row['object_id'],seed=row['seed'],arm=row['arm'],
                                        outcome=row['outcome'],frames=records))
    print(json.dumps(records))


if __name__=='__main__':
    main()

"""Cantilever and extreme-parameter checks, independent of robot actions."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import mujoco_warp as mjw
import numpy as np
import warp as wp
from contents.parcel import Batch
from contents.parcel import kernels as K


@wp.kernel
def cantilever(q: wp.array2d(dtype=float), v: wp.array2d(dtype=float),
               rest: wp.array2d(dtype=float), p: wp.array(dtype=K.Parameters),
               force: wp.array2d(dtype=float), body_force: wp.array2d(dtype=wp.spatial_vector), end: int):
    e=wp.tid()
    for i in range(3):
        target=rest[e,i]
        if i==2:target+=.5
        force[e,i]=30000.0*(target-q[e,i])-100.0*v[e,i]
        force[e,3+i]=-200.0*q[e,4+i]-.05*v[e,3+i]
    body_force[e,end]=wp.spatial_vector(0.0,0.0,-p[e].mass*9.81,0.0,0.0,0.0)


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    result={'sag':[],'extremes':[]}
    for factor in (.5,1.,2.):
        b=Batch(8,prototype=3,level=0)
        q=b.data.qpos.numpy();q[:,2]+=.5;b.data.qpos.assign(q)
        b.model.jnt_stiffness.assign(b.model.jnt_stiffness.numpy()*factor)
        mjw.forward(b.model,b.data)
        end=int(b.ids.numpy()[2,0])
        def step():
            b.data.xfrc_applied.zero_();b.data.qfrc_applied.zero_()
            wp.launch(cantilever,8,[b.data.qpos,b.data.qvel,b.model.qpos0,b.params,b.data.qfrc_applied,b.data.xfrc_applied,end])
            mjw.step(b.model,b.data)
        step();wp.synchronize()
        with wp.ScopedCapture() as capture:step()
        series=[]
        for i in range(4000):
            wp.capture_launch(capture.graph)
            if i%40==0:
                x=b.data.xpos.numpy()
                series.append(x[:,int(b.ids.numpy()[0,0]),2]-x[:,end,2])
        sag=float(np.mean(series[-1]))
        result['sag'].append(dict(stiffness_factor=factor,sag_m=sag,finite=bool(np.isfinite(series).all())))
        np.savez_compressed(out/f'sag-{factor}.npz',time=np.arange(len(series))*.02,sag=series)
    cfg=json.loads(Path('contents/parcel/config.json').read_text())
    for bound in (0,1):
        fixed={k:[v[bound],v[bound]] for k,v in cfg.items()}
        path=out/f'extreme-{bound}.json';path.write_text(json.dumps(fixed,indent=2))
        for rigid in (True,False):
            b=Batch(8,rigid=rigid,config=path,action=1)
            graph=b.graph()
            maximum=0.
            for i in range(4000):
                wp.capture_launch(graph)
                if i%40==0:
                    maximum=max(maximum,float(np.abs(b.data.qvel.numpy()).max()))
            q=b.data.qpos.numpy()
            result['extremes'].append(dict(bound=bound,rigid=rigid,finite=bool(np.isfinite(q).all()),peak_velocity=maximum))
    result['sag_monotone']=all(result['sag'][i]['sag_m']>result['sag'][i+1]['sag_m'] for i in range(2))
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    main(parser.parse_args().output)

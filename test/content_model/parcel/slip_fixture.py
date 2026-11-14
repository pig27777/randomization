"""Module B isolation: initialize an engaged fold and vary friction only.

The physical fingers supply normal load. Capture is explicitly initialized to
isolate load transfer, as allowed by the bag/Module-B material test milestone.
Robot grasp behavior is validated separately through the shared YAM protocol.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import warp as wp
from contents.parcel import Batch


def main(output,dt=.0005):
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for friction in (.05,.8):
        b=Batch(8,rigid=True,prototype=1,action=2,level=0,dt=dt)
        graph=b.graph()
        b.reset()
        for _ in range(round(.8/b.dt)):wp.capture_launch(graph)
        # An explicit material-fixture engagement at the current anchors.
        x=b.data.xpos.numpy();ids=b.ids.numpy();p=b.params.numpy();s=b.states.numpy()
        cart=x[:,b.cart]
        centers=x[:,ids[0,0]]
        for i in range(3):
            anchor=centers.copy();anchor[:,0]+=(i-1)*p['size'][:,0]/3
            s['rest'][:,i]=np.linalg.norm(anchor-cart,axis=1)+p['lift_slack']
        s['mode']=1;s['captured']=.015;s['gathered']=.015
        b.states.assign(s)
        p['mu_s']=friction;p['mu_d']=.8*friction;b.params.assign(p)
        b.level=2
        graph=b.graph()
        time=[];states=[];normal=[]
        for i in range(round(1.5/b.dt)):
            wp.capture_launch(graph)
            if i%round(.02/b.dt)==0:
                time.append(i*b.dt);states.append(b.states.numpy());normal.append(b.normal.numpy())
        states=np.array(states)
        row=dict(friction=friction,detached=int((states[-1]['mode']==3).sum()),
                 peak_slip=float(states['slip'].max()),max_tension=float(states['tension'].max()),finite=bool(np.isfinite(b.data.qpos.numpy()).all()))
        rows.append(row)
        np.savez_compressed(output/f'slip-{friction}.npz',time=time,state=states,normal=normal)
        print(json.dumps(row),flush=True)
    (output/'summary.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dt',type=float,default=.0005)
    args=p.parse_args();main(args.output,args.dt)

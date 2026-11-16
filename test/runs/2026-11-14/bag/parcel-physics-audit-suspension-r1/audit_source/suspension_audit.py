"""Elastic-path material fixture; no robot grasp or capture success criterion."""
import argparse
from common import ROOT,setup,write
import numpy as np
import warp as wp
import mujoco_warp as mjw
from contents.parcel import Batch
from payload_audit import set_params


def main(output):
    out=setup(output,'elastic suspension and path-count isolation')
    rows=[]
    for paths in ([1],[0,2],[0,1,2],[2]):
        b=Batch(8,rigid=True,prototype=1,level=1)
        p=b.params.numpy();p['size']=[.24,.16,.06];p['mass']=.4;p['com']=0;p['rope_k']=400
        set_params(b,p)
        q=b.data.qpos.numpy();q[:,2]=.06;q[:,b.addresses.numpy()[2,0]]=.12
        b.data.qpos.assign(q);mjw.forward(b.model,b.data)
        s=b.states.numpy();s['mode']=1;s['captured']=.015;s['gathered']=.015
        s['rest']=10.
        for i in paths:s['rest'][:,i]=np.sqrt(((i-1)*.08)**2+.06**2)
        b.states.assign(s)
        graph=b.graph()
        times=[];position=[];rotation=[];forces=[];tension=[];velocity=[]
        body=b.ids.numpy()[0,0]
        for i in range(16000):
            wp.capture_launch(graph)
            if i%40==0:
                times.append(i*b.dt);position.append(b.data.xpos.numpy());rotation.append(b.data.xmat.numpy())
                forces.append(b.data.xfrc_applied.numpy());tension.append(b.states.numpy()['tension'])
                velocity.append(b.data.qvel.numpy())
        f=np.array(forces);r=np.array(rotation);v=np.array(velocity)
        vertical=f[-50:,:,body,2].mean(0)
        tilt=np.arccos(np.clip(r[-1,:,body,2,2],-1,1))*180/np.pi
        rows.append(dict(paths=paths,weight=.4*9.81,mean_final_support=float(vertical.mean()),
            max_relative_force_imbalance=float(abs(vertical-.4*9.81).max()/(.4*9.81)),
            first_payload_torque=f[0,0,body,3:],final_tilt_deg=tilt,
            final_speed=float(abs(v[-1,:,:6]).max()),
            note='equilibrium claim requires small terminal speed; otherwise recorded transient only'))
        np.savez_compressed(out/f'paths-{"-".join(map(str,paths))}.npz',time=times,xpos=position,xmat=rotation,
            force=forces,tension=tension,qvel=velocity,params=p,state=b.states.numpy())
    write(out/'summary.json',rows)
    print(out/'summary.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(parser.parse_args().output)

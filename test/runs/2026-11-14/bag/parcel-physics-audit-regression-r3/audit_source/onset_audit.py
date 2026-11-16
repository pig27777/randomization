"""Prescribed-displacement constitutive tests for slack and A timestep response.

Payload and grip poses are controlled inputs, not integrated robot trajectories.
Native response tests and YAM evidence are reported separately.
"""
import argparse
from common import ROOT,setup,write
import numpy as np
import warp as wp
from contents.parcel import Batch
from contents.parcel import kernels as K
from kernel_audit import law


def main(output):
    out=setup(output,'slack onset and engagement timestep probes')
    rows=[];engagement=[];probabilities=[]
    for dt in (.00025,.0005,.001):
        for slack in (0.,.005,.02):
            b=Batch(1,rigid=True,prototype=1,dt=dt)
            p=b.params.numpy()[0];s=b.states.numpy();x0=b.data.xpos.numpy();rot=b.data.xmat.numpy()
            body=b.ids.numpy()[0,0]
            for i in range(3):
                point=x0[0,body]+rot[0,body]@np.array([(i-1)*p['size'][0]/3,0,0])
                s['rest'][0,i]=np.linalg.norm(point-x0[0,b.cart])+slack
            s['mode']=1;s['captured']=.02;b.states.assign(s)
            trace=[];onset=None
            for i in range(round(.3/dt)):
                time=(i+1)*dt;x=x0.copy();x[0,b.cart,2]+=.1*time;b.data.xpos.assign(x)
                b.data.xfrc_applied.zero_();b.data.qfrc_applied.zero_()
                wp.launch(K.dynamics,1,[dt,1,1,b.cart,0,0,0,0,0.,b.params,b.states,b.ids,b.normal,b.gap,
                    b.data.xpos,b.data.xmat,b.data.xipos,b.data.qpos,b.data.qfrc_applied,b.data.xfrc_applied])
                tension=b.states.numpy()['tension'][0].sum()
                if tension>1e-5 and onset is None:onset=time
                trace.append([time,.1*time,float(tension)])
            rows.append(dict(dt=dt,slack=slack,expected_onset=slack/.1,measured_onset=onset))
            np.savez_compressed(out/f'onset-dt{dt}-slack{slack}.npz',trace=trace,params=b.params.numpy())
        b=Batch(512,rigid=True,prototype=1,dt=dt)
        b.normal.fill_(10.);addr=b.addresses.numpy();q0=b.data.qpos.numpy();v=b.data.qvel.numpy()
        v[:,addr[3,1]]=v[:,addr[4,1]]=-.02;b.data.qvel.assign(v)
        n=round(.2/dt)
        for i in range(n):
            gap=.012-.04*(i+1)*dt;q=q0.copy();q[:,addr[3,0]]=q[:,addr[4,0]]=gap/2
            b.data.qpos.assign(q);b.gap.fill_(gap)
            law(b,close=int(i==n-1),speed=.04,region=0)
        state=b.states.numpy();probabilities.append(state['probability'].copy())
        engagement.append(dict(dt=dt,captured=int((state['mode']==1).sum()),
            max_gathered=float(state['gathered'].max()),max_probability=float(state['probability'].max())))
        np.savez_compressed(out/f'engagement-dt{dt}.npz',params=b.params.numpy(),state=state,
            imposed_gap_start=.012,imposed_gap_end=.004,imposed_closing_speed=.04,normal=b.normal.numpy())
    write(out/'summary.json',dict(slack=rows,engagement=engagement,
        max_probability_difference_to_smallest_dt=[float(abs(p-probabilities[0]).max()) for p in probabilities]))
    print(out/'summary.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)

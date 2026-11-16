"""Additional energy, backend-contract and event counterexamples."""
import argparse
from common import ROOT,setup,write
import numpy as np
import warp as wp
from contents.parcel import Batch
from contents.parcel.model import film_proxy
from kernel_audit import law,state_for_load


def main(output):
    out=setup(output,'energy gradient, contact contract and event attacks')
    result={}
    b=Batch(32,rigid=True,prototype=1)
    p=b.params.numpy();s=b.states.numpy();s['gathered']=np.minimum(.003,p['budget'][:,0]*.5)
    b.states.assign(s);b.normal.fill_(10.)
    addr=b.addresses.numpy();q=b.data.qpos.numpy();q[:,addr[3,0]]=q[:,addr[4,0]]=.015
    b.data.qpos.assign(q)
    native=b.normal.numpy().copy()
    wp.launch(film_proxy,32,[b.states,b.params,b.data.qpos,b.data.qvel,b.addresses,b.normal,b.data.qfrc_applied])
    gpu=b.normal.numpy().copy()
    # Exact source contract in YAM pre-engagement: loads += scalar proxy.
    width=np.minimum(.004+2*s['gathered'],.024);proxy=5000*np.maximum(width-.03,0)
    yam=native+proxy[:,None]
    result['pre_engagement_normals']=dict(gpu=gpu[0],yam_contract=yam[0],native=native[0],gap=.03,
        note='same input normal, gap, material and zero velocity; constitutive API comparison, not full YAM replay')
    b.normal.assign(gpu);b.gap.fill_(.03);law(b,speed=.04)
    egpu=b.states.numpy()['gathered'].copy()
    b.states.assign(s);b.normal.assign(yam);law(b,speed=.04)
    eyam=b.states.numpy()['gathered'].copy()
    result['pre_engagement_normals']['gathered_delta_difference']=float(np.max(eyam-egpu))
    # Consistent closing velocity: all pad motion and material state are accounted.
    b.reset();p=b.params.numpy();s=b.states.numpy();s['gathered']=np.minimum(.003,p['budget'][:,0]*.5)
    q=b.data.qpos.numpy();q[:,addr[3,0]]=q[:,addr[4,0]]=.0035
    v=b.data.qvel.numpy();v[:,addr[3,1]]=v[:,addr[4,1]]=-.01
    b.data.qpos.assign(q);b.data.qvel.assign(v);b.states.assign(s);b.normal.fill_(10.);b.gap.fill_(.007)
    e0=s['gathered'].copy();law(b,speed=.02);e1=b.states.numpy()['gathered']
    def U(e,half_gap):return 10000.*np.maximum(np.minimum(.002+e,.012)-half_gap,0)**2
    old=U(e0,.0035);closed=U(e0,.0035-.01*b.dt);new=U(e1,.0035-.01*b.dt)
    result['consistent_closing_energy']=dict(max_unaccounted_storage=float((new-closed).max()),
        e0=e0[0],e1=e1[0],old_U=old[0],new_U=new[0],
        fixed_material_closing_work=closed[0]-old[0],normal_dissipation=2*2*.01**2*b.dt,
        power_formula='dU/de * de/dt; for two active springs dU/de=2*k*penetration',
        note='no native tangential contact in this isolated proxy law; captured mode still inactive')
    np.savez_compressed(out/'closing-energy.npz',params=p,e0=e0,e1=e1,U0=old,U_closed=closed,U1=new,qpos=q,qvel=v)
    # Single closure repeated versus event-flag jitter without actual finger motion.
    b.reset();p=b.params.numpy();s=b.states.numpy();s['gathered']=np.minimum(.003,p['budget'][:,0]*.8)
    b.states.assign(s);b.gap.fill_(.001)
    attempts=[]
    for attempt in range(20):
        b.normal.fill_(10.);law(b,close=0)
        b.normal.fill_(10.);law(b,close=1)
        attempts.append([attempt+1,int((b.states.numpy()['mode']==1).sum())])
    result['flag_jitter']=dict(attempts=attempts,max_qvel=float(abs(b.data.qvel.numpy()).max()))
    # Loss of all contact retains the material state indefinitely with speed zero.
    b.reset();s=b.states.numpy();s['gathered']=.002;b.states.assign(s);b.normal.zero_()
    for _ in range(1000):law(b,speed=0)
    result['contact_loss']=dict(elapsed=.5,gathered_initial=.002,gathered_final=b.states.numpy()['gathered'][0])
    # Zero-normal transient and steady slip resistance.
    b.reset();state_for_load(b,extension=.01,captured=.015);b.normal.zero_()
    trace=[]
    for i in range(60):
        law(b)
        s=b.states.numpy()[0]
        trace.append([i*b.dt,float(s['slip']),float(s['tension'].sum()),int(s['mode'])])
    np.savez_compressed(out/'zero-normal-transient.npz',trace=trace,params=b.params.numpy())
    result['zero_normal_transient']=dict(first_total_tension=trace[0][2],final_total_tension=trace[-1][2],final_mode=trace[-1][3])
    # Independent spatial virtual-work identity, nonzero gripper anchor moment.
    b=Batch(1,rigid=True,prototype=1)
    state_for_load(b,extension=.02)
    s=b.states.numpy();s['anchor'][0]=[.012,.006,.01];b.states.assign(s);b.normal.fill_(10000.)
    law(b)
    force=b.data.xfrc_applied.numpy()[0];base_x=b.data.xpos.numpy();base_R=b.data.xmat.numpy();p=b.params.numpy()[0];s=b.states.numpy()[0]
    body=int(b.ids.numpy()[0,0]);cart=b.cart
    def potential(x,R):
        grip=x[0,cart]+R[0,cart]@s['anchor']
        points=[x[0,body]+R[0,body]@np.array([(i-1)*p['size'][0]/3,0,0]) for i in range(3)]
        dl=np.array([np.linalg.norm(point-grip) for point in points])-s['rest']-s['slip']
        return .5*p['rope_k']*np.maximum(dl,0).dot(np.maximum(dl,0))
    gradients=[];eps=.0001
    for target in (body,cart):
        gradient=[]
        for axis in range(3):
            plus=base_x.astype(float).copy();minus=plus.copy()
            plus[0,target,axis]+=eps;minus[0,target,axis]-=eps
            gradient.append(-(potential(plus,base_R)-potential(minus,base_R))/(2*eps))
        gradients.append(dict(body=target,measured_force=force[target,:3],minus_gradient=gradient,
                              max_error=float(abs(np.array(gradient)-force[target,:3]).max())))
    result['potential_gradient']=gradients
    result['nonzero_cart_moment']=force[cart,3:]
    write(out/'summary.json',result)
    print(out/'summary.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(parser.parse_args().output)

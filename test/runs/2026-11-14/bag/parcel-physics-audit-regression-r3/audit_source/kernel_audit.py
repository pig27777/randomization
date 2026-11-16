"""Adversarial constitutive probes of the unchanged GPU kernels.

Input-controlled states isolate causal laws. These probes do not claim to be
complete robot trajectories. Model state and every diagnostic override are saved.
"""
import argparse
from common import ROOT,setup,write
import numpy as np
import warp as wp
from contents.parcel import Batch
from contents.parcel import kernels as K
from contents.parcel.model import film_proxy
from reference import return_roots


def law(b,dt=None,close=0,speed=0.,region=0,release=0):
    d=b.data
    d.xfrc_applied.zero_();d.qfrc_applied.zero_()
    wp.launch(K.dynamics,b.nworld,[dt or b.dt,int(b.rigid),2,b.cart,region,close,release,0,speed,
        b.params,b.states,b.ids,b.normal,b.gap,d.xpos,d.xmat,d.xipos,d.qpos,d.qfrc_applied,d.xfrc_applied])


def state_for_load(b,extension=.01,captured=.015,k=400.):
    p=b.params.numpy();s=b.states.numpy()
    x=b.data.xpos.numpy();mat=b.data.xmat.numpy();ids=b.ids.numpy()
    for e in range(b.nworld):
        for i in range(3):
            point=x[e,ids[0,0]]+mat[e,ids[0,0]]@np.array([(i-1)*p[e]['size'][0]/3,0,0])
            s['rest'][e,i]=np.linalg.norm(point-x[e,b.cart])-extension
    s['mode']=1;s['captured']=captured;s['gathered']=captured
    p['rope_k']=k
    b.params.assign(p);b.states.assign(s)
    return p,s


def main(output):
    out=setup(output,'constitutive GPU audit')
    b=Batch(32,rigid=True,prototype=1)
    write(out/'original_kappa.json',b.params.numpy())
    results={}
    # Independent force and global moment balance for three tensions and offsets.
    state_for_load(b)
    b.normal.fill_(10000.)
    law(b)
    force=b.data.xfrc_applied.numpy();com=b.data.xipos.numpy()
    torque=force[:,:,3:]+np.cross(com,force[:,:,:3])
    results['reciprocity']=dict(max_net_force=float(abs(force[:,:,:3].sum(1)).max()),
        max_global_moment=float(abs(torque.sum(1)).max()),nonzero_tension=bool((b.states.numpy()['tension']>0).all()))
    # Slack start and eccentric moment, same measured geometry, no slip allowed.
    slack=[]
    for extension in (-.01,0.,.005,.01):
        p,s=state_for_load(b,extension=extension)
        b.normal.fill_(10000.);law(b)
        s=b.states.numpy();f=b.data.xfrc_applied.numpy()
        slack.append(dict(extension=extension,tension=s['tension'][0],payload_moment=f[0,b.ids.numpy()[0,0],3:]))
    results['slack']=slack
    # A simple fixed-point pull without normal contact retains a viscous load.
    b.reset();state_for_load(b,extension=.01)
    b.normal.zero_();law(b)
    results['zero_normal']=dict(state=b.states.numpy()[0],forces=b.data.xfrc_applied.numpy()[0],
        note='one constitutive update, all normal loads zero; slip viscosity remains')
    np.savez_compressed(out/'zero-normal.npz',params=b.params.numpy(),state=b.states.numpy(),forces=b.data.xfrc_applied.numpy())
    # Multiple roots inside a physically admissible state of the constitutive law.
    multi=[]
    for dt in (.00025,.0005,.001):
        b.reset();p,s=state_for_load(b,extension=38./1200,captured=.008,k=400.)
        p['mu_s']=.8;p['mu_d']=.64;p['efficiency']=.9;p['slip_damping']=2.
        b.params.assign(p)
        b.normal.fill_(float(30./(.64*.9)/2))
        before=b.states.numpy().copy();law(b,dt=dt)
        roots,fn=return_roots(np.repeat(38./1200,3),400.,.008,0.,30.,2.,dt)
        s=b.states.numpy()[0]
        multi.append(dict(dt=dt,roots=roots,gpu_slip=float(s['slip']),gpu_mode=int(s['mode']),
            residual_at_gpu=fn(float(s['slip'])),residual_at_bounds=[fn(0),fn(.008)]))
        np.savez_compressed(out/f'multiroot-{dt}.npz',params=p,state_before=before,state_after=b.states.numpy(),
                            normal=b.normal.numpy(),xpos=b.data.xpos.numpy(),ids=b.ids.numpy())
    results['multiroot']=multi
    # Do not pre-label every reopening a flaw: distinguish same-event resampling.
    b.reset();p=b.params.numpy();s=b.states.numpy()
    s['gathered']=.003;b.states.assign(s);b.normal.fill_(10.);b.gap.fill_(.003)
    law(b,close=1,region=1);first=b.states.numpy().copy()
    law(b,close=1,region=1);second=b.states.numpy().copy()
    results['same_event']=dict(rng_unchanged=bool(np.array_equal(first['rng'],second['rng'])))
    # Zero material cannot be gathered even with large speed.
    b.reset();p=b.params.numpy();p['budget'][:]=0;b.params.assign(p);b.normal.fill_(1000.);b.gap.fill_(.001)
    for _ in range(20):law(b,close=0,speed=100.,region=0)
    law(b,close=1,speed=100.)
    results['zero_budget']=dict(captured=int((b.states.numpy()['mode']==1).sum()),max_gathered=float(b.states.numpy()['gathered'].max()))
    # Wrong regional budget at the exact same physical contact position.
    regional=[]
    for region in (0,1,2,3):
        b=Batch(32,rigid=True,prototype=0)
        b.normal.fill_(10.);b.gap.fill_(.001)
        for _ in range(400):law(b,speed=.04,region=region)
        law(b,close=1,region=region)
        regional.append(dict(region=region,captured=int((b.states.numpy()['mode']==1).sum()),
            max_gathered=float(b.states.numpy()['gathered'].max()),anchor=b.states.numpy()['anchor'][0]))
    results['region_selection']=regional
    # Actual pad velocities zero, contact inputs held fixed, speed parameter nonzero.
    b=Batch(32,rigid=True,prototype=1)
    b.normal.fill_(10.);b.gap.fill_(.003)
    for _ in range(400):law(b,speed=.04,region=0)
    law(b,close=1,region=0)
    results['stationary_gather']=dict(captured=int((b.states.numpy()['mode']==1).sum()),
        max_gathered=float(b.states.numpy()['gathered'].max()),max_qvel=float(abs(b.data.qvel.numpy()).max()))
    np.savez_compressed(out/'stationary-gather.npz',params=b.params.numpy(),state=b.states.numpy(),qpos=b.data.qpos.numpy(),qvel=b.data.qvel.numpy(),normal=b.normal.numpy(),gap=b.gap.numpy())
    # Contact loss leaves gathered material; the proxy alone supplies capture normals.
    b.reset();s=b.states.numpy();s['gathered']=.003;b.states.assign(s)
    q=b.data.qpos.numpy();addr=b.addresses.numpy();q[:,addr[3,0]]=.003;q[:,addr[4,0]]=.003
    b.data.qpos.assign(q);b.normal.zero_();b.gap.fill_(.006)
    b.data.qfrc_applied.zero_()
    wp.launch(film_proxy,b.nworld,[b.states,b.params,b.data.qpos,b.data.qvel,b.addresses,b.normal,b.data.qfrc_applied])
    law(b,close=1,region=0)
    results['proxy_only_capture']=dict(captured=int((b.states.numpy()['mode']==1).sum()),
        normal=b.normal.numpy()[0],cart_above_payload=float(b.data.xpos.numpy()[0,b.cart,2]-b.data.xpos.numpy()[0,b.ids.numpy()[0,0],2]))
    # Frozen fingers: growth of gathered state creates contact spring energy.
    b.reset();p=b.params.numpy();s=b.states.numpy();s['gathered']=.003;b.states.assign(s)
    q=b.data.qpos.numpy();q[:,addr[3,0]]=.003;q[:,addr[4,0]]=.003;b.data.qpos.assign(q)
    b.normal.fill_(10.);b.gap.fill_(.006)
    e0=b.states.numpy()['gathered'].copy()
    def proxy_energy(e):return 10000.*np.maximum(np.minimum(.002+e,.012)-.003,0)**2
    U0=proxy_energy(e0);law(b,speed=.04,region=0);e1=b.states.numpy()['gathered']
    U1=proxy_energy(e1)
    results['film_energy']=dict(max_delta_U=float((U1-U0).max()),mechanical_work=0.,
        e0=e0[0],e1=e1[0],normal=b.normal.numpy()[0],note='q and qvel held fixed; state growth only')
    # End states: all tension and all external rope forces must clear.
    failures=[]
    for mode in (3,4):
        b.reset();state_for_load(b);s=b.states.numpy();s['mode']=mode;b.states.assign(s)
        law(b);failures.append(dict(mode=mode,max_force=float(abs(b.data.xfrc_applied.numpy()).max()),max_tension=float(abs(b.states.numpy()['tension']).max())))
    results['failed_states']=failures
    # Native-contact classification probe from actual contacts at the bare table.
    b=Batch(1,rigid=True,prototype=1)
    q=b.data.qpos.numpy();addr=b.addresses.numpy();q[0,addr[0,0]]=.20;q[0,addr[2,0]]=.007
    q[0,addr[3,0]]=q[0,addr[4,0]]=.001
    b.data.qpos.assign(q)
    import mujoco_warp as mjw
    mjw.forward(b.model,b.data)
    d=b.data;c=d.contact;b.normal.zero_()
    wp.launch(K.contacts,c.dist.shape[0],[int(b.cpu.opt.cone),d.nacon,c.worldid,c.geom,c.frame,c.friction,c.dim,c.efc_address,c.adhesion,d.efc.force,d.njmax,b.cpu.geom('left').id,b.cpu.geom('right').id,b.normal])
    count=int(d.nacon.numpy()[0]);pairs=c.geom.numpy()[:count]
    results['table_contacts']=dict(normal=b.normal.numpy()[0],pairs=pairs,
        object_geom_ids=[b.cpu.geom('core0').id,b.cpu.geom('capgeom0').id],
        gap=.002,access_distance=float(np.linalg.norm(d.xpos.numpy()[0,b.cart]-d.xpos.numpy()[0,b.ids.numpy()[0,0]])))
    results['table_contacts']['gathered_before']=0.
    # Repeated evaluation of a measured, frozen bare-table input is an adversarial law test.
    native_normal=b.normal.numpy().copy()
    for _ in range(400):b.normal.assign(native_normal);law(b,speed=.04,region=1)
    b.normal.assign(native_normal);law(b,close=1,region=1)
    results['table_contacts']['state_after']=b.states.numpy()[0]
    write(out/'summary.json',results)
    print(out/'summary.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    main(p.parse_args().output)

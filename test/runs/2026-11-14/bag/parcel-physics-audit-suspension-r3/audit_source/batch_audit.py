"""Whole-batch production stepping, native solver diagnostics and response coverage."""
import argparse
from common import ROOT,setup,write
import numpy as np
import mujoco
import warp as wp
from contents.parcel import Batch
from contents.parcel import kernels as K
from payload_audit import set_params,clone,energy


@wp.kernel
def monitor(nrows: wp.array(dtype=int), force: wp.array2d(dtype=float),
            pos: wp.array2d(dtype=float), ma: wp.array2d(dtype=float),
            smooth: wp.array2d(dtype=float), constraint: wp.array2d(dtype=float),
            qvel: wp.array2d(dtype=float), peaks: wp.array2d(dtype=float)):
    e=wp.tid()
    f=float(0);pen=float(0);residual=float(0);velocity=float(0)
    for j in range(nrows[e]):
        f=wp.max(f,wp.abs(force[e,j]));pen=wp.max(pen,wp.max(-pos[e,j],0.0))
    for j in range(qvel.shape[1]):
        residual=wp.max(residual,wp.abs(ma[e,j]-smooth[e,j]-constraint[e,j]))
        velocity=wp.max(velocity,wp.abs(qvel[e,j]))
    peaks[e,0]=wp.max(peaks[e,0],f)
    peaks[e,1]=wp.max(peaks[e,1],pen)
    peaks[e,2]=wp.max(peaks[e,2],residual)
    peaks[e,3]=wp.max(peaks[e,3],velocity)


def run(b,seconds,out,name):
    peaks=wp.zeros((b.nworld,4),dtype=float)
    def step():
        b.step()
        d=b.data
        wp.launch(monitor,b.nworld,[d.nefc,d.efc.force,d.efc.pos,d.efc.Ma,d.qfrc_smooth,d.qfrc_constraint,d.qvel,peaks])
    step();wp.synchronize()
    with wp.ScopedCapture() as cap:step()
    # Reset after capture; restore controlled heterogeneous or identical κ.
    params=b.params.numpy().copy();b.reset();set_params(b,params);peaks.zero_()
    stride=round(.02/b.dt);q=[];v=[];x=[];state=[];times=[]
    for i in range(round(seconds/b.dt)):
        wp.capture_launch(cap.graph)
        if i%stride==0:
            q.append(b.data.qpos.numpy());v.append(b.data.qvel.numpy());x.append(b.data.xpos.numpy());state.append(b.states.numpy());times.append((i+1)*b.dt)
    wp.synchronize()
    arrays=dict(qpos=np.array(q),qvel=np.array(v),xpos=np.array(x),state=np.array(state),time=times,peaks=peaks.numpy(),params=params)
    np.savez_compressed(out/f'{name}.npz',**arrays)
    return arrays


def main(output):
    out=setup(output,'whole GPU batch and timestep audit')
    summary={'static':[],'batch':[],'timestep':[]}
    for prototype in range(6):
        b=Batch(8,rigid=prototype<2,prototype=prototype,level=0)
        data=run(b,2.,out,f'static-p{prototype}')
        # Separate translational free-body speed from mixed-unit joint rates.
        velocity=float(abs(data['qvel'][-1,:,:3]).max())
        ids=b.ids.numpy();p=b.params.numpy();mass=b.model.body_mass.numpy()
        payload=[*ids[:1 if b.rigid else 3,0],*ids[:1 if b.rigid else 3,2]]
        summary['static'].append(dict(prototype=prototype,max_translation_speed=velocity,
            mass_error=float(abs(mass[:,payload].sum(1)-p['mass']).max()),
            max_constraint_force=float(data['peaks'][:,0].max()),max_constraint_penetration=float(data['peaks'][:,1].max()),
            max_force_balance_residual=float(data['peaks'][:,2].max()),finite=bool(np.isfinite(data['qpos']).all())))
    reference=None
    for n in (1,64,512,2048):
        b=Batch(n,prototype=3,level=0,action=1)
        p=b.params.numpy();p[:]=p[0];set_params(b,p)
        data=run(b,1.2,out,f'batch-N{n}')
        first=data['qpos'][:,0]
        if reference is None:reference=first.copy()
        summary['batch'].append(dict(n=n,max_difference_from_N1=float(abs(first-reference).max()),
            max_within_batch_difference=float(abs(data['qpos']-first[:,None]).max())))
    for dt in (.00025,.0005,.001):
        b=Batch(8,prototype=3,dt=dt,level=0,action=1)
        data=run(b,2.,out,f'timestep-{dt}')
        # Exact native energies reconstructed at sampled states, with actual κ.
        cpu,d=clone(b,clamp=False,fixture=True,dt=dt)
        E=[]
        for q,v in zip(data['qpos'][:,0],data['qvel'][:,0]):
            d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(cpu,d)
            E.append(energy(cpu,d,b.params.numpy()[0]))
        np.savez_compressed(out/f'gpu-energy-{dt}.npz',time=data['time'],energy=E)
        summary['timestep'].append(dict(dt=dt,max_constraint_force=float(data['peaks'][:,0].max()),
            max_constraint_penetration=float(data['peaks'][:,1].max()),max_force_balance_residual=float(data['peaks'][:,2].max()),
            final_compression=float(-data['qpos'][-1,:,b.ids.numpy()[0,9]].mean()),max_mechanical_energy=float(np.sum(E,axis=1).max())))
    b=Batch(512,rigid=False,level=0,action=1)
    data=run(b,2.,out,'coverage-segmented')
    p=data['params'];q=data['qpos'];ids=b.ids.numpy()
    depth=-q[:,:,ids[:,9]].max(axis=2)
    depthmax=(-q[:,:,ids[:,9]]).max(axis=(0,2))
    target=10/p['compression'];geometric=.2*p['size'][:,2]
    # Correlation describes this prescribed-displacement fixture only.
    summary['coverage']=dict(n=512,unreachable_20N_target_by_core_geometry=int((target>geometric).sum()),
        target_range=[float(target.min()),float(target.max())],depth_range=[float(depthmax.min()),float(depthmax.max())],
        target_vs_depth_pearson=float(np.corrcoef(target,depthmax)[0,1]),
        unique_native_mass=int(len(np.unique(p['mass']))),unique_compression=int(len(np.unique(p['compression']))))
    b=Batch(512,rigid=True)
    p=b.params.numpy()
    summary['coverage']['rigid_dynamically_unused_fields']=['bend','twist','damping','compression','quadratic','weights']
    np.savez_compressed(out/'coverage-rigid.npz',params=p,body_mass=b.model.body_mass.numpy(),geom_size=b.model.geom_size.numpy())
    # Graph warmup/capture and reset side effects.
    before=b.clock.numpy().copy();b.graph();after=b.clock.numpy().copy()
    b.data.xfrc_applied.fill_(3.);b.data.qfrc_applied.fill_(4.)
    s=b.states.numpy();s['mode']=1;s['tension']=2.;b.states.assign(s)
    b.reset()
    summary['reset']=dict(clock_before=int(before[0]),clock_after_graph=int(after[0]),
        state_nonzero_after_reset=bool((b.states.numpy()['mode']!=0).any()),
        external_after_reset=float(abs(b.data.xfrc_applied.numpy()).max()),
        generalized_after_reset=float(abs(b.data.qfrc_applied.numpy()).max()))
    write(out/'summary.json',summary)
    print(out/'summary.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(parser.parse_args().output)

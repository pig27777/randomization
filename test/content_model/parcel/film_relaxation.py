"""Compare isolated membrane rebound with the same mass, geometry and deflection."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import mujoco,numpy as np
from film_material_fixture import model
from film_checks import manifest


def main(output,seconds):
    output.mkdir(parents=True,exist_ok=False);manifest(output,dt=.0000625,seconds=seconds)
    summary=[]
    for variant in ('previous','damped'):
        m,d,s,o=model();m.opt.timestep=.0000625
        p=s.parameters.numpy()
        if variant=='previous':
            p['stretch']/=3.;p['damping_ratio']-=.2;p['bend_damping_ratio']=0
            s.parameters.assign(p)
        d.mocap_pos[:]=np.array([[2,2,2],[-2,-2,2]])
        rest=s.rest.numpy()[0];bodies=s.bodies.numpy();dofs=s.dofs.numpy();mass=s.masses.numpy()[0]
        nodes=np.flatnonzero((s.topo.vertices[:,2]==.5)&(np.abs(s.topo.vertices[:,0])<.5)&(np.abs(s.topo.vertices[:,1])<.5))
        for v in nodes:
            q=m.jnt_qposadr[m.body_jntadr[bodies[v]]]
            d.qpos[q+2]+=.005
        mujoco.mj_forward(m,d)
        import warp as wp
        buffers=dict(xpos=wp.array(d.xpos[None].astype('f'),dtype=wp.vec3,device='cpu'),
            qvel=wp.array(d.qvel[None].astype('f'),dtype=float,device='cpu'),
            forces=wp.zeros((1,m.nbody),dtype=wp.spatial_vector,device='cpu'))
        rows=[];initial_com=np.average(rest,axis=0,weights=mass)
        for i in range(round(seconds/m.opt.timestep)):
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            buffers['xpos'].assign(d.xpos[None].astype('f'));buffers['qvel'].assign(d.qvel[None].astype('f'))
            buffers['forces'].zero_();s.apply(buffers['xpos'],buffers['qvel'],buffers['forces'])
            d.xfrc_applied[:]=buffers['forces'].numpy()[0]
            if i%80==0:
                x=d.xpos[bodies];v=np.array([d.qvel[adr:adr+3] for adr in dofs])
                com=np.average(x,axis=0,weights=mass)
                displacement=float((x[nodes,2]-rest[nodes,2]).mean()-(com-initial_com)[2])
                energy=float(s.energy.numpy()[0]+.5*(mass[:,None]*v*v).sum())
                rows.append(dict(t=float(d.time),displacement=displacement,total_energy=energy,power=float(s.power.numpy()[0])))
            mujoco.mj_step(m,d)
        initial=rows[0]['total_energy'];final=rows[-1]['total_energy']
        result=dict(variant=variant,initial_energy=initial,final_energy=final,residual_energy_fraction=final/initial,
                    finite=bool(np.isfinite(d.qpos).all()),parameters={n:p[n][0].tolist() for n in p.dtype.names})
        summary.append(result)
        (output/f'{variant}.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(result),flush=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    assert all(x['finite'] for x in summary)
    assert summary[1]['residual_energy_fraction'] < summary[0]['residual_energy_fraction'], summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=1.)
    a=p.parse_args();main(a.output,a.seconds)

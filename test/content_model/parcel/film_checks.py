"""Physical checks for the closed membrane; run under the shared GPU lock."""
from pathlib import Path
import argparse,json,sys,time,hashlib,platform,datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import mujoco
import numpy as np
import warp as wp
from contents.parcel.model import Batch
from contents.parcel.film import topology,settings


def manifest(out,**configuration):
    files=list(Path('contents/parcel').glob('*.py'))+list(Path('contents/parcel').glob('*.json'))
    data=dict(machine=platform.node(),time=datetime.datetime.now().astimezone().isoformat(),
        mujoco=mujoco.__version__,warp=wp.__version__,configuration=configuration,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'metadata.json').write_text(json.dumps(data,indent=2))


def energy_checks(b):
    s=b.surface
    rest=s.rest.numpy();x=rest.copy()
    x[:,0]+=[.001,-.001,.002]
    s.points.assign(x);s.elastic_forces()
    E=s.energy.numpy().copy();gradient=s.points.grad.numpy().copy()
    force=-gradient
    net_force=float(np.linalg.norm(force.sum(1),axis=1).max())
    net_torque=float(np.linalg.norm(np.cross(x,force).sum(1),axis=1).max())
    numerical=[];eps=2e-5
    for axis in range(3):
        plus=x.copy();plus[0,0,axis]+=eps;s.points.assign(plus);s.elastic_forces();up=s.energy.numpy()[0]
        minus=x.copy();minus[0,0,axis]-=eps;s.points.assign(minus);s.elastic_forces();down=s.energy.numpy()[0]
        numerical.append((up-down)/(2*eps))
    relative=float(np.linalg.norm(np.array(numerical)-gradient[0,0])/np.linalg.norm(gradient[0,0]))
    angle=.73;R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    # Reference and current positions rotate together; material rest lengths/angles are objective.
    s.points.assign(x@R.T+np.array([.4,-.2,.7]));s.elastic_forces()
    rotation_relative=float(np.max(np.abs(s.energy.numpy()-E)/np.maximum(E,1e-10)))
    s.points.assign(x);s.elastic_forces()
    return dict(energy=E.tolist(),net_force=net_force,net_torque=net_torque,
                gradient_relative_error=relative,rigid_transform_relative_error=rotation_relative,
                gradient_pass=relative<.01,force_pass=net_force<1e-4,torque_pass=net_torque<1e-4,
                rotation_pass=rotation_relative<.01)


def main(out,n,seconds,dt=.0000625):
    out.mkdir(parents=True,exist_ok=False)
    manifest(out,n=n,seconds=seconds,dt=dt)
    t=topology(settings()['subdivisions'])
    result=dict(topology=dict(vertices=len(t.vertices),triangles=len(t.triangles),edges=len(t.edges),
                  faces={name:int((t.face_id==i).sum()) for i,name in enumerate(('TOP','BOTTOM','LEFT','RIGHT','FRONT','BACK'))},
                  closed=bool(len(t.vertices)-len(t.edges)+len(t.triangles)==2)))
    (out/'topology.json').write_text(json.dumps(dict(vertices=t.vertices.tolist(),triangles=t.triangles.tolist(),
        face_id=t.face_id.tolist(),edges=t.edges.tolist(),adjacent=t.adjacent.tolist()),indent=2))
    result['runs']=[]
    for rigid in (True,False):
        b=Batch(n,rigid=rigid,envelope=True,dt=dt)
        if n==1:result['energy_rigid' if rigid else 'energy_segmented']=energy_checks(b)
        graph=b.graph();b.reset()
        samples=[]
        start=time.perf_counter()
        for i in range(round(seconds/b.dt)):
            wp.capture_launch(graph)
            if i%100==0:
                film=b.data.flexvert_xpos.numpy()
                edges=b.surface.topo.edges
                ratio=np.linalg.norm(film[:,edges[:,1]]-film[:,edges[:,0]],axis=2)/b.surface.lengths.numpy()
                samples.append(dict(time=i*b.dt,finite=bool(np.isfinite(b.data.qpos.numpy()).all()),
                    max_velocity=float(np.abs(b.data.qvel.numpy()).max()),
                    max_edge_ratio=float(ratio.max()),
                    energy=float(b.surface.energy.numpy().max()),broken=int(b.surface.damage.numpy().sum()),overflow=int(b.data.overflow.numpy().max())))
        wp.synchronize();elapsed=time.perf_counter()-start
        np.savez_compressed(out/f'final-{rigid}.npz',qpos=b.data.qpos.numpy(),qvel=b.data.qvel.numpy(),
            film=b.data.flexvert_xpos.numpy(),rest=b.surface.rest.numpy(),parameters=b.surface.parameters.numpy(),
            masses=b.surface.masses.numpy())
        row=dict(rigid=rigid,n=n,seconds=seconds,finite=all(x['finite'] for x in samples),
                 samples=samples,wall_seconds=elapsed,env_steps_s=n*round(seconds/b.dt)/elapsed,
                 final_energy=float(b.surface.energy.numpy().max()),broken=int(b.surface.damage.numpy().sum()),overflow=int(b.data.overflow.numpy().max()))
        result['runs'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='samples'}),flush=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    assert all(x['finite'] and x['overflow']==0 and x['broken']==0
               and all(s['max_edge_ratio']<2 for s in x['samples']) for x in result['runs']), 'Unstable settling or solver failure'
    for key,value in result.items():
        if key.startswith('energy_'):assert all(v for k,v in value.items() if k.endswith('_pass')), value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--dt',type=float,default=.0000625);p.add_argument('--n',type=int,default=1);p.add_argument('--seconds',type=float,default=2.)
    a=p.parse_args();main(a.output,a.n,a.seconds,a.dt)

"""Native six-direction probes and membrane support/containment diagnostics."""
from pathlib import Path
import argparse,json,sys,xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from contents.parcel.model import Batch,xml,ids_for_surface
from contents.parcel.film import topology,settings,add_xml,FACES
from film_checks import manifest


def coverage(rotation=None, gravity=None):
    cfg=settings();t=topology(cfg['subdivisions']);dims=np.array([.28,.18,.075]);center=np.array([0,0,.039])
    points=t.vertices*dims+center
    root=add_xml(ET.fromstring(xml(True)),points,t,cfg)
    root.find('option').set('gravity','0 0 0')
    world=root.find('worldbody');world.remove(world.find("body[@name='cart']"))
    probe=ET.SubElement(world,'body',name='probe',mocap='true')
    ET.SubElement(probe,'geom',name='probe',type='sphere',size='.006',contype='4',conaffinity='2',priority='1')
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m)
    transform=np.eye(3) if rotation is None else np.asarray(rotation)
    if rotation is not None:
        nodes=[m.body(f'film_{v}').id for v in range(len(t.vertices))]
        m.body_pos[nodes]=m.body_pos[nodes]@transform.T
    if gravity is not None:
        m.opt.gravity[:]=transform@np.asarray(gravity)
        for geom in ids_for_surface(m,True)+[m.geom('table').id]:
            m.geom_contype[geom]=0;m.geom_conaffinity[geom]=0
    mujoco.mj_setConst(m,d)
    rows=[]
    for face,(axis,sign) in enumerate(((2,1),(2,-1),(0,-1),(0,1),(1,-1),(1,1))):
        mujoco.mj_resetData(m,d)
        normal=np.eye(3)[axis]*sign
        location=center+normal*dims[axis]/2
        d.mocap_pos[0]=transform@(location+normal*(cfg['contact_radius']+.006-.0005))
        mujoco.mj_forward(m,d)
        load=0.;count=0;vertex=-1;buffer=np.zeros(6);force=np.zeros(3)
        for c in range(d.ncon):
            con=d.contact[c]
            if m.geom('probe').id in con.geom and 0 in con.flex:
                mujoco.mj_contactForce(m,d,c,buffer)
                load+=max(buffer[0],0);count+=1
                force+=(1 if con.geom[1]==m.geom('probe').id else -1)*con.frame.reshape(3,3).T@buffer[:3]
                fside=0 if con.flex[0]==0 else 1
                vertex=int(con.vert[fside])
        rows.append(dict(face=FACES[face],contacts=count,normal_load=load,pass_contact=bool(load>0),vertex=vertex,
                         contact_force_world=force.tolist()))
    return rows


def winding(points,vertices,triangles):
    result=[]
    for p in points:
        a=vertices[triangles[:,0]]-p;b=vertices[triangles[:,1]]-p;c=vertices[triangles[:,2]]-p
        la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
        top=np.einsum('ij,ij->i',a,np.cross(b,c))
        bottom=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
        result.append(abs(np.sum(2*np.arctan2(top,bottom)))/(4*np.pi))
    return np.array(result)


def surface_distance(points,vertices,triangles):
    distance=[]
    for p in points:
        best=float('inf')
        for ids in triangles:
            a,b,c=vertices[ids]
            ab=b-a;ac=c-a;n=np.cross(ab,ac)
            norm=np.linalg.norm(n)
            if norm<1e-12:continue
            n=n/norm
            projected=p-np.dot(p-a,n)*n
            dots=[ab@ab,ab@ac,ac@ac,(projected-a)@ab,(projected-a)@ac]
            den=dots[0]*dots[2]-dots[1]**2
            u=(dots[2]*dots[3]-dots[1]*dots[4])/max(den,1e-20)
            v=(dots[0]*dots[4]-dots[1]*dots[3])/max(den,1e-20)
            if u>=0 and v>=0 and u+v<=1:best=min(best,abs(np.dot(p-a,n)))
            for x,y in ((a,b),(b,c),(c,a)):
                t=np.clip((p-x)@(y-x)/max((y-x)@(y-x),1e-20),0,1)
                best=min(best,np.linalg.norm(p-(x+t*(y-x))))
        distance.append(best)
    return np.array(distance)


def diagnostics(b):
    d=b.data;m=b.model
    count=int(d.nacon.numpy()[0])
    force=wp.empty(count,dtype=wp.spatial_vector,device=b.device)
    mjw.contact_force(m,d,wp.array(np.arange(count),dtype=int,device=b.device),True,force)
    f=force.numpy();geoms=d.contact.geom.numpy()[:count];flex=d.contact.flex.numpy()[:count]
    world=d.contact.worldid.numpy()[:count]
    support=np.zeros(b.nworld);bare=0
    table=b.cpu.geom('table').id;core=set(ids_for_surface(b.cpu,b.rigid))
    for i,(g,fl) in enumerate(zip(geoms,flex)):
        if table in g:
            if any(x in core for x in g):bare+=1
            if 0 in fl:
                support[world[i]]+=(1 if g[0]==table else -1)*f[i,2]
    weight=(b.params.numpy()['mass']+b.surface.masses.numpy().sum(1))*9.81
    vertices=d.flexvert_xpos.numpy();gx=d.geom_xpos.numpy();gm=d.geom_xmat.numpy();size=m.geom_size.numpy()
    samples=[]
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for e in range(b.nworld):
        points=np.concatenate([corners*size[e,g]@gm[e,g].T+gx[e,g] for g in core])
        w=winding(points,vertices[e],b.surface.topo.triangles)
        distances=surface_distance(points[w<.9],vertices[e],b.surface.topo.triangles)
        samples.append(dict(min_winding=float(w.min()),outside_mid_surface=int((w<.9).sum()),
            max_outside_distance=float(distances.max()) if len(distances) else 0.,
            beyond_contact_skin=int((distances>b.surface.cfg['contact_radius']).sum())))
    return dict(support_ratio=(support/weight).tolist(),bare_core_table_contacts=bare,containment=samples,
                max_contact_penetration=float(max(0,-d.contact.dist.numpy()[:count].min())) if count else 0.)


def main(output,prototypes=None):
    output.mkdir(parents=True,exist_ok=False)
    manifest(output,prototypes=prototypes,seconds=2.0)
    result=dict(coverage=coverage(),support=[])
    for rigid in (True,False):
        for proto in ((0,1) if rigid else (2,3,4,5)):
            if prototypes is not None and proto not in prototypes:continue
            b=Batch(1,rigid=rigid,prototype=proto,envelope=True)
            g=b.graph();b.reset()
            initial=diagnostics(b)
            for _ in range(round(2/b.dt)):wp.capture_launch(g)
            wp.synchronize()
            final=diagnostics(b)
            row=dict(prototype=proto,initial=initial,final=final,finite=bool(np.isfinite(b.data.qpos.numpy()).all()),overflow=int(b.data.overflow.numpy().max()))
            result['support'].append(row)
            print(json.dumps(row),flush=True)
            np.savez_compressed(output/f'p{proto}.npz',film=b.data.flexvert_xpos.numpy(),qpos=b.data.qpos.numpy(),
                rest=b.surface.rest.numpy(),parameters=b.surface.parameters.numpy(),masses=b.surface.masses.numpy())
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print('coverage',result['coverage'])
    assert all(x['pass_contact'] for x in result['coverage']), 'Uncovered face'
    for x in result['support']:
        assert x['finite'] and x['overflow']==0, x
        assert x['final']['bare_core_table_contacts']==0, x
        assert all(.95<=v<=1.05 for v in x['final']['support_ratio']), x
        assert all(v['beyond_contact_skin']==0 for v in x['final']['containment']), x


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--prototypes',default='0,1,2,3,4,5')
    a=p.parse_args();main(a.output,[int(i) for i in a.prototypes.split(',')])

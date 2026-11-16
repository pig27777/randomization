"""Isolated membrane-clamp tests on each face, not robot grasp demonstrations.

Two fixed spherical jaws prepare a membrane patch between them. Payload contact
is disabled in this material fixture. All deformation/loads remain dynamic and
use the same Warp elastic forces as the parcel model.
"""
from pathlib import Path
import argparse,json,sys,xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import mujoco,numpy as np,warp as wp
from contents.parcel.model import xml,ids_for_surface
from contents.parcel import kernels as K
from contents.parcel.film import topology,settings,add_xml,Surface,FACES
from contents.parcel.film_contacts import Observer
from film_checks import manifest


def model():
    cfg=settings();t=topology();dims=np.array([.28,.18,.09]);center=np.array([0,0,.047])
    root=add_xml(ET.fromstring(xml(True,.0000625)),t.vertices*dims+center,t,cfg)
    root.find('option').set('gravity','0 0 0')
    world=root.find('worldbody');world.remove(world.find("body[@name='cart']"))
    for g in world.findall('.//geom'):g.set('contype','0');g.set('conaffinity','0')
    for side in ('left','right'):
        body=ET.SubElement(world,'body',name=side,mocap='true')
        ET.SubElement(body,'geom',name=side,type='sphere',size='.006',contype='4',conaffinity='2',
                      priority='1',solref='.001 1',solimp='.99 .99 .001')
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m)
    surface=Surface(m,1,'cpu')
    params=wp.empty(1,dtype=K.Parameters,device='cpu');state=wp.zeros(1,dtype=K.State,device='cpu')
    bounds=json.loads(Path('contents/parcel/config.json').read_text())
    wp.launch(K.sample,1,[7,0,1,1,wp.array(list(bounds.values()),dtype=wp.vec2,device='cpu'),params,state],device='cpu')
    position=wp.array(m.body_pos[None].astype('f'),dtype=wp.vec3,device='cpu')
    mass=wp.array(m.body_mass[None].astype('f'),dtype=float,device='cpu')
    friction=wp.array(m.geom_friction[None].astype('f'),dtype=wp.vec3,device='cpu')
    armature=wp.array(m.dof_armature[None].astype('f'),dtype=float,device='cpu')
    surface.initialize(params,position,mass,friction,wp.array(ids_for_surface(m,True),dtype=int,device='cpu'),
        m.geom('table').id,wp.array([m.geom('left').id,m.geom('right').id],dtype=int,device='cpu'),armature=armature)
    m.body_pos[:]=position.numpy()[0];m.body_mass[:]=mass.numpy()[0]
    m.geom_friction[:]=friction.numpy()[0];m.dof_armature[:]=armature.numpy()[0]
    mujoco.mj_setConst(m,d);mujoco.mj_resetData(m,d)
    observer=Observer(m,surface,[{m.geom('left').id},{m.geom('right').id}],device='cpu',capacity=2048)
    return m,d,surface,observer


def main(out):
    out.mkdir(parents=True,exist_ok=False);manifest(out,fixture='empty membrane with two fixed jaws',dt=.0000625)
    rows=[]
    for face,(axis,sign) in enumerate(((2,1),(2,-1),(0,-1),(0,1),(1,-1),(1,1))):
        for load in (.02,1.):
            m,d,s,o=model();normal=np.eye(3)[axis]*sign
            rest=s.rest.numpy()[0];verts=np.unique(s.topo.triangles[s.topo.face_id==face])
            point=rest[verts].mean(0)
            d.mocap_pos[0]=point-normal*.0065;d.mocap_pos[1]=point+normal*.0065
            tangent=np.eye(3)[(axis+1)%3]
            mujoco.mj_forward(m,d);o.observe_cpu(m,d,close=1)
            initial=o.states.numpy()[0].copy()
            buffers=dict(xpos=wp.array(d.xpos[None].astype('f'),dtype=wp.vec3,device='cpu'),
                qvel=wp.array(d.qvel[None].astype('f'),dtype=float,device='cpu'),
                forces=wp.zeros((1,m.nbody),dtype=wp.spatial_vector,device='cpu'))
            history=[]
            for i in range(round(.1/m.opt.timestep)):
                mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
                buffers['xpos'].assign(d.xpos[None].astype('f'));buffers['qvel'].assign(d.qvel[None].astype('f'))
                buffers['forces'].zero_();s.apply(buffers['xpos'],buffers['qvel'],buffers['forces'])
                d.xfrc_applied[:]=buffers['forces'].numpy()[0]
                d.xfrc_applied[s.bodies.numpy(),:3]+=tangent*load/len(rest)
                mujoco.mj_step(m,d);o.observe_cpu(m,d,close=1)
                if i%80==0:
                    st=o.states.numpy()[0]
                    history.append(dict(t=float(d.time),mode=int(st['mode']),slip=float(st['slip_distance']),normal=float(st['normal_load'])))
            state=o.states.numpy()[0]
            row=dict(face=FACES[face],load=load,initial_face=int(initial['face_id']),initial_mode=int(initial['mode']),
                     initial_normal=float(initial['normal_load']),final_mode=int(state['mode']),
                     slip=float(state['slip_distance']),finite=bool(np.isfinite(d.qpos).all()),history=history)
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
    (out/'summary.json').write_text(json.dumps(rows,indent=2))
    assert all(x['initial_mode']==1 and x['initial_face']==FACES.index(x['face']) and x['finite'] for x in rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);main(p.parse_args().output)

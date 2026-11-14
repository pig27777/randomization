"""Parcel material model adapter for the shared YAM protocol.

CPU YAM validation uses the same Warp force kernel on the CPU device.
GPU topology batches remain in contents.parcel.model, with no host transfers per step.
"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import warp as wp
from contents.parcel import kernels as K
from contents.parcel.model import xml
from experience.yam_platform.protocol import TrialConfig,MuJoCoSim,attach_robot,measure_contacts,tips,vec


@dataclass
class Config(TrialConfig):
    prototype: int = 3
    seed: int = 7
    region: int = 0
    grasp_face: str = "top"
    side_pitch_deg: float = 0.0

    def __post_init__(self):
        if self.grasp_face not in ("top", "side") or self.region not in range(4):
            raise ValueError("Invalid grasp face or material region")
        if not np.isfinite(self.side_pitch_deg) or not 0 <= self.side_pitch_deg < 90:
            raise ValueError("Side approach pitch must be in [0,90) degrees")


def scene(cfg):
    import json
    bounds=json.loads((Path(K.__file__).with_name('config.json')).read_text())
    names=('length','width','height','mass','loose_slack','rope_stiffness','static_friction','table_friction','film_strength','segment_weight','com_fraction',
           'bend_sag','soft_sag','twist_ratio','damping_ratio','compression_depth',
           'soft_compression_depth','slack_y_ratio')
    params=wp.empty(1,dtype=K.Parameters,device='cpu')
    states=wp.zeros(1,dtype=K.State,device='cpu')
    rigid=cfg.prototype<2
    wp.launch(K.sample,1,[cfg.seed,0,int(rigid),cfg.prototype,
        wp.array([bounds[k] for k in names],dtype=wp.vec2,device='cpu'),params,states],device='cpu')
    p=params.numpy()[0]
    size=p['size']; n=1 if rigid else 3
    root=ET.fromstring(xml(rigid,cfg.dt))
    ET.SubElement(root,'equality')
    root.find(".//geom[@name='table']").set('friction',f"{p['table_mu']} .005 .0001")
    world=root.find('worldbody')
    world.remove(world.find("body[@name='cart']"))
    cell=size/np.array([n,1,1])
    for i in range(n):
        body=root.find(f".//body[@name='segment{i}']")
        position=np.array([cell[0],0,0]) if i else np.array([-.25-(0 if rigid else size[0]/3),.04,size[2]*.4+.001])
        body.set('pos',vec(position))
        mass=p['mass']*(1 if rigid else p['weights'][i])
        shift=p['com'].copy()
        if not rigid:shift[0]-=cell[0]*(p['weights'][2]-p['weights'][0])
        for name,fraction,dims,offset in ((f'segment{i}',.9,cell*[1,1,.8],shift),
                                         (f'cap{i}',.1,cell*[1,1,.2],shift)):
            b=root.find(f".//body[@name='{name}']")
            inertial=b.find('inertial')
            inertial.set('mass',str(mass*fraction))
            inertial.set('pos',vec(offset))
            inertial.set('diaginertia',vec(mass*fraction/12*(np.sum(dims**2)-dims**2)))
        for name,half in ((f'core{i}',cell*[.49,.5,.4]),(f'capgeom{i}',cell*[.49,.5,.1])):
            g=root.find(f".//geom[@name='{name}']")
            g.set('size',vec(half));g.set('friction',f"{p['table_mu']} .005 .0001")
            g.set('contype','1');g.set('conaffinity','5')
        cap=root.find(f".//body[@name='cap{i}']")
        cap.set('pos',vec([0,0,size[2]*.5]))
        if not rigid:
            joint=cap.find('joint');joint.set('stiffness',str(p['compression']))
            joint.set('damping',str(2*p['damping']*np.sqrt(p['compression']*mass*.1)))
        if i:
            for j,joint in enumerate(body.findall('joint')):
                k=p['twist'] if j==0 else p['bend']
                joint.set('pos',vec([-cell[0]/2,0,0]));joint.set('stiffness',str(k))
                joint.set('damping',str(2*p['damping']*np.sqrt(k*mass*cell[0]**2/12)))
    built=SimpleNamespace(xml=ET.tostring(root,encoding='unicode'),desc=SimpleNamespace(),cfg=cfg,
                          params=params,states=states,rigid=rigid)
    center=np.array([-.25,.04])
    approach=None
    if cfg.grasp_face == "side":
        pitch=np.radians(cfg.side_pitch_deg)
        approach=dict(direction=np.array([0.,np.cos(pitch),-np.sin(pitch)]),closing=np.array([-1.,0.,0.]),
                      contact_point=np.array([center[0],center[1]-size[1]/2,.001+size[2]/2]))
    robot=attach_robot(cfg,built,ParcelSim,center,size[2]+.001,size,float(p['mass']),approach=approach)
    robot.sim.robot=robot
    return robot


class ParcelSim(MuJoCoSim):
    def __init__(self,built):
        super().__init__(built)
        self.params,self.states=built.params,built.states
        m=self.m
        ids=np.zeros((3,10),dtype=np.int32)
        for i in range(1 if built.rigid else 3):
            ids[i,0]=m.body(f'segment{i}').id
            if not built.rigid:
                j=m.joint(f'compress{i}').id
                ids[i,7]=m.jnt_dofadr[j];ids[i,9]=m.jnt_qposadr[j]
        self.ids=wp.array(ids,dtype=int,device='cpu')
        self.core=ids[:1 if built.rigid else 3,0]
        self.obj_geoms={m.geom(f'{prefix}{i}').id for i in range(1 if built.rigid else 3) for prefix in ('core','capgeom')}
        self.cart=m.body('pinch_frame').id
        self.normal=wp.zeros((1,2),device='cpu')
        self.gap=wp.zeros(1,device='cpu')
        self.buffers={}
        self.close=0;self.releasing=0;self.previous_gap=None
        self.events=[]

    def com(self):
        mass=self.m.body_mass[self.core]
        return np.average(self.d.xipos[self.core],weights=mass,axis=0)

    def apply(self):
        d,m=self.d,self.m
        if not hasattr(self,'robot'):
            return
        gap=self.robot.gap()
        speed=0.0 if self.previous_gap is None else (self.previous_gap-gap)/m.opt.timestep
        self.previous_gap=gap
        loads=np.zeros((1,2),dtype=np.float32)
        contact=np.zeros(6)
        for i,c in enumerate(d.contact):
            if not any(g in self.obj_geoms for g in c.geom):continue
            mujoco.mj_contactForce(m,d,i,contact)
            for j,group in enumerate(self.robot.all_finger_sets):
                if any(g in group for g in c.geom):loads[0,j]+=max(contact[0],0)
        state=self.states.numpy()
        proxy=self.film_spring.last_force if hasattr(self,'film_spring') else 0.0
        if state['mode'][0] in (1,2):
            loads[:]=proxy
        else:
            midpoint=tips(self.robot).mean(0)
            state['anchor'][0]=d.xmat[self.cart].reshape(3,3).T@(midpoint-d.xpos[self.cart])
            self.states.assign(state)
            loads+=proxy
        self.normal.assign(loads)
        self.gap.assign(np.array([gap],dtype=np.float32))
        arrays=dict(xpos=d.xpos[None].astype('f'),xmat=d.xmat.reshape(1,m.nbody,3,3).astype('f'),
                    xipos=d.xipos[None].astype('f'),qpos=d.qpos[None].astype('f'),
                    qfrc=d.qfrc_applied[None].astype('f'),forces=d.xfrc_applied[None].astype('f'))
        types=dict(xpos=wp.vec3,xmat=wp.mat33,xipos=wp.vec3,qpos=float,qfrc=float,forces=wp.spatial_vector)
        for name,value in arrays.items():
            if name not in self.buffers:self.buffers[name]=wp.array(value,dtype=types[name],device='cpu')
            else:self.buffers[name].assign(value)
        b=self.buffers
        wp.launch(K.dynamics,1,[m.opt.timestep,int(self.built.rigid),2,self.cart,self.built.cfg.region,
            self.close,self.releasing,0,max(speed,0),self.params,self.states,self.ids,self.normal,self.gap,
            b['xpos'],b['xmat'],b['xipos'],b['qpos'],b['qfrc'],b['forces']],device='cpu')
        d.qfrc_applied[:]=b['qfrc'].numpy()[0]
        d.xfrc_applied[:]=b['forces'].numpy()[0]
        self.releasing=0

    def step(self):
        mujoco.mj_kinematics(self.m,self.d)
        mujoco.mj_comPos(self.m,self.d)
        self.apply()
        mujoco.mj_step(self.m,self.d)
        self.d.qfrc_applied[:]=0


class Attachment:
    def __init__(self,robot):self.sim=robot.sim
    @property
    def active(self):return {} if self.sim.states.numpy()['mode'][0] in (1,2) else None
    @property
    def load(self):return float(self.sim.states.numpy()['tension'].sum())
    @property
    def events(self):return self.sim.events
    def try_bind(self,current_tips):
        self.sim.close=1
        qfrc=self.sim.d.qfrc_applied.copy()
        xfrc=self.sim.d.xfrc_applied.copy()
        self.sim.apply()
        self.sim.d.qfrc_applied[:]=qfrc
        self.sim.d.xfrc_applied[:]=xfrc
        self.sim.events.append(dict(time=float(self.sim.d.time),state=int(self.sim.states.numpy()['mode'][0]),
                                    reason=int(self.sim.states.numpy()['reason'][0]),
                                    gathered=float(self.sim.states.numpy()['gathered'][0]),
                                    normal=self.sim.normal.numpy()[0].tolist(),gap=float(self.sim.robot.gap()),
                                    probability=float(self.sim.states.numpy()['probability'][0])))
        return self.active is not None
    def check_load(self,*args):return self.active is None
    def release(self,*args):self.sim.releasing=1;self.sim.close=0


def attachment(robot,cfg):return Attachment(robot)
def measure(robot):return measure_contacts(robot,target_geoms=robot.sim.obj_geoms)
def check_state(sim):return (None if np.isfinite(sim.d.qpos).all() else 'nonfinite'),{}
def summarize(trace):return {}
def dimensions(sim):return dict(nv=sim.m.nv,rigid_bodies=len(sim.core))
def source_files():return {f'parcel_{p.stem}':p for p in Path(K.__file__).parent.glob('*.py')}


class MaterialSpring:
    """Gathered material supports normal load only when the fingers reach it."""
    def __init__(self,sim):
        self.sim=sim
        self.reference_gap=None
        self.last_force=0.0
        self.previous_gap=None
    def arm(self,gap,eligible):
        self.reference_gap=gap if eligible else None
    def force(self,gap):
        s=self.sim.states.numpy()[0]
        p=self.sim.params.numpy()[0]
        available=s['gathered']>p['threshold'] and s['mode']<3
        velocity=0.0 if self.previous_gap is None else (gap-self.previous_gap)/self.sim.m.opt.timestep
        self.previous_gap=gap
        length=float(s["captured"] if s["mode"] in (1,2) else s["gathered"])
        width=min(.004+2*length,.024)
        self.last_force=max(5000.0*(width-gap)-velocity,0.0) if available and gap<width and self.reference_gap is not None else 0.0
        return self.last_force
    def release(self):
        self.reference_gap=None
        self.last_force=0.0


def gathering_spring(robot,cfg):
    robot.sim.film_spring=MaterialSpring(robot.sim)
    return robot.sim.film_spring

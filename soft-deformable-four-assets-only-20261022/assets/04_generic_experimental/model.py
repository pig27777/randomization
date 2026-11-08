"""MuJoCo 3.14 flex membrane + conservative nonlinear dihedral bending.

One process owns one callback/model. Forces map through validated world-aligned
slide DOFs; unrelated robot/external forces are never overwritten.
"""
from pathlib import Path
import json
import math
import time
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
from shell import make_mesh, topology_report, compute_bag_hinge_batch, smoothstep
from rheology import wrap, moment, potential, advance, plastic_return, crease_history

ROOT = Path(__file__).resolve().parent
DEFAULTS = dict(nx=10, ny=10, width_mm=180., height_mm=240., mass_g=14.,
    young_pa=2e7, poisson=.3, thickness_mm=.4, bend=.0004, seam=6.,
    knee_deg=12., post_ratio=.2, bend_damping=2e-7, drag=120., rayleigh=.004,
    travel_mm=33., ramp_s=2., grip_height=.6, gravity=True, self_contact=True,
    ipc=True, timestep=.0005, frame_dt=.02, iterations=200,tolerance=1e-6,
    rest_gap_mm=4., imperfection_mm=.7, radius_mm=.15, margin_mm=.1,
    memory_ratio=.8, recovery_tau_s=.6,plasticity=False,
    longitudinal_stiffness=6.,rigid_seams=True,bottom_mode='sealed',release_s=.6,
    centered_grip=False,stiff_bottom=False,grip_patch_mm=24.,crease_memory=False,
    crease_onset_deg=25.,crease_width_deg=35.,crease_max_softening=.65)


def validate(values):
    p = {**DEFAULTS, **values}
    bounds = dict(nx=(4,30),ny=(4,40),width_mm=(120,350),height_mm=(150,450),
        mass_g=(2,500),young_pa=(1e4,1e8),poisson=(0,.49),thickness_mm=(.05,8),
        bend=(1e-5,.01),seam=(1,20),knee_deg=(.2,90),post_ratio=(.001,1),
        bend_damping=(0,1e-5),drag=(0,300),rayleigh=(0,.1),travel_mm=(0,60),
        ramp_s=(.5,6),grip_height=(.2,1),timestep=(.0001,.002),frame_dt=(.01,.05),
        iterations=(10,600),tolerance=(1e-9,1e-3),rest_gap_mm=(.2,12),imperfection_mm=(0,2),
        radius_mm=(.02,4),margin_mm=(0,1),memory_ratio=(0,3),recovery_tau_s=(.1,5),longitudinal_stiffness=(1,12),release_s=(0,2),grip_patch_mm=(6,40),crease_onset_deg=(5,120),crease_width_deg=(5,120),crease_max_softening=(0,.85))
    for k,v in p.items():
        if k=='bottom_mode':
            if v not in ('sealed','plastic'):raise ValueError('bottom_mode: sealed or plastic required')
        elif k in ("gravity","self_contact","ipc","plasticity","rigid_seams","centered_grip","stiff_bottom","crease_memory"):
            if type(v) is not bool: raise ValueError(f"{k}: boolean required")
        elif k in bounds:
            lo,hi = bounds[k]
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not lo<=v<=hi:
                raise ValueError(f"{k}: expected finite value in [{lo},{hi}]")
            if k in ("nx","ny","iterations") and int(v)!=v:
                raise ValueError(f"{k}: integer required")
        else: raise ValueError(f"Unknown parameter {k}")
    if abs(p["frame_dt"]/p["timestep"]-round(p["frame_dt"]/p["timestep"]))>1e-8:
        raise ValueError("frame_dt must be divisible by timestep")
    return p


class MuJoCoBag:
    def __init__(self, params=None, *, mesh_factory=None):
        if mujoco.__version__ != '3.14.0':
            raise RuntimeError('Validated against mujoco==3.14.0')
        self.params = validate(params or {})
        p = self.params
        create_mesh = make_mesh if mesh_factory is None else mesh_factory
        self.rest,self.faces,self.grid,self.regions = create_mesh(p['nx'],p['ny'],
            p['width_mm']*.001,p['height_mm']*.001,p['rest_gap_mm']*.001,.015)
        row = max(1,min(p['ny'],round(p['grip_height']*p['ny'])))
        if p['centered_grip']:
            # Adapt row positions so the finite grip patch has the requested
            # centre, instead of silently snapping it to the coarse mesh.
            half=.5*p['grip_patch_mm']/p['height_mm'];centre=np.clip(p['grip_height'],half,1-half)
            row=max(1,min(p['ny'],int(math.floor(centre*p['ny']))+1))
            old=(self.rest[:,2].astype(float)-.065)/(p['height_mm']*.001)
            lo=(row-1)/p['ny'];hi=row/p['ny']
            anchors=np.array([0.,lo,hi,1.]);mapped=np.array([0.,centre-half,centre+half,1.])
            keep=np.r_[True,np.diff(anchors)>1e-12]
            new=np.interp(np.clip(old,0,1),anchors[keep],mapped[keep])
            scale=np.divide(new,old,out=np.ones_like(old),where=old>1e-6)
            self.rest[:,1]*=scale;self.rest[:,2]=.065+new*p['height_mm']*.001
        self.actual_grip_height_percent=float(np.mean([self.rest[self.grid[0,0,j],2] for j in (row-1,row)])-.065)/(p['height_mm']*.001)*100
        self.beam_sides=tuple(([0,1] if p['rigid_seams'] else [])+
            ([2] if p['bottom_mode']=='sealed' and (p['rigid_seams'] or p['stiff_bottom']) else []))
        self.topology = topology_report(self.rest,self.faces)
        adjacency = {}
        hinges, seam = [],[]
        for tri,region in zip(self.faces,self.regions):
            for k in range(3):
                a,b = sorted([int(tri[k]),int(tri[(k+1)%3])])
                opp = int(tri[(k+2)%3])
                if (a,b) in adjacency:
                    old,r = adjacency[(a,b)]
                    hinges.append([a,b,old,opp]);seam.append(bool(r and region))
                else:adjacency[(a,b)] = (opp,region)
        self.hinges = np.asarray(hinges,np.int32)
        self.seam_hinges = np.asarray(seam,bool)
        self.rest_angles,_ = compute_bag_hinge_batch(self.rest,self.hinges)
        self.memory = np.zeros(len(self.hinges))
        self.plastic = np.zeros(len(self.hinges))
        self.accumulated_plastic = np.zeros(len(self.hinges))
        self.plastic_dissipated = 0.
        self.memory_dissipated = 0.
        edges=self.rest[self.hinges[:,1]]-self.rest[self.hinges[:,0]]
        vertical=edges[:,2]**2/np.sum(edges**2,axis=1)
        # An edge normal to the long direction resists longitudinal curvature.
        anisotropy=1+(p['longitudinal_stiffness']-1)*(1-vertical)
        self.stiffness = p['bend']*np.where(self.seam_hinges,p['seam'],1.)*anisotropy
        self.base_stiffness=self.stiffness.copy()
        self.crease_peak=np.zeros(len(self.hinges));self.crease_damage=np.zeros(len(self.hinges))
        self.crease_dissipated=0.
        mid_height=np.mean(self.rest[self.hinges[:,:2],2],axis=1)
        self.plastic_mask=np.full(len(self.hinges),p['plasticity'],bool)
        if p['bottom_mode']=='plastic':self.plastic_mask|=mid_height<.065+1.5*p['height_mm']*.001/p['ny']+1e-8
        self.surface_edges = np.asarray(list(adjacency),np.int32)
        self.rest_lengths = np.linalg.norm(self.rest[self.surface_edges[:,0]]-self.rest[self.surface_edges[:,1]],axis=1)
        self.grip_row=row
        self.left = np.array([self.grid[(0,0,j)] for j in (row-1,row)])
        self.right = np.array([self.grid[(0,p['nx'],j)] for j in (row-1,row)])
        self.grip_ids = np.concatenate([self.left,self.right])
        self.front = np.array([self.grid[(0,i,p['ny'])] for i in range(p['nx']+1)])
        self.back = np.array([self.grid[(1,i,p['ny'])] for i in range(p['nx']+1)])
        self.xml = self._xml()
        mujoco.set_mjcb_passive(None)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.flex_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_FLEX,'bag')
        adr = int(self.model.flex_vertadr[self.flex_id])
        self.vert_slice = slice(adr,adr+len(self.rest))
        self.body_ids = self.model.flex_vertbodyid[self.vert_slice].copy()
        self.beam_ids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'edge_beam_{side}') for side in self.beam_sides]
        dofs = []
        for bid in self.body_ids:
            start = self.model.body_jntadr[bid]
            joints = np.arange(start,start+3)
            assert self.model.body_jntnum[bid]==3
            assert self.model.body_parentid[bid]==0
            assert np.all(self.model.jnt_type[joints]==mujoco.mjtJoint.mjJNT_SLIDE)
            assert np.allclose(self.model.jnt_axis[joints],np.eye(3))
            dofs.append(self.model.jnt_dofadr[joints])
        self.dofs = np.asarray(dofs)
        self.qpos_ids = self.model.jnt_qposadr[self.model.body_jntadr[self.body_ids,None]+np.arange(3)]
        self.model.dof_damping[self.dofs] = self.model.body_mass[self.body_ids,None]*p['drag']
        self.initial = self._initial_positions()
        if self.initial.shape != self.rest.shape or not np.isfinite(self.initial).all():
            raise ValueError('Initial positions must be finite and match the reference mesh')
        self.data.qpos[self.qpos_ids] = self.initial-self.rest
        self.initial_mocap = self.data.mocap_pos.copy()
        self.right_eq = np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_EQUALITY,f'pin_{i}') for i in range(len(self.left),len(self.grip_ids))])
        self.release_continuous = None
        self.unload_started = None
        self.initial_right_impedance=self.model.eq_solimp[self.right_eq].copy()
        self.released = False
        self.running = False
        self.mode = 'ready'
        self.travel = self.target = 0.
        self.move_start = 0.
        self.start_travel = 0.
        self.experiment = None
        self.frame = 0
        self.cycle=1
        self.repeat_rotation=np.eye(3);self.regrip_continuous=None
        self.step_ms = 0.
        self.callback_calls = 0
        mujoco.set_mjcb_passive(self._passive)
        mujoco.mj_forward(self.model,self.data)
        self.q = self.data.flexvert_xpos[self.vert_slice].copy()

    def _initial_positions(self):
        """Initial coordinates are separate from manufacturing reference data."""
        p = self.params
        initial = self.rest.copy()
        for (side,i,j),vid in self.grid.items():
            if side and (j==0 or i in (0,p['nx'])):continue
            initial[vid,1] += ((1 if side==0 else -1)+.05)*p['imperfection_mm']*.001*math.sin(math.pi*i/p['nx'])*math.sin(math.pi*j/p['ny'])
        initial[self.grip_ids] = self.rest[self.grip_ids]
        return initial

    def _xml(self):
        p = self.params
        root = ET.Element('mujoco',model='mujoco_elastic_bag')
        ET.SubElement(root,'compiler',angle='radian')
        opt = ET.SubElement(root,'option',timestep=str(p['timestep']),integrator='discrete',
            solver='CG',iterations=str(p['iterations']),tolerance=str(p['tolerance']),jacobian='sparse',
            gravity='0 0 -9.81' if p['gravity'] else '0 0 0')
        ET.SubElement(opt,'flag',ipc='enable' if p['ipc'] and p['self_contact'] else 'disable',autoreset='disable')
        world = ET.SubElement(root,'worldbody')
        for i,vid in enumerate(self.grip_ids):
            b = ET.SubElement(world,'body',name=f'grip_{i}',mocap='true',pos=' '.join(map(str,self.rest[vid])))
            ET.SubElement(b,'site',size='.003',rgba='1 .6 .2 1')
        if self.beam_sides:
            width=.015;mass=p['mass_g']*.001*.12/len(self.beam_sides)
            for side in self.beam_sides:
                length=(p['height_mm'] if side<2 else p['width_mm'])*.001
                x=(-1 if side==0 else 1)*p['width_mm']*.0005 if side<2 else 0.
                z=.065+length*.5 if side<2 else .065
                b=ET.SubElement(world,'body',name=f'edge_beam_{side}',pos=f'{x} 0 {z}')
                ET.SubElement(b,'freejoint')
                inertia=mass*(length*length+width*width)/12
                transverse=mass*width*width/6
                ET.SubElement(b,'inertial',pos='0 0 0',mass=str(mass),diaginertia=f'{inertia} {inertia} {transverse}' if side<2 else f'{transverse} {inertia} {inertia}')
        flex = ET.SubElement(world,'flexcomp',name='bag',type='direct',dim='2',
            point=' '.join(f'{x:.12g}' for x in self.rest.ravel()),
            element=' '.join(map(str,self.faces.ravel())),mass=str(p['mass_g']*.001*(.88 if self.beam_sides else 1.)),
            radius=str(p['radius_mm']*.001),rgba='.7 .8 .76 1')
        ET.SubElement(flex,'elasticity',young=str(p['young_pa']),poisson=str(p['poisson']),
            thickness=str(p['thickness_mm']*.001),damping=str(p['rayleigh']),elastic2d='stretch')
        ET.SubElement(flex,'contact',selfcollide='auto' if p['self_contact'] else 'none',
            internal='false',margin=str(p['margin_mm']*.001),condim='1' if p['ipc'] else '3',
            friction='.2 .005 .0001',solref='.004 1',solimp='.95 .99 .0002')
        eq = ET.SubElement(root,'equality')
        for i,vid in enumerate(self.grip_ids):
            ET.SubElement(eq,'connect',name=f'pin_{i}',body1=f'grip_{i}',body2=f'bag_{vid}',
                anchor='0 0 0',solref='.004 1',solimp='.99 .999 .0001')
        if self.beam_sides:
            for side,i in ((0,0),(1,p['nx'])):
                if side not in self.beam_sides:continue
                for j in range(p['ny']+1):
                    vid=self.grid[0,i,j];z=float(self.rest[vid,2])-(.065+p['height_mm']*.0005)
                    ET.SubElement(eq,'connect',name=f'beam_pin_{side}_{j}',body1=f'edge_beam_{side}',body2=f'bag_{vid}',
                        anchor=f'0 0 {z}',solref='.003 1',solimp='.99 .999 .00005')
            if 2 in self.beam_sides:
                for i in range(p['nx']+1):
                    vid=self.grid[0,i,0]
                    ET.SubElement(eq,'connect',name=f'beam_pin_bottom_{i}',body1='edge_beam_2',body2=f'bag_{vid}',
                        anchor=f'{self.rest[vid,0]} 0 0',solref='.003 1',solimp='.99 .999 .00005')
        return ET.tostring(root,encoding='unicode')

    def _passive(self,m,d):
        if m.nv != self.model.nv or m.nflex != self.model.nflex:return
        self.callback_calls += 1
        q = d.flexvert_xpos[self.vert_slice]
        angle,grad = compute_bag_hinge_batch(q,self.hinges)
        delta = wrap(angle-self.rest_angles)
        knee = np.deg2rad(self.params['knee_deg'])
        elastic_moment = moment(delta,self.memory,knee,self.params['post_ratio'],self.params['memory_ratio'])
        if np.any(self.plastic_mask):
            trial_p,_,_=plastic_return(delta,self.plastic,self.accumulated_plastic,knee,self.params['post_ratio'])
            elastic_moment=np.where(self.plastic_mask,wrap(delta-trial_p)+self.params['memory_ratio']*wrap(delta-self.memory),elastic_moment)
        velocity = d.qvel[self.dofs]
        rate = np.einsum('hij,hij->h',grad,velocity[self.hinges])
        multiplier = self.stiffness*elastic_moment+self.params['bend_damping']*rate
        f = -multiplier[:,None,None]*grad
        force = np.zeros((len(q),3))
        for k in range(4):np.add.at(force,self.hinges[:,k],f[:,k])
        d.qfrc_passive[self.dofs] += force

    def _release_right(self):
        q,v = self.data.qpos.copy(),self.data.qvel.copy()
        self.data.eq_active[self.right_eq] = 0
        self.release_continuous = bool(np.array_equal(q,self.data.qpos) and np.array_equal(v,self.data.qvel))
        self.released=True

    def _regrip(self):
        qpos=self.data.qpos.copy();velocity=self.data.qvel.copy()
        q=self.rest+self.data.qpos[self.qpos_ids]
        a=self.rest-self.rest.mean(axis=0);b=q-q.mean(axis=0)
        u,_,vt=np.linalg.svd(a.T@b)
        self.repeat_rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
        self.initial_mocap=q[self.grip_ids].copy()
        self.data.mocap_pos[:]=self.initial_mocap
        self.model.eq_solimp[self.right_eq]=self.initial_right_impedance
        self.data.eq_active[self.right_eq]=1
        self.released=False;self.unload_started=None;self.cycle=2
        self.travel=0.;self.mode='experiment'
        self.regrip_continuous=bool(np.array_equal(qpos,self.data.qpos) and np.array_equal(velocity,self.data.qvel))

    def duration(self):
        p=self.params
        if self.experiment=='crease_repeat':return 18.
        if self.experiment=='single_grip':return 3.
        if self.experiment=='recovery_test':return max(12.,max(3.5,1.5+p['ramp_s'])+6*p['recovery_tau_s']*(1+p['memory_ratio']))
        if self.experiment=='return_test':return max(6.,max(3.,1.+p['ramp_s'])+p['ramp_s']+1.)
        return max(6.,p['ramp_s']+2.)

    def command(self,action):
        if action=='pause':self.running=not self.running;return
        self.running=True
        if action=='release':
            self.unload_started=float(self.data.time)
            self.mode='unloading';self.experiment=None
            return
        if self.released:raise ValueError('Reset the bag before moving both grips')
        self.move_start=float(self.data.time)
        self.start_travel=self.travel
        if action in ('experiment','return_test','recovery_test','single_grip','crease_repeat'):
            self.experiment=action
            self.target=self.params['travel_mm']*.001
            self.mode='experiment'
            if action=='single_grip':
                self.target=0.;self._release_right();self.mode='single_grip'
        elif action in ('squeeze','return'):
            self.experiment=None
            self.target=self.params['travel_mm']*.001 if action=='squeeze' else 0.
            self.mode=action
        else:raise ValueError(action)

    def step(self):
        if not self.running:return
        before=time.perf_counter()
        p=self.params
        for _ in range(round(p['frame_dt']/p['timestep'])):
            t=float(self.data.time)-self.move_start
            local=t
            if self.experiment=='crease_repeat':
                if t>=9. and self.cycle==1:
                    # Regrip the actual recovered surface; preserve every
                    # position, velocity and constitutive history variable.
                    self._regrip()
                local=t-(9. if self.cycle==2 else 0.)
                if local>=3.5 and self.unload_started is None:
                    self.unload_started=float(self.data.time);self.mode='unloading'
            if self.experiment=='recovery_test' and t>=max(3.5,1.5+p['ramp_s']) and self.unload_started is None:
                self.unload_started=float(self.data.time);self.mode='unloading'
            if self.unload_started is not None and not self.released:
                elapsed=float(self.data.time)-self.unload_started
                if p['release_s']==0 or elapsed>=p['release_s']:
                    self._release_right();self.mode='recovering'
                else:
                    s,_=smoothstep(elapsed,0,p['release_s']);weight=math.exp(-14*s)
                    initial=self.initial_right_impedance[:,:2]
                    self.model.eq_solimp[self.right_eq,:2]=initial*weight/(1-initial+initial*weight)
            if self.experiment:
                if local<.5:travel=0.
                elif local<.5+p['ramp_s']:
                    s,_=smoothstep(local,.5,.5+p['ramp_s']);travel=self.target*s
                elif self.experiment=='return_test' and t>max(3.,1.+p['ramp_s']):
                    start=max(3.,1.+p['ramp_s'])
                    s,_=smoothstep(t,start,start+p['ramp_s']);travel=self.target*(1-s)
                else:travel=self.target
            else:
                s,_=smoothstep(t,0.,p['ramp_s']);travel=self.start_travel+(self.target-self.start_travel)*s
            if not self.released:
                self.travel=travel
                self.data.mocap_pos[:]=self.initial_mocap
                repeat=self.experiment=='crease_repeat' and self.cycle==2
                if repeat:
                    axis=self.repeat_rotation[0]
                    self.data.mocap_pos[:len(self.left)]+=travel*axis
                    self.data.mocap_pos[len(self.left):]-=travel*axis
                else:
                    self.data.mocap_pos[:len(self.left),0]+=travel
                    self.data.mocap_pos[len(self.left):,0]-=travel
                # Rotate the finite grip patch during compression, avoiding
                # an orientation clamp that would conflict with the stiff rim.
                reach=max(.025,float(self.rest[self.left[-1],2])-.065)
                angle=math.asin(min(.8,travel/reach)) if p['rigid_seams'] else 0.
                for ids,sign,start in ((self.left,-1,0),(self.right,1,len(self.left))):
                    upper=self.data.mocap_pos[start+len(ids)-1].copy()
                    for k,vid in enumerate(ids[:-1]):
                        length=float(self.rest[ids[-1],2]-self.rest[vid,2])
                        if repeat:
                            v=(self.initial_mocap[start+k]-self.initial_mocap[start+len(ids)-1])@self.repeat_rotation.T
                            c,s=math.cos(angle),math.sin(angle)
                            rotated=np.array([c*v[0]-sign*s*v[2],v[1],sign*s*v[0]+c*v[2]])
                            self.data.mocap_pos[start+k]=upper+rotated@self.repeat_rotation
                        else:self.data.mocap_pos[start+k]=upper+np.array([sign*math.sin(angle)*length,0,-math.cos(angle)*length])
            mujoco.mj_step(self.model,self.data)
            if not np.isfinite(self.data.qpos).all() or np.max(abs(self.data.qpos[self.qpos_ids]))>2:
                raise RuntimeError('MuJoCo trajectory diverged')
            if p['memory_ratio']:
                # State is updated once per accepted mj_step, never inside the
                # passive callback (which MuJoCo may evaluate multiple times).
                q=self.rest+self.data.qpos[self.qpos_ids]
                angle,_=compute_bag_hinge_batch(q,self.hinges,gradient=False)
                delta=wrap(angle-self.rest_angles)
                old=wrap(delta-self.memory)
                self.memory=advance(self.memory,delta,p['timestep'],p['recovery_tau_s'])
                new=wrap(delta-self.memory)
                self.memory_dissipated+=float(np.sum(.5*self.stiffness*p['memory_ratio']*(old**2-new**2)))
            if np.any(self.plastic_mask):
                if not p['memory_ratio']:
                    q=self.rest+self.data.qpos[self.qpos_ids]
                    angle,_=compute_bag_hinge_batch(q,self.hinges,gradient=False);delta=wrap(angle-self.rest_angles)
                next_p,next_alpha,increment=plastic_return(delta,self.plastic,self.accumulated_plastic,np.deg2rad(p['knee_deg']),p['post_ratio'])
                self.plastic=np.where(self.plastic_mask,next_p,self.plastic)
                self.accumulated_plastic=np.where(self.plastic_mask,next_alpha,self.accumulated_plastic)
                increment=np.where(self.plastic_mask,increment,0.)
                self.plastic_dissipated+=float(np.sum(self.stiffness*np.deg2rad(p['knee_deg'])*increment))
            if p['crease_memory']:
                if not p['memory_ratio'] and not np.any(self.plastic_mask):
                    angle,_=compute_bag_hinge_batch(self.rest+self.data.qpos[self.qpos_ids],self.hinges,gradient=False)
                    delta=wrap(angle-self.rest_angles)
                peak,damage=crease_history(self.crease_peak,delta,np.deg2rad(p['crease_onset_deg']),
                    np.deg2rad(p['crease_width_deg']),p['crease_max_softening'])
                density=self._bending_density(delta)
                self.crease_dissipated+=float(np.sum(self.base_stiffness*(damage-self.crease_damage)*density))
                self.crease_peak=peak;self.crease_damage=damage
                self.stiffness=self.base_stiffness*(1-damage)
        mujoco.mj_forward(self.model,self.data)
        self.q=self.data.flexvert_xpos[self.vert_slice].copy()
        self.frame+=1
        self.step_ms=(time.perf_counter()-before)*1000

    def _bending_density(self,delta):
        p=self.params
        density=potential(delta,self.memory,np.deg2rad(p['knee_deg']),p['post_ratio'],p['memory_ratio'])
        if np.any(self.plastic_mask):
            hardening=p['post_ratio']/max(1e-12,1-p['post_ratio'])
            stored=.5*(wrap(delta-self.plastic)**2+hardening*self.accumulated_plastic**2+p['memory_ratio']*wrap(delta-self.memory)**2)
            density=np.where(self.plastic_mask,stored,density)
        return density

    def metrics(self):
        q=self.q
        gap=q[self.front]-q[self.back]
        length=np.linalg.norm(q[self.surface_edges[:,0]]-q[self.surface_edges[:,1]],axis=1)
        strain=abs(length/self.rest_lengths-1)
        actual_left=float(q[self.left[-1],0]-self.rest[self.left[-1],0])*1000
        actual_right=float(self.rest[self.right[-1],0]-q[self.right[-1],0])*1000
        angle,_=compute_bag_hinge_batch(q,self.hinges,gradient=False)
        delta=wrap(angle-self.rest_angles)
        bend_energy=float(np.sum(self.stiffness*potential(delta,self.memory,np.deg2rad(self.params['knee_deg']),self.params['post_ratio'],self.params['memory_ratio'])))
        if np.any(self.plastic_mask):
            hardening=self.params['post_ratio']/max(1e-12,1-self.params['post_ratio'])
            stored=.5*(wrap(delta-self.plastic)**2+hardening*self.accumulated_plastic**2+self.params['memory_ratio']*wrap(delta-self.memory)**2)
            bend_energy=float(np.sum(self.stiffness*np.where(self.plastic_mask,stored,potential(delta,self.memory,np.deg2rad(self.params['knee_deg']),self.params['post_ratio'],self.params['memory_ratio']))))
        # Shape error removes global translation/rotation: support position is
        # allowed to change, whereas permanent deformation must remain visible.
        a=q-q.mean(axis=0);b=self.rest-self.rest.mean(axis=0)
        u,_,vt=np.linalg.svd(a.T@b);rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
        shape_rms=float(np.sqrt(np.mean(np.sum((a@rotation-b)**2,axis=1)))*1000)
        aligned=a@rotation
        mid=self.params['nx']//2
        rim_v=float((.5*(aligned[self.front[0],2]+aligned[self.front[-1],2])-
            .5*(aligned[self.front[mid],2]+aligned[self.back[mid],2]))*1000)
        channel_depth=float(abs(.5*(aligned[self.front[mid],1]+aligned[self.back[mid],1])-
            .5*(aligned[self.front[0],1]+aligned[self.front[-1],1]))*1000)
        grip_front=np.array([self.grid[0,mid,j] for j in (self.grip_row-1,self.grip_row)])
        grip_back=np.array([self.grid[1,mid,j] for j in (self.grip_row-1,self.grip_row)])
        grip_gap=float(np.mean(np.linalg.norm(q[grip_front]-q[grip_back],axis=1))*1000)
        straightness=[]
        for side in (0,self.params['nx']):
            pts=q[[self.grid[0,side,j] for j in range(self.params['ny']+1)]]
            direction=(pts[-1]-pts[0]);direction/=np.linalg.norm(direction)
            straightness.append(float(np.max(np.linalg.norm(np.cross(pts-pts[0],direction),axis=1)))*1000)
        return dict(time=float(self.data.time),frame=self.frame,opening_mm=float(np.linalg.norm(gap,axis=1).max()*1000),
            depth_mm=float(np.ptp(q[:,1])*1000),left_mm=actual_left,right_mm=actual_right,
            shortening_mm=actual_left+actual_right,travel_mm=self.travel*1000,
            strain_p95=float(np.percentile(strain,95)*100),strain_max=float(strain.max()*100),
            contacts=int(self.data.ncon),max_speed_m_s=float(np.linalg.norm(self.data.qvel[self.dofs],axis=1).max()),
            step_ms=self.step_ms,mode=self.mode,running=self.running,released=self.released,
            protocol=self.experiment,duration_s=self.duration(),
            memory_deg=float(np.max(abs(self.memory))*180/np.pi),bending_energy_mJ=bend_energy*1000,
            dissipated_mJ=self.memory_dissipated*1000,plastic_dissipated_mJ=self.plastic_dissipated*1000,
            plastic_deg=float(np.max(abs(self.plastic))*180/np.pi),shape_rms_mm=shape_rms,
            edge_straightness_mm=max(straightness),rim_v_mm=rim_v,gap_at_grip_mm=grip_gap,
            channel_depth_mm=channel_depth,cycle=self.cycle,
            crease_softening_percent=float(self.crease_damage.max()*100),
            crease_hinges=int(np.count_nonzero(self.crease_damage>.05)),crease_dissipated_mJ=self.crease_dissipated*1000)

    def metadata(self):
        return dict(engine=f'MuJoCo {mujoco.__version__}',solver='CG · discrete',
            model='StVK membrane + '+('elastoplastic + SLS bending' if self.params['plasticity'] else 'recoverable nonlinear SLS bending')+(' + crease stiffness history' if self.params['crease_memory'] else ''),
            params=self.params,actual_grip_height_percent=self.actual_grip_height_percent,
            rest=self.rest.ravel().tolist(),faces=self.faces.ravel().tolist(),
            regions=self.regions.tolist(),left=self.left.tolist(),right=self.right.tolist(),
            front=self.front.tolist(),back=self.back.tolist(),hinges=self.hinges.tolist(),plastic_mask=self.plastic_mask.tolist(),vertices=len(self.rest),triangles=len(self.faces),
            topology=self.topology,calibrated=False,ipc=self.params['ipc'] and self.params['self_contact'],
            contact_note='IPC contacts frictionless' if self.params['ipc'] else 'soft frictional contacts')

    def snapshot(self):
        return dict(**self.metrics(),positions=self.q.ravel().tolist(),hinge_memory_deg=np.rad2deg(self.memory).round(3).tolist(),hinge_plastic_deg=np.rad2deg(self.plastic).round(3).tolist(),hinge_crease_percent=(100*self.crease_damage).round(3).tolist())

    def close(self):mujoco.set_mjcb_passive(None)

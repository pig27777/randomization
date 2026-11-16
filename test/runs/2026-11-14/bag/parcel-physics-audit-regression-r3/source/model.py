"""MJWarp topology batches with GPU sampling and native rigid-body dynamics."""
from dataclasses import fields
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from . import kernels as K


@wp.kernel
def fixture_drive(dt: float, step: wp.array(dtype=int), params: wp.array(dtype=K.Parameters),
                  qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                  force: wp.array2d(dtype=float), addresses: wp.array2d(dtype=int),
                  action: int, eccentric: float):
    e = wp.tid()
    t = float(step[0])*dt
    p = params[e]
    z = p.size[2]+.06
    gap = .035
    if action > 0:
        # A material test platen, independent of the robot grasp protocol.
        z = p.size[2]+.06-wp.clamp((t-.2)/.4,0.0,1.0)*.068
        if action >= 2:
            gap = .035-wp.clamp((t-.4)/.4,0.0,1.0)*.034
            z += wp.clamp(t-1.0,0.0,1.0)*.10
        else:
            z += wp.clamp((t-1.0)/.5,0.0,1.0)*.068
    targets = wp.vec3(eccentric,0.0,z)
    for i in range(3):
        q = addresses[i,0]
        v = addresses[i,1]
        force[e,v] += 1500.0*(targets[i]-qpos[e,q])-60.0*qvel[e,v]
    for i in range(2):
        q = addresses[3+i,0]
        v = addresses[3+i,1]
        force[e,v] += 800.0*(gap-qpos[e,q])-8.0*qvel[e,v]


@wp.kernel
def initialize_fixture(params: wp.array(dtype=K.Parameters), addresses: wp.array2d(dtype=int),
                       qpos0: wp.array2d(dtype=float), qpos: wp.array2d(dtype=float)):
    e = wp.tid()
    qpos[e,addresses[2,0]] = params[e].size[2]+.06
    qpos[e,addresses[3,0]] = .035
    qpos[e,addresses[4,0]] = .035


@wp.kernel
def measure_gap(qpos: wp.array2d(dtype=float), addresses: wp.array2d(dtype=int), gap: wp.array(dtype=float)):
    e = wp.tid()
    gap[e] = qpos[e,addresses[3,0]]+qpos[e,addresses[4,0]]


@wp.kernel
def tick(step: wp.array(dtype=int)):
    step[0] += 1


@wp.kernel
def film_proxy(states: wp.array(dtype=K.State), params: wp.array(dtype=K.Parameters), qpos: wp.array2d(dtype=float),
               qvel: wp.array2d(dtype=float), addresses: wp.array2d(dtype=int),
               normal: wp.array2d(dtype=float), force: wp.array2d(dtype=float)):
    e = wp.tid()
    s = states[e]
    feasible_fold = s.mode == K.IDLE and s.gathered > params[e].threshold
    if s.mode == K.ENGAGED or s.mode == K.SLIP or feasible_fold:
        # The reduced fold spans twice its gathered material length plus 4 mm.
        # Each pad receives its spring force; lateral reactions cancel at the film.
        for i in range(2):
            q = addresses[3+i,0]
            v = addresses[3+i,1]
            length = s.captured
            if s.mode == K.IDLE:
                length = s.gathered
            half_width = .5*wp.min(.004+2.0*length,.024)
            penetration = half_width-qpos[e,q]
            load = 0.0
            if penetration > 0.0:
                load = wp.max(10000.0*penetration-2.0*qvel[e,v],0.0)
            force[e,v] += load
            normal[e,i] = load


def xml(rigid=False, dt=.0005):
    n = 1 if rigid else 3
    bodies = ''
    for i in range(n):
        joint = '<freejoint name="root"/>' if i == 0 else ''.join(
            f'<joint name="rot{i}_{axis}" axis="{xyz}" pos="-.04 0 0" stiffness="1" damping=".1"/>'
            for axis,xyz in enumerate(('1 0 0','0 1 0','0 0 1')))
        compression = '' if rigid else f'<joint name="compress{i}" type="slide" axis="0 0 1" range="-.03 0" limited="true" stiffness="1000" damping="1"/>'
        bodies += f'''<body name="segment{i}" pos="{0 if i == 0 else .08} 0 { .03 if i == 0 else 0}">{joint}
        <inertial pos="0 0 0" mass=".2" diaginertia=".001 .001 .001"/>
        <geom name="core{i}" size=".039 .07 .024" rgba=".3 .5 .8 1"/>
        <body name="cap{i}" pos="0 0 .024">{compression}
        <inertial pos="0 0 0" mass=".02" diaginertia=".0001 .0001 .0001"/>
        <geom name="capgeom{i}" size=".039 .07 .006" rgba=".7 .8 .9 1"/></body>'''
    bodies += '</body>'*n
    return f'''<mujoco><compiler fusestatic="false"/><option timestep="{dt}" integrator="implicitfast" cone="elliptic" iterations="30"/>
    <default><geom type="box" friction=".5 .005 .0001" solref=".004 1"/><joint armature=".00001"/></default>
    <worldbody><geom name="table" size="1 1 .025" pos="0 0 -.025"/>{bodies}
    <body name="cart"><inertial pos="0 0 0" mass=".2" diaginertia=".001 .001 .001"/>
    <joint name="cart_x" type="slide" axis="1 0 0"/><joint name="cart_y" type="slide" axis="0 1 0"/>
    <joint name="cart_z" type="slide" axis="0 0 1"/>
    <body name="left"><joint name="left" type="slide" axis="0 1 0"/>
    <geom name="left" size=".02 .004 .008" pos="0 .004 0" mass=".02" rgba=".8 .3 .2 1"/></body>
    <body name="right"><joint name="right" type="slide" axis="0 -1 0"/>
    <geom name="right" size=".02 .004 .008" pos="0 -.004 0" mass=".02" rgba=".8 .3 .2 1"/></body></body>
    </worldbody><contact>{''.join(f'<exclude body1="segment{i}" body2="segment{i+1}"/>' for i in range(n-1))}</contact></mujoco>'''


class Batch:
    """One fixed topology, independently randomized native model per world.

    `level`: payload=0, ideal bag test connections=1, engagement+slip=2.
    `action`: static=0, compression fixture=1, closing/lifting fixture=2.
    """
    def __init__(self, nworld=1, rigid=False, seed=7, prototype=-1, dt=.0005,
                 level=2, action=0, eccentric=0.0, device='cuda:0', config=None):
        if nworld < 1 or dt <= 0 or level not in (0,1,2) or action not in (0,1,2):
            raise ValueError('Invalid batch size, timestep, level or action')
        if prototype not in range(-1,6) or (prototype >= 0 and (prototype < 2) != rigid):
            raise ValueError('Prototype and rigid topology must agree')
        self.nworld,self.rigid,self.dt,self.level = nworld,rigid,dt,level
        self.seed,self.prototype,self.action,self.eccentric = seed,prototype,action,eccentric
        self.device = device
        self.cpu = mujoco.MjModel.from_xml_string(xml(rigid,dt))
        m = self.cpu
        # Allocate all supported batched fields, including set_const outputs.
        sizes = {f.name:nworld for f in fields(mjw.Model)
                 if getattr(f.type,'shape',())[0:1] == ('*',)}
        with wp.ScopedDevice(device):
            self.model = mjw.put_model(m,batch_sizes=sizes)
            self.data = mjw.make_data(m,nworld=nworld,nconmax=32,njmax=128)
            self.params = wp.empty(nworld,dtype=K.Parameters)
            self.states = wp.empty(nworld,dtype=K.State)
            self.normal = wp.zeros((nworld,2),dtype=float)
            self.gap = wp.zeros(nworld,dtype=float)
            self.clock = wp.zeros(1,dtype=int)
            ids = np.zeros((3,10),dtype=np.int32)
            for i in range(1 if rigid else 3):
                ids[i,:4] = [m.body(f'segment{i}').id,m.geom(f'core{i}').id,
                             m.body(f'cap{i}').id,m.geom(f'capgeom{i}').id]
                if i > 0:
                    j = m.joint(f'rot{i}_0').id
                    ids[i,4:6] = [j,m.jnt_dofadr[j]]
                if not rigid:
                    j = m.joint(f'compress{i}').id
                    ids[i,6:8] = [j,m.jnt_dofadr[j]]
                    ids[i,9] = m.jnt_qposadr[j]
            ids[0,8] = m.jnt_qposadr[m.joint('root').id]
            self.ids = wp.array(ids,dtype=int)
            addr = [[m.jnt_qposadr[m.joint(name).id],m.jnt_dofadr[m.joint(name).id]]
                    for name in ('cart_x','cart_y','cart_z','left','right')]
            self.addresses = wp.array(addr,dtype=int)
            self.cart = m.body('cart').id
            cfg = json.loads(Path(config or Path(__file__).with_name('config.json')).read_text())
            names = ('length','width','height','mass','loose_slack','rope_stiffness',
                     'static_friction','table_friction','film_strength','segment_weight','com_fraction',
                     'bend_sag','soft_sag','twist_ratio','damping_ratio','compression_depth',
                     'soft_compression_depth','slack_y_ratio')
            bounds = np.array([cfg[k] for k in names],dtype=np.float32)
            if bounds.shape != (18,2) or not np.isfinite(bounds).all() or np.any(np.delete(bounds[:,0],10) <= 0) or np.any(bounds[:,1]<bounds[:,0]):
                raise ValueError('Ranges must be finite, positive and ordered')
            self.bounds = wp.array(bounds,dtype=wp.vec2)
            self.reset()

    def reset(self, episode=0):
        with wp.ScopedDevice(self.device):
            mjw.reset_data(self.model,self.data)
            self.clock.zero_()
            self.normal.zero_()
            wp.launch(K.sample,self.nworld,[self.seed,episode,int(self.rigid),self.prototype,self.bounds,self.params,self.states])
            wp.launch(K.configure,self.nworld,[int(self.rigid),self.cpu.geom("table").id,self.ids,self.params,self.model.body_pos,
                self.model.body_ipos,self.model.body_mass,self.model.body_inertia,self.model.geom_size,
                self.model.geom_friction,self.model.jnt_pos,self.model.jnt_stiffness,self.model.dof_damping,
                self.model.qpos0,self.data.qpos])
            wp.launch(initialize_fixture,self.nworld,[self.params,self.addresses,self.model.qpos0,self.data.qpos])
            mjw.set_const(self.model,self.data)
            mjw.forward(self.model,self.data)

    def step(self, close=0, release=0, bind=0, gather_speed=.04, region=0):
        if region not in range(4) or close not in (0,1) or release not in (0,1) or not np.isfinite(gather_speed):
            raise ValueError("Invalid capture region, event or gathering speed")
        with wp.ScopedDevice(self.device):
            d,m = self.data,self.model
            d.xfrc_applied.zero_()
            d.qfrc_applied.zero_()
            self.normal.zero_()
            c = d.contact
            wp.launch(K.contacts,c.dist.shape[0],[int(self.cpu.opt.cone),d.nacon,c.worldid,c.geom,c.frame,
                c.friction,c.dim,c.efc_address,c.adhesion,d.efc.force,d.njmax,
                self.cpu.geom('left').id,self.cpu.geom('right').id,self.normal])
            # Normal feedback uses the last completed constraint solve. Refresh
            # current transforms before evaluating the current elastic forces.
            mjw.step1(m,d)
            if self.action:
                wp.launch(fixture_drive,self.nworld,[self.dt,self.clock,self.params,d.qpos,d.qvel,
                          d.qfrc_applied,self.addresses,self.action,self.eccentric])
            else:
                wp.launch(fixture_drive,self.nworld,[self.dt,self.clock,self.params,d.qpos,d.qvel,
                          d.qfrc_applied,self.addresses,0,0.0])
            wp.launch(film_proxy,self.nworld,[self.states,self.params,d.qpos,d.qvel,self.addresses,self.normal,d.qfrc_applied])
            wp.launch(measure_gap,self.nworld,[d.qpos,self.addresses,self.gap])
            wp.launch(K.dynamics,self.nworld,[self.dt,int(self.rigid),self.level,self.cart,region,
                close,release,bind,gather_speed,self.params,self.states,self.ids,self.normal,self.gap,
                d.xpos,d.xmat,d.xipos,d.qpos,d.qfrc_applied,d.xfrc_applied])
            mjw.step2(m,d)
            wp.launch(tick,1,[self.clock])

    def graph(self, **kwargs):
        """Warm up before capture. Call only after the desired phase is set."""
        self.step(**kwargs)
        wp.synchronize_device(self.device)
        with wp.ScopedDevice(self.device), wp.ScopedCapture() as cap:
            self.step(**kwargs)
        return cap.graph

"""GPU-resident native flex/contact stepping with analytic hinge and cap forces.

One Batch owns one fixed reference/material group. reset() changes supported
per-world parameters and state without recompiling MJCF or changing topology.
"""
from dataclasses import fields
import numpy as np
import mujoco_warp as mjw
import warp as wp

from .physical_parcel import Config, Parcel


@wp.kernel
def bending(points: wp.array2d(dtype=wp.vec3), velocity: wp.array2d(dtype=float),
            dofs: wp.array(dtype=int), hinges: wp.array(dtype=wp.vec4i),
            rest: wp.array(dtype=float), stiffness: wp.array(dtype=float), damping: wp.array(dtype=float),
            target: wp.array2d(dtype=float)):
    world, hinge = wp.tid()
    ids = hinges[hinge]
    a = points[world, ids[0]]
    b = points[world, ids[1]]
    c = points[world, ids[2]]
    d = points[world, ids[3]]
    edge = b-a
    l2 = wp.max(wp.dot(edge, edge), 1.e-24)
    length = wp.sqrt(l2)
    n1 = wp.cross(edge, c-a)
    n2 = wp.cross(d-a, edge)
    norm1 = wp.max(wp.dot(n1, n1), 1.e-24)
    norm2 = wp.max(wp.dot(n2, n2), 1.e-24)
    unit1 = n1/wp.sqrt(norm1)
    unit2 = n2/wp.sqrt(norm2)
    angle = wp.atan2(wp.dot(edge/length, wp.cross(unit1, unit2)), wp.dot(unit1, unit2))
    delta = wp.atan2(wp.sin(angle-rest[hinge]), wp.cos(angle-rest[hinge]))
    gc = -n1*length/norm1
    gd = -n2*length/norm2
    ga = gc*(-wp.dot(b-c,edge)/l2)+gd*(-wp.dot(b-d,edge)/l2)
    gb = gc*(-wp.dot(c-a,edge)/l2)+gd*(-wp.dot(d-a,edge)/l2)
    rate = float(0.0)
    for corner in range(4):
        grad = ga
        if corner == 1:
            grad = gb
        elif corner == 2:
            grad = gc
        elif corner == 3:
            grad = gd
        for axis in range(3):
            rate += grad[axis]*velocity[world,dofs[ids[corner]]+axis]
    moment = stiffness[hinge]*delta+damping[hinge]*rate
    for corner in range(4):
        grad = ga
        if corner == 1:
            grad = gb
        elif corner == 2:
            grad = gc
        elif corner == 3:
            grad = gd
        for axis in range(3):
            wp.atomic_add(target,world,dofs[ids[corner]]+axis,-moment*grad[axis])


@wp.kernel
def cap_force(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
              qids: wp.array(dtype=int), vids: wp.array(dtype=int),
              parameters: wp.array2d(dtype=float), k1: float, k2: float, damping: float,
              target: wp.array2d(dtype=float)):
    world, cap = wp.tid()
    delta = wp.max(qpos[world,qids[cap]],0.0)
    scale = parameters[world,5]
    force = wp.max(0.0, scale*(k1*delta+k2*delta*delta)
                   +wp.sqrt(scale)*damping*qvel[world,vids[cap]])
    target[world,vids[cap]] -= force


@wp.kernel
def reset_state(initial: wp.array(dtype=float), qpos: wp.array2d(dtype=float)):
    world,j=wp.tid()
    qpos[world,j]=initial[j]


@wp.kernel
def configure(seed: int, episode: int, randomize: bool,
              payload_bodies: wp.array(dtype=int), payload_geoms: wp.array(dtype=int),
              bend_joints: wp.array(dtype=int), bend_dofs: wp.array(dtype=int), table: int,
              base_mass: wp.array(dtype=float), base_inertia: wp.array(dtype=wp.vec3),
              base_k: wp.array(dtype=float), base_damping: wp.array(dtype=float),
              mass: wp.array2d(dtype=float), inertia: wp.array2d(dtype=wp.vec3),
              stiffness: wp.array2d(dtype=float), damping: wp.array2d(dtype=float),
              friction: wp.array2d(dtype=wp.vec3), base_mu: float, base_table_mu: float,
              parameters: wp.array2d(dtype=float)):
    world=wp.tid()
    rng=wp.rand_init(seed,world+episode*parameters.shape[0])
    scale=float(1.0)
    payload_mu=base_mu
    table_mu=base_table_mu
    kscale=float(1.0)
    dscale=float(1.0)
    cscale=float(1.0)
    if randomize:
        scale=wp.randf(rng,.8,1.2)
        payload_mu=wp.randf(rng,.25,.65)
        table_mu=wp.randf(rng,.3,.7)
        kscale=wp.exp(wp.randf(rng,-1.3862944,1.3862944))
        dscale=wp.exp(wp.randf(rng,-.6931472,.6931472))
        cscale=wp.randf(rng,.5,2.0)
    parameters[world,0]=scale
    parameters[world,1]=payload_mu
    parameters[world,2]=table_mu
    parameters[world,3]=kscale
    parameters[world,4]=dscale
    parameters[world,5]=cscale
    for j in range(payload_bodies.shape[0]):
        body=payload_bodies[j]
        mass[world,body]=base_mass[body]*scale
        inertia[world,body]=base_inertia[body]*scale
    for j in range(payload_geoms.shape[0]):
        friction[world,payload_geoms[j]]=wp.vec3(payload_mu,.005,.0001)
    friction[world,table]=wp.vec3(table_mu,.005,.0001)
    for j in range(bend_joints.shape[0]):
        joint=bend_joints[j]
        dof=bend_dofs[j]
        stiffness[world,joint]=base_k[joint]*kscale
        damping[world,dof]=base_damping[dof]*dscale


@wp.kernel
def state_health(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                 overflow: wp.array(dtype=int), iterations: wp.array(dtype=int),
                 bits: wp.array(dtype=int), invalid: wp.array(dtype=int), peak_iterations: wp.array(dtype=int)):
    world=wp.tid()
    bits[world]=bits[world]|overflow[world]
    peak_iterations[world]=wp.max(peak_iterations[world],iterations[world])
    for j in range(qpos.shape[1]):
        if not wp.isfinite(qpos[world,j]):
            invalid[world]=1
    for j in range(qvel.shape[1]):
        if not wp.isfinite(qvel[world,j]):
            invalid[world]=1


@wp.kernel
def contact_health(nacon: wp.array(dtype=int), worldid: wp.array(dtype=int), distances: wp.array(dtype=float),
                   counts: wp.array(dtype=int), peak_penetration: wp.array(dtype=float)):
    contact=wp.tid()
    if contact<wp.min(nacon[0],worldid.shape[0]):
        world=worldid[contact]
        wp.atomic_add(counts,world,1)
        wp.atomic_max(peak_penetration,world,wp.max(-distances[contact],0.0))


class Batch:
    def __init__(self, nworld, cfg=Config(), *, seed=7, randomize=True, device="cuda:0",
                 contact_capacity=256, constraint_capacity=1024, parcel=None, model=None, data=None):
        if not isinstance(nworld,int) or nworld<1:
            raise ValueError("nworld must be a positive integer")
        self.nworld=nworld
        self.cfg=cfg
        self.seed=seed
        self.randomize=randomize
        self.device=device
        self.parcel=Parcel(cfg) if parcel is None else parcel
        p=self.parcel
        cpu_m = p.m if model is None else model
        cpu_d = p.d if data is None else data
        self.cpu_model, self.cpu_data = cpu_m, cpu_d
        self.episode=0
        with wp.ScopedDevice(device):
            # Mass/inertia changes also require per-world set_const outputs.
            sizes={field.name:nworld for field in fields(mjw.Model)
                   if getattr(field.type,'shape',())[0:1]==('*',)}
            self.m=mjw.put_model(cpu_m,batch_sizes=sizes)
            self.d=mjw.put_data(cpu_m,cpu_d,nworld=nworld,nconmax=contact_capacity,
                                naconmax=contact_capacity*nworld,njmax=constraint_capacity)
            self.hinges=wp.array(p.hinges,dtype=wp.vec4i)
            self.rest=wp.array(p.rest_angles,dtype=float)
            self.k=wp.array(p.hinge_stiffness,dtype=float)
            self.c=wp.array(p.hinge_damping,dtype=float)
            self.dofs=wp.array(p.film_dofs[:,0],dtype=int)
            self.cq=wp.array(cpu_m.jnt_qposadr[p.compression_joints],dtype=int)
            self.cv=wp.array(cpu_m.jnt_dofadr[p.compression_joints],dtype=int)
            self.initial=wp.array(cpu_d.qpos,dtype=float)
            self.initial_velocity=wp.array(cpu_d.qvel,dtype=float)
            self.initial_control=wp.array(cpu_d.ctrl,dtype=float)
            self.payload_bodies=wp.array(p.payload_bodies,dtype=int)
            self.payload_geoms=wp.array([g for g in range(cpu_m.ngeom) if cpu_m.geom_bodyid[g] in p.payload_bodies],dtype=int)
            joints=np.array([j for j in range(cpu_m.njnt) if cpu_m.joint(j).name.startswith('bend_')],dtype=int)
            self.bend_joints=wp.array(joints,dtype=int)
            self.bend_dofs=wp.array(cpu_m.jnt_dofadr[joints],dtype=int)
            self.base_mass=wp.array(cpu_m.body_mass,dtype=float)
            self.base_inertia=wp.array(cpu_m.body_inertia,dtype=wp.vec3)
            self.base_k=wp.array(cpu_m.jnt_stiffness,dtype=float)
            self.base_damping=wp.array(cpu_m.dof_damping,dtype=float)
            self.parameters=wp.zeros((nworld,6),dtype=float)
            self.bits=wp.zeros(nworld,dtype=int)
            self.invalid=wp.zeros(nworld,dtype=int)
            self.peak_iterations=wp.zeros(nworld,dtype=int)
            self.contact_counts=wp.zeros(nworld,dtype=int)
            self.peak_penetration=wp.zeros(nworld,dtype=float)
            self.reset()

    def reset(self, *, initial_qpos=None, initial_qvel=None, initial_ctrl=None):
        with wp.ScopedDevice(self.device):
            if initial_qpos is not None:
                value=np.asarray(initial_qpos)
                if value.shape!=(self.cpu_model.nq,) or not np.isfinite(value).all():
                    raise ValueError("initial_qpos has an invalid shape or nonfinite values")
                self.initial.assign(value)
            wp.launch(reset_state,self.d.qpos.shape,[self.initial,self.d.qpos])
            for value, target, count in ((initial_qvel,self.initial_velocity,self.cpu_model.nv),
                                         (initial_ctrl,self.initial_control,self.cpu_model.nu)):
                if value is not None:
                    value=np.asarray(value)
                    if value.shape!=(count,) or not np.isfinite(value).all():
                        raise ValueError("reset state has invalid dimensions or nonfinite values")
                    target.assign(value)
            wp.launch(reset_state,self.d.qvel.shape,[self.initial_velocity,self.d.qvel])
            if self.cpu_model.nu:
                wp.launch(reset_state,self.d.ctrl.shape,[self.initial_control,self.d.ctrl])
            self.d.qacc_warmstart.zero_()
            self.d.time.zero_()
            wp.launch(configure,self.nworld,[self.seed,self.episode,self.randomize,
                      self.payload_bodies,self.payload_geoms,self.bend_joints,self.bend_dofs,
                      self.cpu_model.geom('table').id,self.base_mass,self.base_inertia,self.base_k,self.base_damping,
                      self.m.body_mass,self.m.body_inertia,self.m.jnt_stiffness,self.m.dof_damping,
                      self.m.geom_friction,float(self.cpu_model.geom_friction[int(self.payload_geoms.numpy()[0]),0]),
                      float(self.cpu_model.geom_friction[self.cpu_model.geom('table').id,0]),self.parameters])
            mjw.set_const(self.m,self.d)
            self.bits.zero_()
            self.invalid.zero_()
            self.peak_iterations.zero_()
            self.contact_counts.zero_()
            self.peak_penetration.zero_()
            self.episode+=1

    def internal_forces(self):
        wp.launch(bending,(self.nworld,len(self.parcel.hinges)),
                  [self.d.flexvert_xpos,self.d.qvel,self.dofs,self.hinges,self.rest,self.k,self.c,self.d.qfrc_passive])
        if len(self.parcel.compression_joints):
            wp.launch(cap_force,(self.nworld,len(self.parcel.compression_joints)),
                      [self.d.qpos,self.d.qvel,self.cq,self.cv,self.parameters,self.cfg.compression_k1,
                       self.cfg.compression_k2,self.cfg.compression_damping,self.d.qfrc_passive])

    def step(self):
        mjw.step1(self.m,self.d)
        self.internal_forces()
        mjw.step2(self.m,self.d)
        wp.launch(state_health,self.nworld,[self.d.qpos,self.d.qvel,self.d.overflow,self.d.solver_niter,
                  self.bits,self.invalid,self.peak_iterations])
        self.contact_counts.zero_()
        wp.launch(contact_health,self.d.naconmax,[self.d.nacon,self.d.contact.worldid,self.d.contact.dist,
                  self.contact_counts,self.peak_penetration])

    def graph(self, steps=1):
        if not isinstance(steps,int) or steps<1:
            raise ValueError("steps must be a positive integer")
        with wp.ScopedDevice(self.device):
            self.step()
            with wp.ScopedCapture() as capture:
                for _ in range(steps):
                    self.step()
        return capture.graph

    def diagnostics(self):
        return dict(overflow_bits=self.bits.numpy().tolist(),nonfinite=self.invalid.numpy().tolist(),
                    native_nacon=int(self.d.nacon.numpy()[0]),
                    peak_solver_iterations=self.peak_iterations.numpy().tolist(),
                    last_contacts_per_world=self.contact_counts.numpy().tolist(),
                    peak_penetration_m=self.peak_penetration.numpy().tolist(),
                    parameters=self.parameters.numpy().tolist(),
                    parameter_columns=['payload_mass_scale','payload_mu','table_mu','joint_k_scale','joint_d_scale','compression_k_scale'])

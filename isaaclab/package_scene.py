"""L5/L6 low-level Isaac Lab Newton MJWarp/VBD proxy scene; shared libraries stay read-only."""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupledProxy
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager

from package_assets import BAG, INPUT_ROOT, area, original_mesh, seam_pairs, wrapped

wp.config.enable_backward=False


@wp.kernel
def drive_pads(targets:wp.array[wp.vec3],velocity:wp.array[wp.vec3],origins:wp.array[wp.vec3],
               starts:wp.array[int],velocity_starts:wp.array[int],
               position_control:wp.array[float],velocity_control:wp.array[float]):
    i=wp.tid();j=starts[i];k=velocity_starts[i]
    for axis in range(3):
        position_control[j+axis]=targets[i][axis]-origins[i][axis]
        velocity_control[k+axis]=velocity[i][axis]


@wp.kernel
def advance_targets(targets:wp.array[wp.vec3],velocities:wp.array[wp.vec3],dt:float):
    i=wp.tid();targets[i]=targets[i]+velocities[i]*dt


@wp.kernel
def native_contact_reading(count:wp.array[int],geoms:wp.array[wp.vec2i],forces:wp.array[wp.spatial_vector],
                          floor:wp.array[int],content:wp.array[int],pads:wp.array[int],
                          pad_force:wp.array[wp.vec3],out:wp.array[float]):
    i=wp.tid()
    if i>=count[0]:return
    a=geoms[i][0];b=geoms[i][1]
    f=wp.vec3(forces[i][0],forces[i][1],forces[i][2])
    if pads[a]>=0 and content[b]!=0:wp.atomic_add(pad_force,pads[a],-f)
    elif pads[b]>=0 and content[a]!=0:wp.atomic_add(pad_force,pads[b],f)
    if floor[a]!=0 and content[b]!=0:wp.atomic_add(out,0,wp.max(forces[i][2],0.))
    elif floor[b]!=0 and content[a]!=0:wp.atomic_add(out,0,wp.max(-forces[i][2],0.))


@wp.kernel
def cloth_contact_reading(raw:wp.array[wp.spatial_vector],pads:wp.array[int],known:wp.array[wp.vec3],
                         pad_force:wp.array[wp.vec3],body:int,out:wp.array[float]):
    i=wp.tid();value=raw[pads[i]]
    pad_force[i]=pad_force[i]+wp.vec3(value[0],value[1],value[2])-known[i]
    if i==0:out[0]=out[0]+wp.max(-raw[body][2],0.)


@wp.kernel
def limit_gap(targets:wp.array[wp.vec3],maximum:wp.array[float]):
    a=targets[0];b=targets[1]
    gap=wp.clamp(b[1]-a[1]-.004,0.,maximum[0])
    middle=(a[1]+b[1])/2.
    targets[0]=wp.vec3(a[0],middle-(gap+.004)/2.,a[2])
    targets[1]=wp.vec3(b[0],middle+(gap+.004)/2.,b[2])


@wp.kernel
def particle_drag(velocity:wp.array[wp.vec3],mass:wp.array[float],force:wp.array[wp.vec3],rate:float):
    i=wp.tid();force[i]=force[i]-rate*mass[i]*velocity[i]


@wp.kernel
def content_failure(pose:wp.array[wp.transform],velocity:wp.array[wp.spatial_vector],
                    particle:wp.array[wp.vec3],particle_velocity:wp.array[wp.vec3],
                    a:wp.array[int],b:wp.array[int],arm:wp.array[wp.vec3],
                    compression:wp.array[float],bending:wp.array[wp.vec3],out:wp.array[float]):
    i=wp.tid()
    if i<pose.shape[0]:
        for c in range(7):
            if not wp.isfinite(pose[i][c]):wp.atomic_max(out,3,1.)
        for c in range(6):
            if not wp.isfinite(velocity[i][c]):wp.atomic_max(out,3,1.)
    if i<particle.shape[0]:
        for c in range(3):
            if not wp.isfinite(particle[i][c]) or not wp.isfinite(particle_velocity[i][c]):wp.atomic_max(out,3,1.)
    if i<a.shape[0]:
        qa=wp.transform_get_rotation(pose[a[i]]);qb=wp.transform_get_rotation(pose[b[i]])
        delta=wp.transform_get_translation(pose[b[i]])-wp.transform_get_translation(pose[a[i]])
        delta=delta-wp.quat_rotate(qb,arm[i])-wp.quat_rotate(qa,arm[i])
        u=wp.quat_rotate_inv(qa,delta);rest=2.*wp.length(arm[i])
        strain=-wp.dot(u,arm[i]/wp.length(arm[i]))/rest
        wp.atomic_max(out,0,strain);wp.atomic_max(out,1,strain/compression[i])
        q=wp.quat_inverse(qa)*qb
        if q[3]<0.:q=-q
        xyz=wp.vec3(q[0],q[1],q[2]);s=wp.length(xyz);phi=2.*xyz
        if s>1.e-8:phi=2.*wp.atan2(s,q[3])*xyz/s
        for c in range(3):wp.atomic_max(out,2,wp.abs(phi[c])/bending[i][c])


def color_seams(builder):
    """Newton's default particle graph omits springs; include their dependencies."""
    labels=np.empty(builder.particle_count,dtype=int)
    for c,group in enumerate(builder.particle_color_groups):labels[group]=c
    neighbors=[set() for _ in labels]
    springs=np.asarray(builder.spring_indices,dtype=int).reshape(-1,2)
    for feature in [*builder.tri_indices,*builder.edge_indices,*springs]:
        for a,b in combinations([int(i) for i in feature if i>=0],2):
            neighbors[a].add(b);neighbors[b].add(a)
    for a,b in springs:
        if labels[a]==labels[b]:
            used={labels[i] for i in neighbors[b]}
            labels[b]=next(c for c in range(len(used)+1) if c not in used)
    if any(labels[a]==labels[b] for a,adj in enumerate(neighbors) for b in adj):
        raise RuntimeError('invalid cloth/seam dependency coloring')
    builder.set_coloring([np.flatnonzero(labels==c) for c in np.unique(labels)])


def add_fingertip(builder,body,cfg,attributes):
    """Match the protocol's rounded 20 x 4 x 40 mm pad in both backends."""
    builder.add_shape_box(body,hx=.008,hy=.002,hz=.018,cfg=cfg,custom_attributes=attributes)
    for x in (-.008,.008):
        builder.add_shape_capsule(body,xform=wp.transform([x,0,0],wp.quat_identity()),
            radius=.002,half_height=.018,cfg=cfg,custom_attributes=attributes)
    for z in (-.018,.018):
        builder.add_shape_capsule(body,xform=wp.transform([0,0,z],wp.quat_from_axis_angle(wp.vec3(0,1,0),np.pi/2)),
            radius=.002,half_height=.008,cfg=cfg,custom_attributes=attributes)


class PackageScene:
    def __init__(self,scale=1.,mass_scale=1.,self_contact=True,cloth=True,mesh=None,dt=None,device='cuda:0',category='hard'):
        wp.set_device(device)
        fine=original_mesh()
        mesh=fine if mesh is None else mesh
        self.category=category
        self.self_contact=self_contact
        self.size=np.array([.08,.045,.02] if category=='hard' else [.35,.25,.12])
        self.mass=.2 if category=='hard' else .5
        rest,initial,center,self.wrap_report=wrapped(mesh,self.size,scale)
        self.rest,self.initial,self.faces=rest,initial,mesh['faces']
        self.seams=seam_pairs(mesh)
        self.center=center;self.cloth=cloth
        sys.path.insert(0,str(BAG))
        from bag_model import parameters
        profile=json.loads((BAG/'profiles/kraft.json').read_text())
        p=parameters(profile['params']);self.params=p
        density=fine['masses_kg'].sum()/area(fine['vertices'],fine['faces']).sum()
        va=np.zeros(len(rest));np.add.at(va,self.faces.ravel(),np.repeat(area(rest,self.faces)/3,3))
        self.masses=va*(fine['masses_kg'].sum()*scale**2*mass_scale/va.sum())
        # Fix the baseline effective membrane; mass-only ablations must not change its stiffness.
        median_edge=float(np.median(np.linalg.norm(rest[self.faces[:,1]]-rest[self.faces[:,0]],axis=1)))
        et=p['young_pa']*p['membrane_thickness_m'];nu=p['poisson']
        fine_edge=np.median(np.linalg.norm(fine['vertices'][fine['faces'][:,1]]-fine['vertices'][fine['faces'][:,0]],axis=1))*scale
        base_wave=fine_edge*np.sqrt(density/(et/(2*(1+nu))))
        self.membrane_scale=min(1.,(.7*base_wave/5e-5)**2)
        effective=et*self.membrane_scale
        allowed=.7*median_edge*np.sqrt(density*mass_scale/(effective/(2*(1+nu))))
        self.dt=min(0.0004,max(5e-5,allowed)) if dt is None else dt
        if not cloth and dt is None:self.dt=.0004
        self.substeps=2*max(1,int(np.ceil(.005/self.dt/2)))
        self.dt=.005/self.substeps
        self.post_step=None
        self.pre_step=None
        self.known_external=None
        self.elapsed=0.
        self.contact_ke=float(p['newton_contact_modulus_pa_per_m']*va.mean())
        contact_kd=2*np.sqrt(self.contact_ke*self.masses.mean()/mass_scale)
        cfg=NewtonCfg(solver_cfg=MJWarpSolverCfg(integrator='implicitfast'))
        builder=NewtonMJWarpManager.create_builder(physics_cfg=cfg,gravity=(0,0,-9.81))
        shape_cfg=newton.ModelBuilder.ShapeConfig(density=0.,mu=p['friction'],ke=self.contact_ke,kd=contact_kd)
        # VBD's per-node damping is not the rigid-pair damping. Keep the two solver materials separate.
        rigid_contact={'mujoco:solref':wp.vec2(.040,1.),'mujoco:solref_mode':1}
        if category=='hard':
            desc=None
            self.box=builder.add_body(xform=wp.transform(center,wp.quat_identity()),mass=self.mass,
                inertia=wp.mat33(*np.diag(self.mass*(self.size@self.size-self.size**2)/12).ravel()),
                lock_inertia=True,label='content')
            self.box_shape=builder.add_shape_box(self.box,hx=self.size[0]/2,hy=self.size[1]/2,hz=self.size[2]/2,cfg=shape_cfg,
                custom_attributes=rigid_contact)
            self.content_bodies=[self.box];self.content_masses=np.array([self.mass])
            self.mapping=dict(bodies=self.content_bodies,represented_edges=0)
        else:
            from segment_adapter import add_content
            import copy
            desc=copy.deepcopy(json.loads((INPUT_ROOT/'isaaclab/results/descriptions/compressible-describe-1.json').read_text())['description'])
            shift=center-[0,0,self.size[2]/2+1e-4]
            for segment in desc['segments']:segment['pos']=(np.asarray(segment['pos'])+shift).tolist()
            self.mapping=add_content(builder,desc,loop_mode='tree')
            self.content_bodies=self.mapping['bodies'];self.box=self.content_bodies[0]
            self.content_masses=np.array([s['mass'] for s in desc['segments']])
        self.content_half=np.array([self.size/2] if category=='hard' else [s['half'] for s in desc['segments']])
        self.pads=[]
        self.pad_joints=[];origins=[]
        for sign in (-1,1):
            pose=wp.transform([0,sign*.022,float(initial[:,2].max()+.12)],wp.quat_identity())
            body=builder.add_link(xform=pose,
                mass=1.,inertia=wp.mat33(*np.diag([.001]*3).ravel()),lock_inertia=True,
                label=f'pad_{len(self.pads)}')
            axes=[newton.ModelBuilder.JointDofConfig(axis=axis,target_ke=100000.,target_kd=632.4555)
                  for axis in [(1,0,0),(0,1,0),(0,0,1)]]
            joint=builder.add_joint_d6(-1,body,parent_xform=pose,linear_axes=axes,angular_axes=[])
            builder.add_articulation([joint]);self.pad_joints.append(joint)
            origins.append(list(pose.p))
            add_fingertip(builder,body,shape_cfg,rigid_contact)
            self.pads.append(body)
        self.table=builder.add_link(mass=0.,lock_inertia=True,is_kinematic=True,label='table')
        builder.add_joint_fixed(-1,self.table)
        rigid_floor=shape_cfg.copy();rigid_floor.has_particle_collision=False
        cloth_floor=shape_cfg.copy();cloth_floor.has_shape_collision=False
        # MuJoCo accepts planes only on world; VBD's instrumented plane supplies cloth reaction.
        builder.add_shape_plane(width=0.,length=0.,cfg=rigid_floor,custom_attributes=rigid_contact)
        if cloth:builder.add_shape_plane(body=self.table,width=0.,length=0.,cfg=cloth_floor)
        if cloth:
            builder.add_cloth_mesh(pos=wp.vec3(0.),rot=wp.quat_identity(),scale=1.,vel=wp.vec3(0.),
                vertices=[wp.vec3(*v) for v in rest],indices=self.faces.ravel().tolist(),density=density*mass_scale,
                tri_ke=effective/(2*(1+nu)),tri_ka=effective*nu/(1-nu*nu),tri_kd=p['membrane_damping'],
                edge_ke=p['bending_rigidity_nm'],edge_kd=p['bending_damping_nms'],particle_radius=p['collision_radius_m'])
            for a,b in self.seams:
                # E*t already has spring units N/m for a patch with width/length of order one.
                builder.add_spring(int(a),int(b),ke=effective,kd=p['membrane_damping'],control=0.)
                builder.spring_rest_length[-1]=2*p['collision_radius_m']+.0002
        builder.color(include_bending=True)
        if cloth:color_seams(builder)
        self.model=builder.finalize(device=device)
        js=[] if desc is None else desc['joints']
        self.failure_a=wp.array([self.content_bodies[j['a']] for j in js],dtype=int,device=device)
        self.failure_b=wp.array([self.content_bodies[j['b']] for j in js],dtype=int,device=device)
        self.failure_arm=wp.array([np.eye(3)[j['axis']]*j['rest']/2 for j in js],dtype=wp.vec3,device=device)
        self.failure_compression=wp.array([j['lock_comp'] for j in js],dtype=float,device=device)
        self.failure_bend=wp.array([np.where(np.arange(3)==j['axis'],1e30,j['lock_bend']) for j in js],dtype=wp.vec3,device=device)
        self.failure_metrics=wp.zeros(4,device=device)
        self.model.soft_contact_ke=self.contact_ke
        self.model.soft_contact_kd=contact_kd
        self.model.soft_contact_mu=p['friction']
        if cloth:
            self.model.particle_mass.assign(self.masses.astype(np.float32))
            self.model.particle_inv_mass.assign((1/self.masses).astype(np.float32))
            # The same per-hinge law as the input profile; crease factors are kept.
            from bag_model import _topology
            edges,lengths,hinges,weights,hf,fa,topo=_topology(rest,self.faces)
            crease=np.any(np.isin(mesh['panel_ids'][hf],p['precreased_panel_ids']),axis=1)
            crease|=mesh['panel_ids'][hf][:,0]!=mesh['panel_ids'][hf][:,1]
            factors=np.where(crease,p['precreased_stiffness_ratio'],1.)
            lookup={tuple(sorted((int(a),int(b)))):i for i,(_,_,a,b) in enumerate(self.model.edge_indices.numpy())}
            properties=self.model.edge_bending_properties.numpy()
            edge_lengths=self.model.edge_rest_length.numpy()
            for (a,b,_,_),weight,factor in zip(hinges,weights,factors):
                i=lookup[tuple(sorted((int(a),int(b))))]
                properties[i]=[p['bending_rigidity_nm']*weight*factor/edge_lengths[i],
                               p['bending_damping_nms']*factor/edge_lengths[i]]
            self.model.edge_bending_properties.assign(properties)
        rigid=list(range(self.model.body_count))
        vbd_options=dict(iterations=10,rigid_compliant_alm=True,
            rigid_body_particle_contact_buffer_size=4096,particle_enable_self_contact=self_contact,
            particle_self_contact_margin=2*p['collision_radius_m'],particle_self_contact_gap=p['collision_radius_m'],
            friction_epsilon=.001)
        def rigid_solver(view):
            return SolverMuJoCo(view,use_mujoco_cpu=device=='cpu',use_mujoco_contacts=True,integrator='implicitfast',iterations=100,
                ls_iterations=20,tolerance=1e-8,njmax=4096,nconmax=4096)
        if cloth:
            self.solver=SolverCoupledProxy(self.model,entries=[
                SolverCoupledProxy.Entry(name='rigid',solver=rigid_solver,bodies=rigid,joints=list(range(self.model.joint_count))),
                SolverCoupledProxy.Entry(name='cloth',solver=lambda v:SolverVBD(v,**vbd_options),
                    particles=list(range(self.model.particle_count)))],
                coupling=SolverCoupledProxy.Config(proxies=[SolverCoupledProxy.Proxy(source='rigid',destination='cloth',
                    bodies=rigid,mode='lagged',collision_pipeline=lambda v:newton.CollisionPipeline(v,broad_phase='sap',
                        soft_contact_gap=p['collision_radius_m'],enable_rigid_soft_full_surface_contact=True),
                    collide_interval=1)],iterations=1))
            self.rigid_solver=self.solver.solver('rigid')
        else:
            self.solver=rigid_solver(self.model);self.rigid_solver=self.solver
        self.s0,self.s1=self.model.state(),self.model.state()
        newton.eval_fk(self.model,self.model.joint_q,self.model.joint_qd,self.s0)
        if cloth:
            self.s0.particle_q.assign(initial.astype(np.float32));self.s1.particle_q.assign(initial.astype(np.float32))
        self.control=self.model.control()
        self.pad_ids=wp.array(self.pads,dtype=int,device=self.model.device)
        self.pad_origins=wp.array(origins,dtype=wp.vec3,device=self.model.device)
        self.pad_coordinate_starts=[builder.joint_q_start[j] for j in self.pad_joints]
        self.pad_target_starts=wp.array(self.pad_coordinate_starts,dtype=int,device=self.model.device)
        self.pad_velocity_starts=wp.array([builder.joint_qd_start[j] for j in self.pad_joints],dtype=int,device=self.model.device)
        effort=self.control.joint_f.numpy()
        for k in self.pad_velocity_starts.numpy():effort[k+2]=9.81
        self.control.joint_f.assign(effort)
        self.targets=wp.array(self.s0.body_q.numpy()[self.pads,:3],dtype=wp.vec3,device=self.model.device)
        self.pad_velocity=wp.zeros(2,dtype=wp.vec3,device=self.model.device)
        self.pad_mj_ids=wp.array([self.rigid_solver.mj_model.body(f'pad_{i}').id for i in range(2)],dtype=int,device=self.model.device)
        self.pad_forces=wp.zeros(2,dtype=wp.vec3,device=self.model.device)
        self.table_force=wp.zeros(1,device=self.model.device)
        m=self.rigid_solver.mj_model
        body_map=self.rigid_solver.mjc_body_to_newton.numpy()[0]
        self.native_floor=np.asarray(m.geom_bodyid==0,dtype=np.int32)
        self.native_content=np.asarray(np.isin(body_map[m.geom_bodyid],self.content_bodies),dtype=np.int32)
        self.native_pads=np.full(m.ngeom,-1,np.int32)
        for i,b in enumerate(self.pad_mj_ids.numpy()):self.native_pads[m.geom_bodyid==b]=i
        self.known_external=wp.zeros(2,dtype=wp.vec3,device=self.model.device)
        if self.model.device.is_cuda:
            capacity=self.rigid_solver.mjw_data.contact.geom.shape[0]
            self.contact_ids=wp.array(np.arange(capacity),dtype=int,device=self.model.device)
            self.native_contact_force=wp.zeros(capacity,dtype=wp.spatial_vector,device=self.model.device)
            self.native_floor_flags=wp.array(self.native_floor,dtype=int,device=self.model.device)
            self.native_content_flags=wp.array(self.native_content,dtype=int,device=self.model.device)
            self.native_pad_flags=wp.array(self.native_pads,dtype=int,device=self.model.device)
        self.max_opening=wp.full(1,.095,device=self.model.device)
        if cloth:self.solver.solver('cloth').rebuild_bvh(self.solver.entry_state('cloth'))

    def step(self):
        ramp=float(self.params['newton_self_contact_ramp_s'])
        if self.cloth and self.elapsed<ramp:
            self.solver.solver('cloth').particle_self_contact_margin=(
                2*self.params['collision_radius_m']*min(1.,(self.elapsed+self.dt)/ramp)**2)
        self.elapsed+=self.dt
        self.s0.clear_forces()
        wp.launch(advance_targets,dim=2,inputs=[self.targets,self.pad_velocity,self.dt],device=self.model.device)
        wp.launch(limit_gap,dim=1,inputs=[self.targets,self.max_opening],device=self.model.device)
        wp.launch(drive_pads,dim=2,inputs=[self.targets,self.pad_velocity,self.pad_origins,
            self.pad_target_starts,self.pad_velocity_starts,self.control.joint_target_q,self.control.joint_target_qd],device=self.model.device)
        if self.cloth:
            wp.launch(particle_drag,dim=self.model.particle_count,inputs=[self.s0.particle_qd,self.model.particle_mass,
                self.s0.particle_f,float(self.params['velocity_damping_per_s'])],device=self.model.device)
        if self.pre_step is not None:self.pre_step(self)
        self.solver.step(self.s0,self.s1,self.control,None,self.dt)
        if self.model.device.is_cuda:
            import mujoco_warp
            native=self.rigid_solver.mjw_data
            mujoco_warp.contact_force(self.rigid_solver.mjw_model,native,self.contact_ids,True,self.native_contact_force)
            self.table_force.zero_()
            self.pad_forces.zero_()
            wp.launch(native_contact_reading,dim=len(self.contact_ids),inputs=[native.nacon,native.contact.geom,
                self.native_contact_force,self.native_floor_flags,self.native_content_flags,self.native_pad_flags,
                self.pad_forces,self.table_force])
        else:
            import mujoco
            force=np.zeros(6);total=0.;pads=np.zeros((2,3));data=self.rigid_solver.mj_data
            for i,c in enumerate(data.contact[:data.ncon]):
                a,b=int(c.geom1),int(c.geom2)
                if ((self.native_floor[a] or self.native_pads[a]>=0) and self.native_content[b]) or ((self.native_floor[b] or self.native_pads[b]>=0) and self.native_content[a]):
                    mujoco.mj_contactForce(self.rigid_solver.mj_model,data,i,force)
                    f=c.frame.reshape(3,3).T@force[:3]
                    if self.native_floor[a] or self.native_floor[b]:total+=max(float(f[2]) if self.native_floor[a] else -float(f[2]),0.)
                    if self.native_pads[a]>=0:pads[self.native_pads[a]]-=f
                    elif self.native_pads[b]>=0:pads[self.native_pads[b]]+=f
            self.table_force.fill_(total)
            self.pad_forces.assign(pads.astype(np.float32))
        if self.cloth:
            wp.launch(cloth_contact_reading,dim=2,inputs=[self.solver.entry_state('rigid','input').body_f,
                self.pad_ids,self.known_external,self.pad_forces,self.table,self.table_force],device=self.model.device)
        if self.post_step is not None:self.post_step(self)
        wp.launch(content_failure,dim=max(self.model.body_count,self.model.particle_count,self.failure_a.shape[0]),
            inputs=[self.s1.body_q,self.s1.body_qd,self.s1.particle_q,self.s1.particle_qd,
                self.failure_a,self.failure_b,self.failure_arm,self.failure_compression,self.failure_bend,self.failure_metrics],
            device=self.model.device)
        self.s0,self.s1=self.s1,self.s0

    def failure_reason(self):
        strain,compression,bending,nonfinite=self.failure_metrics.numpy()
        if nonfinite:return 'non-finite state or velocity'
        if strain>.5:return f'adjacent overlap {strain:.6g} of segment size'
        if compression>1.2:return f'compression {compression:.6g} x limit'
        if bending>1.2:return f'bending {bending:.6g} x limit'
        return None

    def frame(self,graph=None):
        if self.cloth:self.solver.solver('cloth').rebuild_bvh(self.solver.entry_state('cloth'))
        if graph is not None:wp.capture_launch(graph)
        else:
            for _ in range(self.substeps):self.step()

    def state_report(self):
        bodies=self.s0.body_q.numpy();particles=self.s0.particle_q.numpy() if self.cloth else np.empty((0,3))
        finite=bool(np.isfinite(bodies).all() and np.isfinite(particles).all())
        velocity=self.s0.particle_qd.numpy() if self.cloth else np.empty((0,3))
        return dict(finite=finite,
            content_position=np.average(bodies[self.content_bodies,:3],axis=0,weights=self.content_masses).tolist() if finite else None,
            max_particle_speed=(float(np.linalg.norm(velocity,axis=1).max()) if np.isfinite(velocity).all() else None) if self.cloth else 0.,
            particle_bounds=[particles.min(0).tolist(),particles.max(0).tolist()] if self.cloth and finite else None)

    def pad_top_contacts(self):
        if not self.cloth:return np.array([False,False])
        contacts=self.solver.get_proxy_contacts('rigid','cloth')
        n=int(contacts.soft_contact_count.numpy()[0])
        shapes=contacts.soft_contact_shape.numpy()[:n]
        normals=contacts.soft_contact_normal.numpy()[:n]
        points=contacts.soft_contact_body_pos.numpy()[:n]
        bodies=self.model.shape_body.numpy()[shapes]
        return np.array([np.any((bodies==b)&(normals[:,2]<-.5)&(points[:,2]<-.015)) for b in self.pads])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps',type=int,default=200)
    ap.add_argument('--scale',type=float,default=1.)
    ap.add_argument('--dt',type=float)
    ap.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    ap.add_argument('--category',choices=['hard','compressible'],default='hard')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():ap.error('refusing to overwrite output')
    start=time.perf_counter();scene=PackageScene(scale=a.scale,dt=a.dt,device=a.device,category=a.category);wp.synchronize()
    init=time.perf_counter()-start;start=time.perf_counter();snapshots=[]
    for i in range(a.steps):
        scene.step()
        if (i+1)%max(1,a.steps//10)==0:
            snapshots.append(dict(step=i+1,**scene.state_report()))
            if not snapshots[-1]['finite']:break
    wp.synchronize();wall=time.perf_counter()-start
    result=dict(steps=i+1,dt=scene.dt,initialization_s=init,wall_s=wall,simulation_s=(i+1)*scene.dt,
        model_bodies=scene.model.body_count,particles=scene.model.particle_count,
        membrane_scale=scene.membrane_scale,bag_mass_kg=float(scene.masses.sum()),wrap=scene.wrap_report,
        snapshots=snapshots,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='L5 coupled entry only, not a grasp or throughput acceptance')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
    if not snapshots[-1]['finite']:raise SystemExit(3)


if __name__=='__main__':main()

"""L5 protocol-v2.2 force-controlled package grasp, with cached rest and early termination."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import warp as wp

from package_assets import INPUT_ROOT, original_mesh
from package_scene import PackageScene
from pinch_effect import ClothPinch, point_connect
from package_metrics import geometry,ContactSliding

DT_CONTROL=.005


@wp.kernel
def sensors(pads:wp.array[wp.vec3],poses:wp.array[wp.transform],pad_ids:wp.array[int],
            reference:wp.array[float],spring_on:wp.array[int],broke:wp.array[int],
            values:wp.array[float],current_force:wp.array[float],phase:wp.array[int],
            velocity:wp.array[wp.vec3],first_contact:wp.array[float],threshold:float,dt:float):
    a=pads[0];b=pads[1]
    pa=wp.transform_get_translation(poses[pad_ids[0]])
    pb=wp.transform_get_translation(poses[pad_ids[1]])
    real=(wp.max(-a[1],0.)+wp.max(b[1],0.))/2.
    spring=float(0.)
    if spring_on[0]==1 and broke[0]==0:
        spring=40000.*wp.max(reference[0]-.010-(pb[1]-pa[1]-.004),0.)
    current_force[0]=real+spring
    values[0]=current_force[0]
    values[1]=values[1]+dt/(.005+dt)*(current_force[0]-values[1])
    values[2]=values[2]+dt/(.020+dt)*(current_force[0]-values[2])
    values[3]=wp.max(a[2]+b[2],0.)
    values[4]=wp.max(values[4],values[2])
    values[5]=real;values[6]=spring
    values[7]=wp.max(values[7],real);values[8]=wp.max(values[8],spring)
    if phase[0]==1 or phase[0]==2:
        if first_contact[0]==0. and (wp.length(a)>.05 or wp.length(b)>.05):
            first_contact[0]=1.
            first_contact[1]=(pa[2]+pb[2])/2.-.020
        if values[3]>=20.:
            velocity[0]=wp.vec3(velocity[0][0],velocity[0][1],0.)
            velocity[1]=wp.vec3(velocity[1][0],velocity[1][1],0.)
            first_contact[2]=1.
    if phase[0]>0 and values[2]>threshold:values[9]=1.


@wp.kernel
def effect_reaction(poses:wp.array[wp.transform],pad_ids:wp.array[int],
                    reference:wp.array[float],spring_on:wp.array[int],broke:wp.array[int],
                    active:wp.array[int],reaction:wp.array[wp.vec3],point:wp.array[wp.vec3],
                    body_force:wp.array[wp.spatial_vector],known:wp.array[wp.vec3]):
    a=wp.transform_get_translation(poses[pad_ids[0]]);b=wp.transform_get_translation(poses[pad_ids[1]])
    spring=float(0.)
    if spring_on[0]!=0 and broke[0]==0:spring=40000.*wp.max(reference[0]-.010-(b[1]-a[1]-.004),0.)
    for i in range(2):
        sign=float(2*i-1);force=wp.vec3(0.,sign*spring,0.);torque=wp.vec3(0.)
        if active[0]!=0:
            load=reaction[0]/2.;force=force+load
            torque=wp.cross(point[0]-wp.transform_get_translation(poses[pad_ids[i]]),load)
        body_force[pad_ids[i]]=body_force[pad_ids[i]]+wp.spatial_vector(force,torque)
        known[i]=force


@wp.kernel
def pinch_target(poses:wp.array[wp.transform],pad_ids:wp.array[int],offset:wp.array[wp.vec3],
                 targets:wp.array[wp.vec3]):
    a=wp.transform_get_translation(poses[pad_ids[0]])
    b=wp.transform_get_translation(poses[pad_ids[1]])
    targets[0]=(a+b)/2.-wp.vec3(0.,0.,.020)+offset[0]


def quat_matrix(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def surface_heights(vertices,faces,points):
    triangle=vertices[faces];a=triangle[:,0];u=triangle[:,1]-a;v=triangle[:,2]-a
    det=u[:,0]*v[:,1]-u[:,1]*v[:,0]
    heights=[]
    for p in points:
        d=p-a[:,:2];valid=np.abs(det)>1e-15;denom=np.where(valid,det,1.)
        b=(d[:,0]*v[:,1]-d[:,1]*v[:,0])/denom
        c=(u[:,0]*d[:,1]-u[:,1]*d[:,0])/denom
        valid&=(b>=-1e-8)&(c>=-1e-8)&(b+c<=1+1e-8)
        if not np.any(valid):return None
        heights.append(float(np.max((a[:,2]+b*u[:,2]+c*v[:,2])[valid])))
    return np.array(heights)


def oriented_bbox(points):
    points=np.unique(np.asarray(points),axis=0)
    def cross(a,b,c):
        u=b-a;v=c-a
        return float(u[0]*v[1]-u[1]*v[0])
    hull=[]
    for chain in (points,points[::-1]):
        side=[]
        for p in chain:
            while len(side)>1 and cross(side[-2],side[-1],p)<=0:side.pop()
            side.append(p)
        hull.extend(side[:-1])
    hull=np.asarray(hull);edges=np.roll(hull,-1,axis=0)-hull
    best=None
    for angle in np.unique(np.arctan2(edges[:,1],edges[:,0])%(np.pi/2)):
        R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        local=hull@R;lo,hi=local.min(0),local.max(0);size=hi-lo
        if best is None or np.prod(size)<best[0]:best=(np.prod(size),(lo+hi)/2@R.T,size,angle)
    _,center,size,angle=best
    if size[0]<size[1]:size=size[::-1];angle+=np.pi/2
    return center,size,float(angle%np.pi)


class Evaluation:
    def __init__(self,scene,mode='top',seed=20261105,w=.04,d=.015,F=10.,u=0.,v=0.,psi=np.pi/2,cache=None,capture=True):
        self.scene=scene;self.mode=mode;self.seed=seed;self.w=w;self.d=d;self.F=F
        self.psi=psi
        dev=scene.model.device
        self.pinch=ClothPinch(dev)
        self.reference=wp.zeros(1,device=dev);self.spring_on=wp.zeros(1,dtype=int,device=dev)
        self.values=wp.zeros(10,device=dev);self.phase_code=wp.zeros(1,dtype=int,device=dev)
        self.first_contact=wp.zeros(3,device=dev);self.offset=wp.zeros(1,dtype=wp.vec3,device=dev)
        self.known_external=scene.known_external
        self.threshold=15. if scene.category=='compressible' else 50.
        scene.post_step=self.after_physics
        scene.pre_step=self.before_physics
        self.graph=None;self.init_s=0.;self.settle_wall_s=0.;self.cache_used=False
        self.preparation_failure=None;self.initial_grasp_failure=None;self.preparation_trace=[]
        if cache and cache.exists():self.restore(cache)
        else:
            begin=time.perf_counter()
            for i in range(round(2./DT_CONTROL)):
                scene.frame(self.graph)
                if capture and dev.is_cuda and i==19:self.capture_graph()
                if (i+1)%10==0:
                    body,cloth=self.positions()
                    if not np.isfinite(body).all() or not np.isfinite(cloth).all():
                        self.preparation_failure='non-finite state during preparation';break
                    if scene.failure_reason():
                        self.preparation_failure=scene.failure_reason();break
                    metric=geometry(scene,body,cloth)
                    self.preparation_trace.append(dict(t=(i+1)*DT_CONTROL,**metric))
                    if metric['centres_outside']>0:
                        self.preparation_failure='content centre outside bag during preparation';break
            wp.synchronize();self.settle_wall_s=time.perf_counter()-begin
            if cache and self.preparation_failure is None:self.store(cache)
        # Rest is generated once; sensor/constraint state starts fresh for every attempt.
        self.values.zero_();self.first_contact.zero_()
        if self.preparation_failure:
            self.rest_com=self.rest_bag_com=self.original_top=self.required_opening=None
            self.rest_state_report=scene.state_report()
            return
        body,cloth=self.positions()
        if scene.cloth:points=cloth[:,:2]
        else:
            half=scene.content_half[0];signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
            points=(body[scene.box,:3]+(signs*half)@quat_matrix(body[scene.box,3:]).T)[:,:2]
        center,box_size,angle=oriented_bbox(points)
        turn=-angle-psi+np.pi/2
        self.object_frame=dict(obb_center_m=center.tolist(),obb_size_m=box_size.tolist(),obb_angle_rad=angle,
                               coordinate_turn_rad=turn,psi_rad=psi)
        R=np.array([[np.cos(turn),-np.sin(turn),0],[np.sin(turn),np.cos(turn),0],[0,0,1]])
        data=self.snapshot();q=data['body_q'];velocity=data['body_qd'];origin=np.r_[center,0.]
        rotation=np.array([0.,0.,np.sin(turn/2),np.cos(turn/2)])
        for b in scene.content_bodies:
            q[b,:3]=(q[b,:3]-origin)@R.T
            old=q[b,3:].copy()
            q[b,3:6]=rotation[3]*old[:3]+old[3]*rotation[:3]+np.cross(rotation[:3],old[:3])
            q[b,6]=rotation[3]*old[3]-rotation[:3]@old[:3]
            velocity[b,:3]=velocity[b,:3]@R.T;velocity[b,3:]=velocity[b,3:]@R.T
        data['joint_q'][:7]=q[scene.box];data['joint_qd'][:6]=velocity[scene.box]
        if scene.cloth:
            data['particle_q']=((data['particle_q']-origin)@R.T).astype(data['particle_q'].dtype)
            data['particle_qd']=(data['particle_qd']@R.T).astype(data['particle_qd'].dtype)
        self.load_snapshot(data)
        body,cloth=self.positions()
        self.rest_com=self.com(body);self.rest_bag_com=np.average(cloth,axis=0,weights=scene.masses) if scene.cloth else self.rest_com.copy()
        heading=np.pi/2-psi
        axes=np.array([[np.cos(heading),-np.sin(heading)],[np.sin(heading),np.cos(heading)]])
        pickup=axes@(np.array([u,v])*box_size)
        self.rest_state_report=scene.state_report()
        if mode=='side':
            footprint=np.ptp(cloth[:,:2],axis=0) if scene.cloth else scene.size[:2]
            self.required_opening=float(footprint[1])
            gap=.095
            z=max(float(self.rest_com[2]),.021)
        else:
            self.required_opening=w;gap=w
            xy=np.array([pickup+[0,-w/2-.002],pickup+[0,w/2+.002]])
            levels=surface_heights(cloth,scene.faces,xy) if scene.cloth else np.repeat(self.rest_com[2]+scene.size[2]/2,2)
            if levels is None:
                self.original_top=None;self.initial_grasp_failure='fingertips outside the package top';return
            self.original_top=levels;z=float(np.max(levels))+.005+.020
        targets=np.array([[pickup[0],pickup[1]-gap/2-.002,z],
                          [pickup[0],pickup[1]+gap/2+.002,z]],dtype=np.float32)
        scene.targets.assign(targets)
        q=scene.s0.body_q.numpy();q[scene.pads,:3]=targets
        scene.s0.body_q.assign(q);scene.s1.body_q.assign(q)
        for state in (scene.s0,scene.s1):
            coords=state.joint_q.numpy()
            for i,j in enumerate(scene.pad_coordinate_starts):coords[j:j+3]=targets[i]-scene.pad_origins.numpy()[i]
            state.joint_q.assign(coords)
        if capture and dev.is_cuda and self.graph is None:
            self.capture_graph()

    def capture_graph(self):
        scene=self.scene
        if scene.model.device.is_cuda:
            # Compile once before capture; restore the same physical starting state afterwards.
            snapshot=self.snapshot();start=time.perf_counter();scene.frame();wp.synchronize()
            self.load_snapshot(snapshot)
            with wp.ScopedCapture() as cap:
                for _ in range(scene.substeps):scene.step()
            self.graph=cap.graph
            self.load_snapshot(snapshot);self.init_s+=time.perf_counter()-start

    def before_physics(self,scene):
        wp.launch(effect_reaction,dim=1,inputs=[scene.s0.body_q,scene.pad_ids,self.reference,
            self.spring_on,self.pinch.broke,self.pinch.active,self.pinch.reaction,self.pinch.targets,
            scene.s0.body_f,self.known_external],device=scene.model.device)

    def after_physics(self,scene):
        wp.launch(sensors,dim=1,inputs=[scene.pad_forces,scene.s1.body_q,scene.pad_ids,
            self.reference,self.spring_on,self.pinch.broke,self.values,self.pinch.grip_force,
            self.phase_code,scene.pad_velocity,self.first_contact,self.threshold,scene.dt],device=scene.model.device)
        if scene.cloth:
            wp.launch(pinch_target,dim=1,inputs=[scene.s1.body_q,scene.pad_ids,self.offset,self.pinch.targets],device=scene.model.device)
            wp.launch(point_connect,dim=1,inputs=[scene.s1.particle_q,scene.s1.particle_qd,scene.model.particle_inv_mass,
                self.pinch.ids,self.pinch.weights,self.pinch.targets,self.pinch.active,self.pinch.coefficient,
                self.pinch.grip_force,self.pinch.filtered,self.pinch.reaction,self.pinch.broke,scene.dt],device=scene.model.device)

    def snapshot(self):
        s=self.scene.s0
        data={k:getattr(s,k).numpy().copy() for k in ('body_q','body_qd','joint_q','joint_qd','particle_q','particle_qd')
              if getattr(s,k) is not None}
        data['failure_metrics']=self.scene.failure_metrics.numpy().copy()
        return data

    def load_snapshot(self,data):
        scene=self.scene
        for state in (scene.s0,scene.s1):
            for k,v in data.items():
                if k!='failure_metrics':getattr(state,k).assign(v)
        scene.failure_metrics.assign(data['failure_metrics'])
        # flags=0 clears solver internals without replacing the supplied physical state.
        scene.solver.reset(scene.s0,flags=0)
        for k,v in data.items():
            current=scene.failure_metrics if k=='failure_metrics' else getattr(scene.s0,k)
            if not np.array_equal(current.numpy(),v):
                raise RuntimeError(f'solver reset changed cached {k}')
        self.values.zero_();self.first_contact.zero_()

    def store(self,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:
            np.savez(f,config_sha256=self.cache_fingerprint(),
                preparation_trace=json.dumps(getattr(self,'preparation_trace',[]),allow_nan=False),**self.snapshot())

    def restore(self,path):
        with np.load(path,allow_pickle=False) as g:
            if 'config_sha256' not in g or str(g['config_sha256'])!=self.cache_fingerprint():
                raise ValueError(f'cached state does not match this scene: {path}')
            self.load_snapshot({k:g[k] for k in g.files if k not in ('config_sha256','preparation_trace')})
            self.preparation_trace=json.loads(str(g['preparation_trace']))
        self.scene.elapsed=2.
        if self.scene.cloth:self.scene.solver.solver('cloth').particle_self_contact_margin=2*self.scene.params['collision_radius_m']
        self.cache_used=True

    def cache_fingerprint(self):
        s=self.scene;h=hashlib.sha256()
        for name in ('package_scene.py','package_assets.py','segment_adapter.py'):
            h.update((Path(__file__).parent/name).read_bytes())
        for a in (s.rest,s.initial,s.faces,s.masses,s.model.spring_rest_length.numpy() if s.cloth else np.array([])):
            h.update(np.ascontiguousarray(a).tobytes())
        for name in ('joint_type','joint_target_ke','joint_target_kd','body_mass','body_inertia'):
            h.update(getattr(s.model,name).numpy().tobytes())
        for name in ('shape_type','shape_scale','shape_transform','spring_ke','spring_kd'):
            a=getattr(s.model,name,None)
            if a is not None:h.update(a.numpy().tobytes())
        h.update(json.dumps(dict(category=s.category,self_contact=s.self_contact,cloth=s.cloth,dt=s.dt,
            membrane_scale=s.membrane_scale,preparation='zero-preload-sealed-rest2.0-all-neighbours',params=s.params,
            body_count=s.model.body_count),sort_keys=True).encode())
        h.update(s.model.body_mass.numpy().tobytes())
        h.update(s.model.body_q.numpy().tobytes())
        for a in (s.rigid_solver.mj_model.geom_solref,s.rigid_solver.mj_model.geom_solimp):h.update(a.tobytes())
        return h.hexdigest()

    def positions(self):
        s=self.scene.s0
        return s.body_q.numpy(),s.particle_q.numpy() if self.scene.cloth else np.empty((0,3))

    def com(self,bodies):
        return np.average(bodies[self.scene.content_bodies,:3],axis=0,weights=self.scene.content_masses)

    def lowest(self,bodies):
        s=self.scene
        return min(bodies[b,2]-(np.abs(quat_matrix(bodies[b,3:]))@half)[2]
                   for b,half in zip(s.content_bodies,s.content_half))

    def eligible(self,bodies,forces):
        tips=bodies[self.scene.pads,:3]-[0,0,.020]
        return bool(self.scene.cloth and np.all(self.original_top-tips[:,2]>=.003-1e-8)
                    and np.all(forces[:,2]>.05) and np.all(self.scene.pad_top_contacts()))

    def run(self,limit=90.):
        s=self.scene
        phase='approach';phase_t=t=0.;outcome=reason=None
        stages=[dict(phase=phase,t=0.)];trace=[]
        self.phase_code.fill_(1)
        carry_angle=np.pi/2-float(np.random.default_rng(self.seed).uniform(0,2*np.pi))-self.psi
        carry=.5*np.array([np.cos(carry_angle),np.sin(carry_angle),0.])
        carry_T=np.sqrt(np.pi/2)
        lift_start=press_start=None;effective_d=self.d;lost=0.;side_seen=np.zeros(2,bool)
        max_relative=0.;table_hold=None;last_bound=False;peak_load=0.
        max_penetration=0.;max_corners_outside=0.;max_centres_outside=0.;sliding=ContactSliding()
        start=time.perf_counter()
        next_progress=1.
        def go(name):
            nonlocal phase,phase_t
            phase=name;phase_t=t;stages.append(dict(phase=name,t=t))
            self.phase_code.fill_(dict(approach=1,press=2,close=3,lift=4,carry=5,lower=6,open=7)[name])
        if self.preparation_failure:
            outcome,reason='sim_fail',self.preparation_failure
        elif self.initial_grasp_failure:
            outcome,reason='grasp_fail',self.initial_grasp_failure
        elif self.mode=='side' and self.required_opening>.085:
            outcome,reason='grasp_fail','package width exceeds 85 mm side-grasp scope'
        while outcome is None and t<limit:
            body,cloth=self.positions();com=self.com(body)
            if not np.isfinite(body).all() or not np.isfinite(cloth).all():
                outcome,reason='sim_fail','non-finite state';break
            if s.failure_reason():outcome,reason='sim_fail',s.failure_reason();break
            readings=self.values.numpy();forces=s.pad_forces.numpy()
            if not np.isfinite(readings).all() or not np.isfinite(forces).all():
                outcome,reason='sim_fail','non-finite contact or force reading';break
            active=bool(self.pinch.active.numpy()[0]);broken=bool(self.pinch.broke.numpy()[0])
            if last_bound and not active and broken:
                self.pinch.events.append(dict(event='overload',t=t,filtered_force_N=float(self.pinch.filtered.numpy()[0]),
                                              limit_N=2.*float(readings[0])))
                self.spring_on.zero_()
            last_bound=active
            if readings[9]>0:
                outcome,reason='over_squeeze',f'{readings[2]} N in {phase}';break
            tau=t-phase_t
            velocity=np.zeros((2,3),np.float32)
            tips=body[s.pads,:3]-[0,0,.020]
            if self.mode=='side' and phase=='approach':
                side_seen|=np.linalg.norm(forces,axis=1)>.05
                if np.all(side_seen):
                    s.max_opening.fill_(min(.095,float(body[s.pads[1],1]-body[s.pads[0],1]-.004)+.005))
                    go('close')
                else:
                    velocity[0,1]=0. if side_seen[0] else .025
                    velocity[1,1]=0. if side_seen[1] else -.025
            elif phase=='approach':
                seen=self.first_contact.numpy()
                if seen[0]:
                    press_start=float(seen[1]);effective_d=min(self.d,.6*max(press_start,0.));go('press')
                elif np.mean(tips[:,2])<-.010:outcome,reason='grasp_fail','fingertips passed below table without contact'
                else:velocity[:,2]=-.010
            elif phase=='press':
                stop=self.first_contact.numpy()[2]>0 or float(np.mean(tips[:,2]))<=press_start-effective_d
                if stop:
                    gap=float(body[s.pads[1],1]-body[s.pads[0],1]-.004)
                    self.reference.fill_(gap);self.spring_on.fill_(int(self.eligible(body,forces)));go('close')
                    s.max_opening.fill_(min(.095,gap+.005))
                else:velocity[:,2]=-.010
            elif phase=='close':
                gap=float(body[s.pads[1],1]-body[s.pads[0],1]-.004)
                if readings[1]>=.95*self.F or gap<=0 or tau>=8.:
                    if self.mode=='top':
                        if readings[0]>0 and self.eligible(body,forces):
                            self.pinch.bind(cloth,s.faces,np.mean(tips,axis=0),t)
                            self.offset.assign(np.array([self.pinch.offset],np.float32));last_bound=True
                            lift_start=float(np.mean(body[s.pads,2]));go('lift')
                        else:outcome,reason='grasp_fail','top pinch contact/depth/force absent'
                    else:lift_start=float(np.mean(body[s.pads,2]));go('lift')
            elif phase=='lift':
                if tau<2.5:velocity[:,2]=min(.1,max(lift_start+.25-float(np.mean(body[s.pads,2])),0.)/DT_CONTROL)
                elif tau>=2.7:
                    table_hold=bool(self.lowest(body)<=0 or (s.cloth and np.min(cloth[:,2])<=s.params['collision_radius_m']))
                    if table_hold:outcome,reason='grasp_fail','package still contacts table after lift hold'
                    else:go('carry')
            elif phase=='carry':
                if tau<carry_T:velocity[:]=carry/carry_T*(1-np.cos(2*np.pi*tau/carry_T))
                elif tau>=carry_T+.3:
                    if com[2]-self.rest_com[2]<.03:outcome,reason='drop','content returned to table'
                    else:go('lower')
            elif phase=='lower':
                if float(s.table_force.numpy()[0])>=.3*(s.mass+(s.masses.sum() if s.cloth else 0.))*9.81:
                    self.pinch.release(t);self.spring_on.zero_();go('open')
                    s.max_opening.fill_(.095)
                elif float(np.mean(body[s.pads,2]))<=lift_start+.005:
                    self.pinch.release(t);self.spring_on.zero_();go('open')
                    s.max_opening.fill_(.095)
                else:velocity[:,2]=-.2
            elif phase=='open':
                if tau<.6:velocity[0,1]=-.05;velocity[1,1]=.05
                elif tau>=1.1:
                    error=com[:2]-self.rest_com[:2]-carry[:2]
                    outcome='success' if np.all(np.abs(error)<=.1) else 'drop'
                    reason=None if outcome=='success' else 'COM outside target square'
            if phase in ('close','lift','carry','lower'):
                gathering=(phase=='close' and self.mode=='top' and self.spring_on.numpy()[0]
                           and readings[6]<=.01 and readings[5]<=.05)
                closing=.025 if gathering else float(np.clip(.0001*(self.F-readings[1]),-.025,.025))
                velocity[0,1]+=closing;velocity[1,1]-=closing
            if phase in ('lift','carry'):
                has_contact=active or bool(np.max(np.linalg.norm(forces,axis=1))>.05)
                lost=0. if has_contact else lost+DT_CONTROL
                if lost>.050:
                    outcome='grasp_fail' if phase=='lift' else 'drop';reason='both contacts lost for >50 ms'
            if s.cloth:
                relative=(com-np.average(cloth,axis=0,weights=s.masses))-(self.rest_com-self.rest_bag_com)
                max_relative=max(max_relative,float(np.linalg.norm(relative)))
            peak_load=max(peak_load,float(np.linalg.norm(self.pinch.reaction.numpy()[0])))
            if len(trace)==0 or t-trace[-1]['t']>=.05:
                metric=geometry(s,body,cloth,penetration=True)
                max_penetration=max(max_penetration,metric['penetration_mm'])
                max_corners_outside=max(max_corners_outside,metric['corners_outside'])
                max_centres_outside=max(max_centres_outside,metric['centres_outside'])
                sliding.update(s,body,cloth,t)
                if metric['centres_outside']>0:outcome,reason='sim_fail','content centre outside bag'
                trace.append(dict(t=t,phase=phase,com=com.tolist(),pin_active=active,
                    table_force_N=float(s.table_force.numpy()[0]),lowest_content_m=self.lowest(body),
                    pad_positions_m=body[s.pads,:3].tolist(),
                    grip_force_N=float(readings[0]),control_force_N=float(readings[1]),measured_force_N=float(readings[2]),
                    real_force_N=float(readings[5]),spring_force_N=float(readings[6]),vertical_force_N=float(readings[3]),**metric))
            if outcome is not None:break
            s.pad_velocity.assign(velocity);s.frame(self.graph);t+=DT_CONTROL
            if t>=next_progress:
                print(json.dumps(dict(progress_t=t,phase=phase,grip_force_N=float(readings[0]),pin_active=active)),flush=True)
                next_progress+=1.
        if outcome is None:
            outcome='incomplete' if limit<90 else 'grasp_fail'
            reason='short smoke ended before task result' if limit<90 else 'execution timeout'
        wp.synchronize();wall=time.perf_counter()-start
        return dict(outcome=outcome,reason=reason,simulation_s=t,wall_s=wall,phase=phase,
            completed_attempt=outcome not in ('incomplete','sim_fail'),success=outcome=='success',
            bag_content_relative_motion_mm=max_relative*1000,peak_connection_force_N=peak_load,
            material_contact_sliding_mm=sliding.max_mm,sliding_evidence=sliding.evidence,
            sliding_measurements=sliding.measured,max_penetration_mm=max_penetration,
            max_centres_outside=max_centres_outside,max_corners_outside=max_corners_outside,
            preparation_failure=self.preparation_failure,preparation_trace=self.preparation_trace,
            simulation_failure_metrics=self.scene.failure_metrics.numpy().tolist(),
            rest_state_report=self.rest_state_report,
            table_contact_after_lift_hold=table_hold,phases=stages,pinch_events=self.pinch.events,trace=trace,
            cache_used=self.cache_used,settle_wall_s=self.settle_wall_s,capture_warmup_s=self.init_s,
            wrap_relaxation_s=1.7,protocol_rest_s=.3,
            membrane_scale=s.membrane_scale,dt=s.dt,substeps=s.substeps,category=s.category,grasp=self.mode,
            required_opening_m=self.required_opening,
            object_frame=getattr(self,'object_frame',None),
            limitations=['VBD elastic cloth; material plasticity/memory not migrated',
                         'relative COM motion is diagnostic only; contact-material tangential motion is measured separately'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--scale',type=float,default=1.)
    ap.add_argument('--category',choices=['hard','compressible'],default='hard')
    ap.add_argument('--grasp',choices=['top','side'],default='top');ap.add_argument('--seed',type=int,default=20261105)
    ap.add_argument('--u',type=float,default=0.);ap.add_argument('--v',type=float,default=0.)
    ap.add_argument('--psi',type=float,default=np.pi/2)
    ap.add_argument('--w',type=float,default=.04);ap.add_argument('--d',type=float,default=.015);ap.add_argument('--force',type=float,default=10.)
    ap.add_argument('--mass-scale',type=float,choices=[1.,5.,10.],default=1.)
    ap.add_argument('--no-self-contact',action='store_true');ap.add_argument('--coarse',type=Path)
    ap.add_argument('--no-cloth',action='store_true');ap.add_argument('--cache',type=Path)
    ap.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    ap.add_argument('--limit',type=float,default=90.);ap.add_argument('--no-capture',action='store_true')
    a=ap.parse_args()
    if a.output.exists():ap.error('refusing to overwrite output')
    if not(-.5<=a.u<=.5 and -.5<=a.v<=.5 and 0<=a.psi<np.pi and .020<=a.w<=.095 and 0<=a.d<=.040 and 2<=a.force<=50):
        ap.error('grasp parameters outside protocol v2.2 bounds')
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path(__file__).parent.glob('package_*.py')}
    source_sha256['pinch_effect.py']=hashlib.sha256((Path(__file__).parent/'pinch_effect.py').read_bytes()).hexdigest()
    source_sha256['segment_adapter.py']=hashlib.sha256((Path(__file__).parent/'segment_adapter.py').read_bytes()).hexdigest()
    input_manifest=INPUT_ROOT/'manifest.json'
    mesh=None
    if a.coarse:
        with np.load(a.coarse,allow_pickle=False) as g:mesh={k:g[k] for k in g.files}
    start=time.perf_counter()
    scene=PackageScene(scale=a.scale,mass_scale=a.mass_scale,self_contact=not a.no_self_contact,
        cloth=not a.no_cloth,mesh=mesh,category=a.category,device=a.device)
    evaluation=Evaluation(scene,a.grasp,a.seed,w=a.w,d=a.d,F=a.force,u=a.u,v=a.v,psi=a.psi,cache=a.cache,capture=not a.no_capture)
    initialization=time.perf_counter()-start
    q=evaluation.run(a.limit)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        initialization_s=initialization,result=q,
        source_sha256=source_sha256,
        input_manifest=json.loads(input_manifest.read_text()) if input_manifest.exists() else None)
    if q['outcome']=='sim_fail':
        evidence=a.output.with_suffix('.state.npz')
        with evidence.open('xb') as f:np.savez(f,**evaluation.snapshot())
        payload['failure_state']=str(evidence)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(payload,f,indent=2,allow_nan=False)
    print(json.dumps(q,indent=2))
    if q['outcome']=='sim_fail':raise SystemExit(3)


if __name__=='__main__':main()

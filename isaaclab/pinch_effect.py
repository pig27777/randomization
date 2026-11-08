"""V1.3 point attachment for a material point on a VBD triangle; no rotational constraint."""
from __future__ import annotations

import numpy as np
import warp as wp

FILTER_TAU=.005
SPRING_K=40000.
GATHER_TRAVEL=.010
DEPTH_MIN=.003


def lowpass_force(previous,force,dt):
    if dt<=0 or not np.isfinite([previous,force,dt]).all() or min(previous,force)<0:
        raise ValueError('finite nonnegative forces and positive dt required')
    return previous+dt/(FILTER_TAU+dt)*(force-previous)


def spring_force(reference_gap,gap,armed=True):
    return SPRING_K*max(reference_gap-GATHER_TRAVEL-gap,0.) if armed else 0.


def closest_material_point(vertices,faces,point):
    """Closest triangle point and barycentric weights; reject degenerate triangles."""
    triangles=vertices[faces];a=triangles[:,0];u=triangles[:,1]-a;v=triangles[:,2]-a
    normal=np.cross(u,v);norm2=np.sum(normal*normal,axis=1)
    good=norm2>1e-18
    if not np.any(good):raise ValueError('no nondegenerate surface triangle')
    delta=point-a
    uu=np.sum(u*u,axis=1);vv=np.sum(v*v,axis=1);uv=np.sum(u*v,axis=1)
    du=np.sum(delta*u,axis=1);dv=np.sum(delta*v,axis=1)
    det=uu*vv-uv*uv
    safe=np.where(good,det,1.)
    b=(du*vv-dv*uv)/safe;c=(dv*uu-du*uv)/safe
    bary=np.c_[1-b-c,b,c]
    candidates=[]
    valid=good&np.all(bary>=0,axis=1)
    for i in np.flatnonzero(valid):candidates.append((int(i),bary[i]))
    for i,j in ((0,1),(1,2),(2,0)):
        edge=triangles[:,j]-triangles[:,i]
        length2=np.sum(edge*edge,axis=1)
        t=np.clip(np.sum((point-triangles[:,i])*edge,axis=1)/np.maximum(length2,1e-24),0,1)
        for k in np.flatnonzero(good):
            weights=np.zeros(3);weights[i]=1-t[k];weights[j]=t[k]
            candidates.append((int(k),weights))
    i,weights=min(candidates,key=lambda x:np.linalg.norm(x[1]@triangles[x[0]]-point))
    return i,weights,weights@triangles[i]


@wp.kernel
def point_connect(q:wp.array[wp.vec3],qd:wp.array[wp.vec3],inverse_mass:wp.array[float],
                  ids:wp.array2d[int],weights:wp.array[wp.vec3],targets:wp.array[wp.vec3],
                  active:wp.array[int],coefficient:wp.array[float],grip_force:wp.array[float],
                  filtered:wp.array[float],reaction:wp.array[wp.vec3],broke:wp.array[int],dt:float):
    e=wp.tid()
    reaction[e]=wp.vec3(0.)
    if active[e]==0:return
    a=ids[e,0];b=ids[e,1];c=ids[e,2];w=weights[e]
    point=w[0]*q[a]+w[1]*q[b]+w[2]*q[c]
    metric=w[0]*w[0]*inverse_mass[a]+w[1]*w[1]*inverse_mass[b]+w[2]*w[2]*inverse_mass[c]
    if metric<=0.:return
    impulse_position=(targets[e]-point)/metric
    da=inverse_mass[a]*w[0]*impulse_position
    db=inverse_mass[b]*w[1]*impulse_position
    dc=inverse_mass[c]*w[2]*impulse_position
    q[a]=q[a]+da;q[b]=q[b]+db;q[c]=q[c]+dc
    qd[a]=qd[a]+da/dt;qd[b]=qd[b]+db/dt;qd[c]=qd[c]+dc/dt
    force=-impulse_position/(dt*dt)
    reaction[e]=force
    filtered[e]=filtered[e]+dt/(FILTER_TAU+dt)*(wp.length(force)-filtered[e])
    if filtered[e]>coefficient[e]*grip_force[e]:
        active[e]=0;broke[e]=1


class ClothPinch:
    """Three position DOFs at a barycentric material point, with the reviewed load filter."""
    def __init__(self,device):
        self.ids=wp.zeros((1,3),dtype=int,device=device)
        self.weights=wp.zeros(1,dtype=wp.vec3,device=device)
        self.targets=wp.zeros(1,dtype=wp.vec3,device=device)
        self.active=wp.zeros(1,dtype=int,device=device)
        self.coefficient=wp.full(1,2.,device=device)
        self.grip_force=wp.zeros(1,device=device)
        self.filtered=wp.zeros(1,device=device)
        self.reaction=wp.zeros(1,dtype=wp.vec3,device=device)
        self.broke=wp.zeros(1,dtype=int,device=device)
        self.offset=None
        self.events=[]

    def bind(self,vertices,faces,midpoint,t):
        i,w,point=closest_material_point(vertices,faces,np.asarray(midpoint))
        self.ids.assign(faces[i:i+1].astype(np.int32));self.weights.assign(np.array([w],dtype=np.float32))
        self.offset=point-midpoint
        self.targets.assign(np.array([point],dtype=np.float32))
        self.filtered.zero_();self.broke.zero_();self.active.fill_(1)
        self.events.append(dict(event='bind',t=t,triangle=int(i),vertices=faces[i].tolist(),
                                weights=w.tolist(),point=point.tolist(),midpoint=np.asarray(midpoint).tolist(),
                                point_offset_mm=(self.offset*1000).tolist()))

    def release(self,t,reason='open'):
        self.active.zero_()
        self.events.append(dict(event=reason,t=t))

    def project(self,scene,midpoint,grip_force):
        if self.offset is None:return
        self.targets.assign(np.array([midpoint+self.offset],dtype=np.float32))
        self.grip_force.fill_(max(float(grip_force),0.))
        wp.launch(point_connect,dim=1,inputs=[scene.s1.particle_q,scene.s1.particle_qd,scene.model.particle_inv_mass,
            self.ids,self.weights,self.targets,self.active,self.coefficient,self.grip_force,
            self.filtered,self.reaction,self.broke,scene.dt])

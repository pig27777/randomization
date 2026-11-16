"""Material-coordinate observations of native two-finger membrane contact.

This module adds no attachment forces. Coulomb friction, slip and detachment
are solved by the native film contacts. The state records which material was
pinched and a single shared-node ledger prevents face-switch budget renewal.
"""
import numpy as np
import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5


@wp.struct
class ContactState:
    mode: int
    grasp_class: int  # 1 gathered-surplus pinch, 2 resolved native pinch without surplus allocation
    face_id: int
    element: int
    barycentric: wp.vec3
    local_contact_position: wp.vec3
    local_surface_normal: wp.vec3
    local_surface_tangent0: wp.vec3
    local_surface_tangent1: wp.vec3
    material_reference: wp.vec3
    gathered_length: float
    captured_length: float
    slip_distance: float
    previous_gap: float
    closed: int
    lost_time: float
    normal_load: float


@wp.kernel
def extract(cone: int, nacon: wp.array(dtype=int), world: wp.array(dtype=int),
            geom: wp.array(dtype=wp.vec2i), flex: wp.array(dtype=wp.vec2i), elem: wp.array(dtype=wp.vec2i),
            vert: wp.array(dtype=wp.vec2i), frame: wp.array(dtype=wp.mat33),
            friction: wp.array(dtype=vec5), dim: wp.array(dtype=int), address: wp.array2d(dtype=int),
            adhesion: wp.array(dtype=float), efc: wp.array2d(dtype=float), njmax: int,
            side_of_geom: wp.array(dtype=int), vertex_triangle: wp.array(dtype=int),
            force: wp.array(dtype=wp.vec3), elements: wp.array(dtype=int), sides: wp.array(dtype=int)):
    c=wp.tid()
    if c<nacon[0]:
        side=-1;gside=-1;fside=-1
        for j in range(2):
            if geom[c][j]>=0:
                if side_of_geom[geom[c][j]]>=0:
                    side=side_of_geom[geom[c][j]];gside=j;fside=1-j
        if side>=0 and flex[c][fside]==0:
            value=contact_force_fn(cone,frame,friction,dim,address,adhesion,efc,njmax,nacon,world[c],c,False)
            f=wp.max(value[0],0.0)*wp.vec3(frame[c][0,0],frame[c][0,1],frame[c][0,2])
            if gside==0:f=-f
            triangle=elem[c][fside]
            if triangle<0 and vert[c][fside]>=0:
                triangle=vertex_triangle[vert[c][fside]]
            if triangle>=0:
                force[c]=f;elements[c]=triangle;sides[c]=side


@wp.kernel
def collect(nacon: wp.array(dtype=int), world: wp.array(dtype=int), forces: wp.array(dtype=wp.vec3),
            sides: wp.array(dtype=int), total: wp.array2d(dtype=wp.vec3), winner: wp.array2d(dtype=int)):
    c=wp.tid()
    if c<nacon[0] and sides[c]>=0:
        if wp.length(forces[c])>.05:
            wp.atomic_add(total,world[c],sides[c],forces[c])
            wp.atomic_min(winner,world[c],sides[c],c)


@wp.func
def barycentric(p: wp.vec3,a: wp.vec3,b: wp.vec3,c: wp.vec3):
    v0=b-a;v1=c-a;v2=p-a
    d00=wp.dot(v0,v0);d01=wp.dot(v0,v1);d11=wp.dot(v1,v1)
    d20=wp.dot(v2,v0);d21=wp.dot(v2,v1)
    denom=wp.max(d00*d11-d01*d01,1.e-20)
    v=(d11*d20-d01*d21)/denom;w=(d00*d21-d01*d20)/denom
    result=wp.vec3(wp.max(1.0-v-w,0.0),wp.max(v,0.0),wp.max(w,0.0))
    return result/(result[0]+result[1]+result[2])


@wp.kernel
def update(dt: float, close: int, release: int, requested_face: int,
           winner: wp.array2d(dtype=int), forces: wp.array2d(dtype=wp.vec3),
           elements: wp.array(dtype=int), positions: wp.array(dtype=wp.vec3),
           geom_xpos: wp.array2d(dtype=wp.vec3), pads: wp.array(dtype=int),
           triangles: wp.array(dtype=wp.vec3i), faces: wp.array(dtype=int),
           points: wp.array2d(dtype=wp.vec3), rest: wp.array2d(dtype=wp.vec3),
           available: wp.array2d(dtype=float), gathered: wp.array2d(dtype=float),
           edges: wp.array(dtype=wp.vec2i), damage: wp.array2d(dtype=int),
           states: wp.array(dtype=ContactState)):
    e=wp.tid();s=states[e]
    axis=geom_xpos[e,pads[1]]-geom_xpos[e,pads[0]]
    gap=wp.length(axis);axis=axis/wp.max(gap,1.e-10)
    speed=wp.max((s.previous_gap-gap)/dt,0.0)
    s.previous_gap=gap
    left=winner[e,0];right=winner[e,1]
    touching=left<2147483647 and right<2147483647
    inward0=wp.max(-wp.dot(forces[e,0],axis),0.0)
    inward1=wp.max(wp.dot(forces[e,1],axis),0.0)
    s.normal_load=2.0*wp.min(inward0,inward1)
    pinching=touching and inward0>.05 and inward1>.05
    event_close=close
    if close==2:event_close=int(pinching)
    if release==1:
        event_close=0
        s.mode=0;s.grasp_class=0;s.lost_time=0.0;s.captured_length=0.0;s.slip_distance=0.0
    if touching:
        t=elements[left];ids=triangles[t]
        a=points[e,ids[0]];b=points[e,ids[1]];c=points[e,ids[2]]
        weights=barycentric(positions[left],a,b,c)
        reference=weights[0]*rest[e,ids[0]]+weights[1]*rest[e,ids[1]]+weights[2]*rest[e,ids[2]]
        if s.mode==0:
            s.face_id=faces[t];s.element=t;s.barycentric=weights
            s.local_contact_position=reference
            n=wp.cross(b-a,c-a);n=n/wp.max(wp.length(n),1.e-10)
            tangent=(b-a)/wp.max(wp.length(b-a),1.e-10)
            s.local_surface_normal=n;s.local_surface_tangent0=tangent;s.local_surface_tangent1=wp.cross(n,tangent)
            # Only a local pair of film contacts gathers the same patch.
            other=triangles[elements[right]]
            local=False
            for i in range(3):
                for j in range(3):
                    if ids[i]==other[j]:local=True
            if local:
                tangent_fraction=wp.sqrt(wp.max(1.0-wp.dot(axis,n)*wp.dot(axis,n),0.0))
                increment=speed*dt*tangent_fraction
                for i in range(3):
                    if weights[i]>1.e-8:
                        increment=wp.min(increment,wp.max(available[e,ids[i]]-gathered[e,ids[i]],0.0)/weights[i])
                for i in range(3):gathered[e,ids[i]]+=increment*weights[i]
            amount=float(0.0)
            for i in range(3):amount+=gathered[e,ids[i]]
            s.gathered_length=amount
            if event_close==1 and s.closed==0 and pinching and (requested_face<0 or requested_face==faces[t]):
                s.mode=1;s.captured_length=wp.min(amount,.02)
                s.grasp_class=2
                if amount>.001:s.grasp_class=1
                s.material_reference=reference;s.slip_distance=0.0;s.lost_time=0.0
        else:
            # Reference-coordinate motion of contact records native material slip.
            s.slip_distance=wp.max(s.slip_distance,wp.length(reference-s.material_reference))
    if s.mode==1 or s.mode==2:
        if pinching:
            s.lost_time=0.0
            s.mode=1
            if s.slip_distance>.0001:s.mode=2
        else:
            s.lost_time+=dt
            if s.lost_time>=.02:s.mode=3
        ids=triangles[s.element]
        for h in range(edges.shape[0]):
            if damage[e,h]==1:
                for i in range(3):
                    if edges[h][0]==ids[i] or edges[h][1]==ids[i]:s.mode=4
    s.closed=event_close
    states[e]=s


class Observer:
    def __init__(self,m,surface,pad_groups,device='cuda:0',capacity=256):
        self.device=device;self.surface=surface;self.capacity=capacity
        with wp.ScopedDevice(device):
            mapping=np.full(m.ngeom,-1,dtype=np.int32)
            for side,group in enumerate(pad_groups):mapping[list(group)]=side
            self.mapping=wp.array(mapping,dtype=int)
            self.pads=wp.array([min(group) for group in pad_groups],dtype=int)
            first=np.zeros(len(surface.topo.vertices),dtype=np.int32)
            for v in range(len(first)):first[v]=np.flatnonzero((surface.topo.triangles==v).any(1))[0]
            self.first=wp.array(first,dtype=int)
            self.faces=wp.array(surface.topo.face_id,dtype=int)
            self.states=wp.zeros(surface.nworld,dtype=ContactState)
            self.gathered=wp.zeros_like(surface.available)
            self.total=wp.zeros((surface.nworld,2),dtype=wp.vec3)
            self.winner=wp.empty((surface.nworld,2),dtype=int)
            self.forces=wp.zeros(capacity,dtype=wp.vec3)
            self.elements=wp.zeros(capacity,dtype=int)
            self.sides=wp.empty(capacity,dtype=int)

    def reset(self):
        self.states.zero_();self.gathered.zero_()

    def observe_mjw(self,m,d,close=0,release=0,face_id=-1,dt=.0005):
        with wp.ScopedDevice(self.device):
            self.sides.fill_(-1);self.forces.zero_()
            c=d.contact
            wp.launch(extract,self.capacity,[int(m.opt.cone),d.nacon,c.worldid,c.geom,c.flex,c.elem,c.vert,c.frame,
                c.friction,c.dim,c.efc_address,c.adhesion,d.efc.force,d.njmax,self.mapping,self.first,
                self.forces,self.elements,self.sides])
            self._update(d.nacon,c.worldid,c.pos,d.geom_xpos,d.flexvert_xpos,close,release,face_id,dt)

    def _update(self,nacon,world,positions,geom_xpos,points,close,release,face_id,dt):
        with wp.ScopedDevice(self.device):
            self.total.zero_();self.winner.fill_(2147483647)
            wp.launch(collect,self.capacity,[nacon,world,self.forces,self.sides,self.total,self.winner])
            s=self.surface
            wp.launch(update,s.nworld,[dt,close,release,face_id,self.winner,self.total,self.elements,positions,
                geom_xpos,self.pads,s.triangles,self.faces,points,s.rest,s.available,self.gathered,
                s.edges,s.damage,self.states])

    def observe_cpu(self,m,d,close=0,release=0,face_id=-1):
        import mujoco
        n=d.ncon
        if n>self.capacity:raise RuntimeError(f'Contact observer capacity {self.capacity} exceeded by {n}')
        forces=np.zeros((self.capacity,3),dtype=np.float32)
        sides=np.full(self.capacity,-1,dtype=np.int32);elements=np.zeros(self.capacity,dtype=np.int32)
        mapping=self.mapping.numpy();first=self.first.numpy();scratch=np.zeros(6)
        for c in range(n):
            contact=d.contact[c]
            for j in range(2):
                geom=contact.geom[j]
                if geom>=0 and mapping[geom]>=0 and contact.flex[1-j]==0:
                    mujoco.mj_contactForce(m,d,c,scratch)
                    f=contact.frame[:3]*max(scratch[0],0.)
                    forces[c]=f if j==1 else -f
                    sides[c]=mapping[geom]
                    t=contact.elem[1-j]
                    if t<0 and contact.vert[1-j]>=0:t=first[contact.vert[1-j]]
                    if t<0:continue
                    elements[c]=t
        self.forces.assign(forces);self.sides.assign(sides);self.elements.assign(elements)
        positions=np.zeros((self.capacity,3),dtype=np.float32)
        positions[:n]=d.contact.pos
        self._update(wp.array([n],dtype=int,device='cpu'),wp.zeros(self.capacity,dtype=int,device='cpu'),
            wp.array(positions,dtype=wp.vec3,device='cpu'),wp.array(d.geom_xpos[None].astype('f'),dtype=wp.vec3,device='cpu'),
            wp.array(d.flexvert_xpos[None].astype('f'),dtype=wp.vec3,device='cpu'),close,release,face_id,m.opt.timestep)

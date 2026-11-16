"""Closed six-face membrane: native flex contacts and per-world Warp energies.

The flex has no native elastic forces. All membrane and hinge energies are
specified here, so there is exactly one elastic load path through the envelope.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import warp as wp

from . import kernels as K

FACES = ('TOP','BOTTOM','LEFT','RIGHT','FRONT','BACK')


@dataclass
class Topology:
    vertices: np.ndarray
    triangles: np.ndarray
    face_id: np.ndarray
    edges: np.ndarray
    adjacent: np.ndarray
    seam: np.ndarray


def topology(subdivisions=None):
    if subdivisions is None:subdivisions=settings()['subdivisions']
    if subdivisions < 2:
        raise ValueError('At least 3x3 vertices per face are required')
    vertices=[];lookup={};triangles=[];face_ids=[]
    for face,(axis,sign) in enumerate(((2,1),(2,-1),(0,-1),(0,1),(1,-1),(1,1))):
        tangents=[a for a in range(3) if a!=axis]
        grid=np.zeros((subdivisions+1,subdivisions+1),dtype=int)
        for i in range(subdivisions+1):
            for j in range(subdivisions+1):
                key=[0,0,0];key[axis]=sign*subdivisions
                key[tangents[0]]=2*i-subdivisions;key[tangents[1]]=2*j-subdivisions
                key=tuple(key)
                if key not in lookup:
                    lookup[key]=len(vertices);vertices.append(np.array(key)/(2*subdivisions))
                grid[i,j]=lookup[key]
        for i in range(subdivisions):
            for j in range(subdivisions):
                for tri in ((grid[i,j],grid[i+1,j],grid[i+1,j+1]),(grid[i,j],grid[i+1,j+1],grid[i,j+1])):
                    tri=list(tri);a,b,c=np.array(vertices)[tri]
                    if np.cross(b-a,c-a)[axis]*sign<0:tri[1],tri[2]=tri[2],tri[1]
                    triangles.append(tri);face_ids.append(face)
    triangles=np.array(triangles,dtype=np.int32);face_ids=np.array(face_ids,dtype=np.int32)
    edges={}
    for t,tri in enumerate(triangles):
        for a,b in zip(tri,np.roll(tri,-1)):
            edges.setdefault(tuple(sorted((int(a),int(b)))),[]).append(t)
    if any(len(ts)!=2 for ts in edges.values()):raise ValueError('Envelope must be a closed manifold')
    adjacent=np.array(list(edges.values()),dtype=np.int32)
    return Topology(np.array(vertices),triangles,face_ids,np.array(list(edges),dtype=np.int32),adjacent,
                    (face_ids[adjacent[:,0]]!=face_ids[adjacent[:,1]]).astype(np.int32))


def settings():
    return json.loads(Path(__file__).with_name('film_config.json').read_text())


def add_xml(root, positions, topo, cfg):
    """Initialization only. Contact masks make the film the outer contact layer."""
    opt=root.find('option');opt.set('integrator','implicitfast');opt.set('solver','Newton');opt.set('jacobian','sparse');opt.set('iterations','100')
    for g in root.findall('.//geom'):
        name=g.get('name','')
        if name=='table':g.set('contype','1');g.set('conaffinity','2')
        elif name.startswith(('core','capgeom')):
            g.set('contype','8');g.set('conaffinity','10')
            size=np.fromstring(g.get('size'),sep=' ');size[0]/=.98
            g.set('size',' '.join(str(x) for x in size))
        elif name in ('left','right'):g.set('contype','4');g.set('conaffinity','3')
        g.set('priority','1')
        g.set('solref',f"{cfg['contact_time']} 1")
        g.set('solimp',f"{cfg['contact_impedance']} {cfg['contact_impedance']} .001")
    world=root.find('worldbody')
    flex=ET.SubElement(world,'flexcomp',name='film',type='direct',dim='2',
        point=' '.join(f'{x:.10g}' for x in positions.ravel()),
        element=' '.join(str(int(i)) for i in topo.triangles.ravel()),
        mass='.01',radius=str(cfg['contact_radius']),rgba='.5 .75 .65 .8')
    ET.SubElement(flex,'contact',condim='3',priority='0',friction='0 0 0',selfcollide='auto',
                  contype='2',conaffinity='15',solref='.004 1')
    return root


@wp.struct
class FilmParameters:
    thickness: float
    density: float
    stretch: float
    bend: float
    damping_ratio: float
    payload_mu: float
    finger_mu: float
    seam_ratio: float
    breaking_force: float
    bend_damping_ratio: float


@wp.kernel
def sample(seed: int, episode: int, bounds: wp.array(dtype=wp.vec2), out: wp.array(dtype=FilmParameters)):
    e=wp.tid()
    rng=wp.rand_init(seed+94007,e+episode*out.shape[0])
    p=FilmParameters()
    p.thickness=wp.randf(rng,bounds[0][0],bounds[0][1])
    p.density=wp.randf(rng,bounds[1][0],bounds[1][1])
    p.stretch=wp.randf(rng,bounds[2][0],bounds[2][1])
    p.bend=wp.randf(rng,bounds[3][0],bounds[3][1])
    p.damping_ratio=wp.randf(rng,bounds[4][0],bounds[4][1])
    p.payload_mu=wp.randf(rng,bounds[5][0],bounds[5][1])
    p.finger_mu=wp.randf(rng,bounds[6][0],bounds[6][1])
    p.seam_ratio=wp.randf(rng,bounds[7][0],bounds[7][1])
    p.breaking_force=wp.randf(rng,bounds[8][0],bounds[8][1])
    p.bend_damping_ratio=wp.randf(rng,bounds[9][0],bounds[9][1])
    out[e]=p


@wp.func
def triangle_area(a: wp.vec3,b: wp.vec3,c: wp.vec3):
    return .5*wp.length(wp.cross(b-a,c-a))


@wp.func
def hinge_angle(points: wp.array2d(dtype=wp.vec3), triangles: wp.array(dtype=wp.vec3i),
                adjacent: wp.array(dtype=wp.vec2i), edges: wp.array(dtype=wp.vec2i), e: int, h: int):
    tri0=triangles[adjacent[h][0]];tri1=triangles[adjacent[h][1]]
    a=points[e,tri0[0]];b=points[e,tri0[1]];c=points[e,tri0[2]]
    d=points[e,tri1[0]];f=points[e,tri1[1]];g=points[e,tri1[2]]
    n0=wp.cross(b-a,c-a);n0=n0/wp.sqrt(wp.dot(n0,n0)+1.e-20)
    n1=wp.cross(f-d,g-d);n1=n1/wp.sqrt(wp.dot(n1,n1)+1.e-20)
    edge=points[e,edges[h][1]]-points[e,edges[h][0]]
    edge=edge/wp.sqrt(wp.dot(edge,edge)+1.e-20)
    return wp.atan2(wp.dot(edge,wp.cross(n0,n1)),wp.dot(n0,n1))


@wp.kernel
def configure(payload: wp.array(dtype=K.Parameters), film: wp.array(dtype=FilmParameters),
              unit: wp.array(dtype=wp.vec3), triangles: wp.array(dtype=wp.vec3i),
              bodies: wp.array(dtype=int), offset: wp.vec3, radius: float, patch: float, inner_max: float,
              body_pos: wp.array2d(dtype=wp.vec3), body_mass: wp.array2d(dtype=float),
              rest: wp.array2d(dtype=wp.vec3), masses: wp.array2d(dtype=float),
              available: wp.array2d(dtype=float), geom_ids: wp.array(dtype=int),
              table: int, fingers: wp.array(dtype=int), friction: wp.array2d(dtype=wp.vec3)):
    e=wp.tid()
    p=payload[e];f=film[e]
    dims=wp.vec3(p.size[0]*(1.0+p.slack),p.size[1]*(1.0+p.slack_y),
                 p.size[2]*(1.0+.5*(p.slack+p.slack_y))+.004)
    center=offset+wp.vec3(0.0,0.0,dims[2]*.5+radius+.0002)
    for v in range(unit.shape[0]):
        local=wp.cw_mul(unit[v],dims)
        boundary_count=int(0)
        for axis in range(3):
            if wp.abs(unit[v][axis])>.499:boundary_count+=1
        # Keep every shared boundary unchanged; redistribute only face-interior nodes.
        if boundary_count==1:
            for axis in range(3):
                if wp.abs(unit[v][axis])<.499:
                    local[axis]=unit[v][axis]/wp.max(inner_max,1.e-6)*wp.min(dims[axis]*inner_max,patch)
        point=local+center
        rest[e,v]=point
        body_pos[e,bodies[v]]=point
    # Lump mass and finite material surplus from reference triangle areas.
    for v in range(unit.shape[0]):
        area=float(0.0);surplus=float(0.0)
        for t in range(triangles.shape[0]):
            ids=triangles[t]
            if ids[0]==v or ids[1]==v or ids[2]==v:
                a=rest[e,ids[0]]-center;b=rest[e,ids[1]]-center;c=rest[e,ids[2]]-center
                A=triangle_area(a,b,c)
                aa=wp.cw_mul(wp.cw_div(a,dims),p.size)
                bb=wp.cw_mul(wp.cw_div(b,dims),p.size)
                cc=wp.cw_mul(wp.cw_div(c,dims),p.size)
                area+=A/3.0
                surplus+=wp.max(A-triangle_area(aa,bb,cc),0.0)/(3.0*wp.sqrt(A))
        mass=area*f.thickness*f.density
        body_mass[e,bodies[v]]=mass
        masses[e,v]=mass
        available[e,v]=surplus
    for j in range(geom_ids.shape[0]):
        friction[e,geom_ids[j]]=wp.vec3(f.payload_mu,.005,.0001)
    friction[e,table]=wp.vec3(p.table_mu,.005,.0001)
    for j in range(fingers.shape[0]):
        friction[e,fingers[j]]=wp.vec3(f.finger_mu,.005,.0001)


@wp.kernel
def reference(rest: wp.array2d(dtype=wp.vec3), edges: wp.array(dtype=wp.vec2i),
              triangles: wp.array(dtype=wp.vec3i), adjacent: wp.array(dtype=wp.vec2i),
              lengths: wp.array2d(dtype=float), angles: wp.array2d(dtype=float)):
    e,h=wp.tid()
    lengths[e,h]=wp.length(rest[e,edges[h][1]]-rest[e,edges[h][0]])
    angles[e,h]=hinge_angle(rest,triangles,adjacent,edges,e,h)


@wp.kernel
def stretch_energy(points: wp.array2d(dtype=wp.vec3), edges: wp.array(dtype=wp.vec2i),
                   seam: wp.array(dtype=int), parameters: wp.array(dtype=FilmParameters),
                   lengths: wp.array2d(dtype=float), damage: wp.array2d(dtype=int),
                   energy: wp.array(dtype=float)):
    e,h=wp.tid()
    if damage[e,h]==0:
        k=parameters[e].stretch
        if seam[h]==1:k*=parameters[e].seam_ratio
        delta=wp.length(points[e,edges[h][1]]-points[e,edges[h][0]])-lengths[e,h]
        wp.atomic_add(energy,e,.5*k*delta*delta)


@wp.kernel
def bending_energy(points: wp.array2d(dtype=wp.vec3), edges: wp.array(dtype=wp.vec2i),
                   triangles: wp.array(dtype=wp.vec3i), adjacent: wp.array(dtype=wp.vec2i),
                   parameters: wp.array(dtype=FilmParameters), angles: wp.array2d(dtype=float),
                   damage: wp.array2d(dtype=int), energy: wp.array(dtype=float)):
    e,h=wp.tid()
    if damage[e,h]==0:
        theta=hinge_angle(points,triangles,adjacent,edges,e,h)-angles[e,h]
        delta=wp.atan2(wp.sin(theta),wp.cos(theta))
        wp.atomic_add(energy,e,.5*parameters[e].bend*delta*delta)


@wp.kernel
def dissipative_forces(points: wp.array2d(dtype=wp.vec3), velocities: wp.array2d(dtype=wp.vec3),
                       edges: wp.array(dtype=wp.vec2i), seam: wp.array(dtype=int),
                       parameters: wp.array(dtype=FilmParameters), masses: wp.array2d(dtype=float),
                       damage: wp.array2d(dtype=int), damping: wp.array2d(dtype=wp.vec3),
                       power: wp.array(dtype=float)):
    e,h=wp.tid()
    if damage[e,h]==0:
        a=edges[h][0];b=edges[h][1]
        delta=points[e,b]-points[e,a]
        direction=delta/wp.max(wp.length(delta),1.e-10)
        speed=wp.dot(velocities[e,b]-velocities[e,a],direction)
        m=masses[e,a]*masses[e,b]/(masses[e,a]+masses[e,b])
        k=parameters[e].stretch
        if seam[h]==1:k*=parameters[e].seam_ratio
        c=2.0*parameters[e].damping_ratio*wp.sqrt(k*m)
        f=c*speed*direction
        wp.atomic_add(damping,e,a,f);wp.atomic_sub(damping,e,b,f)
        wp.atomic_add(power,e,-c*speed*speed)


@wp.kernel
def positions_and_velocities(xpos: wp.array2d(dtype=wp.vec3), qvel: wp.array2d(dtype=float),
                            bodies: wp.array(dtype=int), dofs: wp.array(dtype=int),
                            points: wp.array2d(dtype=wp.vec3), velocities: wp.array2d(dtype=wp.vec3)):
    e,v=wp.tid()
    points[e,v]=xpos[e,bodies[v]]
    d=dofs[v]
    velocities[e,v]=wp.vec3(qvel[e,d],qvel[e,d+1],qvel[e,d+2])


@wp.kernel
def write_forces(gradient: wp.array2d(dtype=wp.vec3), damping: wp.array2d(dtype=wp.vec3),
                 bodies: wp.array(dtype=int), target: wp.array2d(dtype=wp.spatial_vector)):
    e,v=wp.tid()
    target[e,bodies[v]]+=wp.spatial_vector(-gradient[e,v]+damping[e,v],wp.vec3(0.0))


@wp.kernel
def fracture(points: wp.array2d(dtype=wp.vec3), edges: wp.array(dtype=wp.vec2i),
             seam: wp.array(dtype=int), params: wp.array(dtype=FilmParameters),
             lengths: wp.array2d(dtype=float), angles: wp.array2d(dtype=float),
             triangles: wp.array(dtype=wp.vec3i), adjacent: wp.array(dtype=wp.vec2i),
             damage: wp.array2d(dtype=int), loss: wp.array(dtype=float)):
    e,h=wp.tid()
    if damage[e,h]==0:
        k=params[e].stretch
        if seam[h]==1:k*=params[e].seam_ratio
        extension=wp.max(wp.length(points[e,edges[h][1]]-points[e,edges[h][0]])-lengths[e,h],0.0)
        limit=params[e].breaking_force
        if seam[h]==1:limit*=params[e].seam_ratio
        if k*extension>limit:
            damage[e,h]=1
            theta=hinge_angle(points,triangles,adjacent,edges,e,h)-angles[e,h]
            delta=wp.atan2(wp.sin(theta),wp.cos(theta))
            wp.atomic_add(loss,e,.5*k*extension*extension+.5*params[e].bend*delta*delta)


class Surface:
    """Shared CPU/GPU elastic workspace. Native MuJoCo owns all node integration."""
    def __init__(self, m, nworld=1, device='cuda:0'):
        self.device=device;self.nworld=nworld;self.cfg=settings();self.topo=topology(self.cfg['subdivisions'])
        t=self.topo
        with wp.ScopedDevice(device):
            self.unit=wp.array(t.vertices,dtype=wp.vec3)
            self.triangles=wp.array(t.triangles,dtype=wp.vec3i)
            self.edges=wp.array(t.edges,dtype=wp.vec2i)
            self.adjacent=wp.array(t.adjacent,dtype=wp.vec2i)
            self.seam=wp.array(t.seam,dtype=int)
            self.bodies=wp.array([m.body(f'film_{v}').id for v in range(len(t.vertices))],dtype=int)
            self.dofs=wp.array([m.body_dofadr[m.body(f'film_{v}').id] for v in range(len(t.vertices))],dtype=int)
            names=('thickness','density','stretch_stiffness','bending_stiffness','damping_ratio',
                   'payload_friction','finger_friction','seam_stiffness_ratio','breaking_force','bending_damping_ratio')
            self.bounds=wp.array([self.cfg[k] for k in names],dtype=wp.vec2)
            self.parameters=wp.empty(nworld,dtype=FilmParameters)
            shape=(nworld,len(t.vertices));edge_shape=(nworld,len(t.edges))
            self.rest=wp.empty(shape,dtype=wp.vec3)
            self.points=wp.zeros(shape,dtype=wp.vec3,requires_grad=True)
            self.velocities=wp.zeros(shape,dtype=wp.vec3,requires_grad=True)
            self.damping_points=wp.empty(shape,dtype=wp.vec3)
            self.rayleigh=wp.zeros(nworld,dtype=float,requires_grad=True)
            self.masses=wp.empty(shape,dtype=float)
            self.available=wp.zeros(shape,dtype=float)
            self.lengths=wp.empty(edge_shape,dtype=float)
            self.angles=wp.empty(edge_shape,dtype=float)
            self.damage=wp.zeros(edge_shape,dtype=int)
            self.fracture_loss=wp.zeros(nworld,dtype=float)
            self.energy=wp.zeros(nworld,dtype=float,requires_grad=True)
            self.ones=wp.ones(nworld,dtype=float)
            self.damping=wp.zeros(shape,dtype=wp.vec3)
            self.power=wp.zeros(nworld,dtype=float)

    def initialize(self, payload, body_pos, body_mass, friction, core_geoms, table, finger_geoms,
                   seed=7, episode=0, offset=(0.,0.,0.), armature=None):
        with wp.ScopedDevice(self.device):
            wp.launch(sample,self.nworld,[seed,episode,self.bounds,self.parameters])
            wp.launch(configure,self.nworld,[payload,self.parameters,self.unit,self.triangles,self.bodies,
                wp.vec3(*offset),self.cfg['contact_radius'],self.cfg['local_patch_half_width'],
                .5-1.0/self.cfg['subdivisions'],body_pos,body_mass,self.rest,self.masses,self.available,
                core_geoms,table,finger_geoms,friction])
            wp.launch(reference,self.lengths.shape,[self.rest,self.edges,self.triangles,self.adjacent,self.lengths,self.angles])
            self.damage.zero_();self.fracture_loss.zero_()
            if armature is not None:
                wp.launch(clear_node_armature,self.masses.shape,[self.dofs,armature])

    def elastic_forces(self):
        self.energy.zero_();self.points.grad.zero_();self.energy.grad.zero_()
        with wp.Tape() as tape:
            wp.launch(stretch_energy,self.lengths.shape,[self.points,self.edges,self.seam,self.parameters,
                self.lengths,self.damage,self.energy])
            wp.launch(bending_energy,self.lengths.shape,[self.points,self.edges,self.triangles,self.adjacent,
                self.parameters,self.angles,self.damage,self.energy])
        tape.backward(grads={self.energy:self.ones})

    def bending_damping(self):
        wp.copy(self.damping_points,self.points)
        self.velocities.grad.zero_();self.rayleigh.zero_();self.rayleigh.grad.zero_()
        with wp.Tape() as tape:
            wp.launch(bending_dissipation,self.lengths.shape,[self.damping_points,self.velocities,
                self.triangles,self.adjacent,self.edges,self.parameters,self.masses,self.damage,self.rayleigh])
        tape.backward(grads={self.rayleigh:self.ones})
        wp.launch(merge_bending_damping,self.points.shape,[self.velocities.grad,self.rayleigh,self.damping,self.power])

    def apply(self, xpos, qvel, target):
        with wp.ScopedDevice(self.device):
            wp.launch(positions_and_velocities,self.points.shape,[xpos,qvel,self.bodies,self.dofs,self.points,self.velocities])
            wp.launch(fracture,self.lengths.shape,[self.points,self.edges,self.seam,self.parameters,
                self.lengths,self.angles,self.triangles,self.adjacent,self.damage,self.fracture_loss])
            self.elastic_forces()
            self.damping.zero_();self.power.zero_()
            wp.launch(dissipative_forces,self.lengths.shape,[self.points,self.velocities,self.edges,self.seam,
                self.parameters,self.masses,self.damage,self.damping,self.power])
            self.bending_damping()
            wp.launch(write_forces,self.points.shape,[self.points.grad,self.damping,self.bodies,target])


@wp.kernel
def close_payload_gaps(payload: wp.array(dtype=K.Parameters), geoms: wp.array(dtype=int),
                       segments: int, sizes: wp.array2d(dtype=wp.vec3)):
    e,j=wp.tid()
    value=sizes[e,geoms[j]]
    value[0]=payload[e].size[0]/(2.0*float(segments))
    sizes[e,geoms[j]]=value


@wp.kernel
def clear_node_armature(dofs: wp.array(dtype=int), armature: wp.array2d(dtype=float)):
    e,v=wp.tid()
    for i in range(3):armature[e,dofs[v]+i]=0.0


@wp.func
def unit_and_rate(vector: wp.vec3,rate: wp.vec3):
    length=wp.sqrt(wp.dot(vector,vector)+1.e-20)
    unit=vector/length
    derivative=(rate-unit*wp.dot(unit,rate))/length
    return unit,derivative


@wp.func
def hinge_rate(points: wp.array2d(dtype=wp.vec3),velocities: wp.array2d(dtype=wp.vec3),
               triangles: wp.array(dtype=wp.vec3i),adjacent: wp.array(dtype=wp.vec2i),
               edges: wp.array(dtype=wp.vec2i),e: int,h: int):
    i=triangles[adjacent[h][0]];j=triangles[adjacent[h][1]]
    a=points[e,i[0]];b=points[e,i[1]];c=points[e,i[2]]
    va=velocities[e,i[0]];vb=velocities[e,i[1]];vc=velocities[e,i[2]]
    d=points[e,j[0]];f=points[e,j[1]];g=points[e,j[2]]
    vd=velocities[e,j[0]];vf=velocities[e,j[1]];vg=velocities[e,j[2]]
    n0,dn0=unit_and_rate(wp.cross(b-a,c-a),wp.cross(vb-va,c-a)+wp.cross(b-a,vc-va))
    n1,dn1=unit_and_rate(wp.cross(f-d,g-d),wp.cross(vf-vd,g-d)+wp.cross(f-d,vg-vd))
    v0=edges[h][0];v1=edges[h][1]
    tangent,dtangent=unit_and_rate(points[e,v1]-points[e,v0],velocities[e,v1]-velocities[e,v0])
    sine=wp.dot(tangent,wp.cross(n0,n1));cosine=wp.dot(n0,n1)
    dsine=wp.dot(dtangent,wp.cross(n0,n1))+wp.dot(tangent,wp.cross(dn0,n1)+wp.cross(n0,dn1))
    dcosine=wp.dot(dn0,n1)+wp.dot(n0,dn1)
    return (cosine*dsine-sine*dcosine)/wp.max(sine*sine+cosine*cosine,1.e-12)


@wp.kernel
def bending_dissipation(points: wp.array2d(dtype=wp.vec3),velocities: wp.array2d(dtype=wp.vec3),
                        triangles: wp.array(dtype=wp.vec3i),adjacent: wp.array(dtype=wp.vec2i),
                        edges: wp.array(dtype=wp.vec2i),parameters: wp.array(dtype=FilmParameters),
                        masses: wp.array2d(dtype=float),
                        damage: wp.array2d(dtype=int),rayleigh: wp.array(dtype=float)):
    e,h=wp.tid()
    if damage[e,h]==0:
        ids=hinge_vertices(triangles,adjacent,edges,h)
        a=ids[0];b=ids[1];c=ids[2];d=ids[3]
        qa,qb,qc,qd=dihedral_gradients(points[e,a],points[e,b],points[e,c],points[e,d])
        inverse_inertia=(wp.dot(qa,qa)/masses[e,a]+wp.dot(qb,qb)/masses[e,b]
                         +wp.dot(qc,qc)/masses[e,c]+wp.dot(qd,qd)/masses[e,d])
        # The angular mode involves all four vertices. Its inertia depends on
        # triangle altitudes, including the strongly nonuniform focused mesh.
        inertia=1.0/wp.max(inverse_inertia,1.e-12)
        coefficient=2.0*parameters[e].bend_damping_ratio*wp.sqrt(parameters[e].bend*inertia)
        rate=(wp.dot(qa,velocities[e,a])+wp.dot(qb,velocities[e,b])
              +wp.dot(qc,velocities[e,c])+wp.dot(qd,velocities[e,d]))
        wp.atomic_add(rayleigh,e,.5*coefficient*rate*rate)


@wp.func
def hinge_vertices(triangles: wp.array(dtype=wp.vec3i),adjacent: wp.array(dtype=wp.vec2i),
                   edges: wp.array(dtype=wp.vec2i),h: int):
    a=edges[h][0];b=edges[h][1]
    c=int(-1);d=int(-1)
    for i in range(3):
        v=triangles[adjacent[h][0]][i]
        if v!=a and v!=b:c=v
        v=triangles[adjacent[h][1]][i]
        if v!=a and v!=b:d=v
    return wp.vec4i(a,b,c,d)


@wp.func
def dihedral_gradients(a: wp.vec3,b: wp.vec3,c: wp.vec3,d: wp.vec3):
    edge=b-a
    squared=wp.dot(edge,edge)+1.e-20
    length=wp.sqrt(squared)
    n0=wp.cross(edge,c-a);n1=wp.cross(d-a,edge)
    qc=-length*n0/(wp.dot(n0,n0)+1.e-20)
    qd=-length*n1/(wp.dot(n1,n1)+1.e-20)
    tc=wp.dot(c-a,edge)/squared;td=wp.dot(d-a,edge)/squared
    qa=-(1.0-tc)*qc-(1.0-td)*qd
    qb=-tc*qc-td*qd
    return qa,qb,qc,qd


@wp.kernel
def merge_bending_damping(velocity_gradient: wp.array2d(dtype=wp.vec3),rayleigh: wp.array(dtype=float),
                          damping: wp.array2d(dtype=wp.vec3),power: wp.array(dtype=float)):
    e,v=wp.tid()
    damping[e,v]-=velocity_gradient[e,v]
    if v==0:power[e]-=2.0*rayleigh[e]

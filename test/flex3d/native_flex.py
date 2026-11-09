"""Native dim=3 flex adapter for the reusable YAM grasp platform."""
from dataclasses import dataclass
from types import SimpleNamespace
from itertools import permutations
from xml.etree import ElementTree as ET
import mujoco
import numpy as np
from grasp_platform import MuJoCoSim,TrialConfig,attach_robot,vec,tips,measure_contacts
from pinch import PinchAttachment

@dataclass
class Config(TrialConfig):
    young: float=10000.
    poisson: float=.2
    damping: float=.02
    surface_refinement: int=1
    shape: str='box'

class FlexSim(MuJoCoSim):
    def __init__(self,built):
        super().__init__(built)
        self.tets = self.m.flex_elem.reshape(-1, 4).copy()
        p = self.d.flexvert_xpos[self.tets]
        self.det0 = np.einsum("ij,ij->i", p[:, 1]-p[:, 0], np.cross(p[:, 2]-p[:, 0], p[:, 3]-p[:, 0]))
        self.bodies = self.m.flex_vertbodyid.copy()
        self.mass = self.m.body_mass[self.bodies]

    def step(self):
        mujoco.mj_step(self.m, self.d)

    def com(self):
        return np.average(self.d.flexvert_xpos, axis=0, weights=self.mass)

    def determinant(self):
        p = self.d.flexvert_xpos[self.tets]
        return np.einsum("ij,ij->i", p[:, 1]-p[:, 0], np.cross(p[:, 2]-p[:, 0], p[:, 3]-p[:, 0])) / self.det0



class FlexPinch(PinchAttachment):
    """v1.3 release law on surface vertices, reacting on the articulated palm.

    Node connections bind at current positions, without teleporting material.
    The frame is an actual arm body, so its reactions load the robot actuators.
    """
    def __init__(self, robot, cfg):
        self.sim = robot.sim
        self.coefficient = cfg.pinch_coeff
        self.frame = self.sim.m.body("pinch_frame").id
        self.top = list(self.sim.built.top_vertices)
        self.equations = {i: self.sim.m.equality(f"pinch_{i}").id for i in self.top}
        self.active = None
        self.segment = None
        self.segments = []
        self.broken = False
        self.load = 0.
        self.events = []
        self.peak_force = 0.
        self.robot = robot

    def try_bind(self, tips):
        if self.coefficient == 0 or self.active is not None or self.broken:
            return False
        sim, r = self.sim, self.robot
        forces = measure(r)
        # The rule needs real top contacts on both fingers and >=3mm insertion.
        if not all(forces["top_contact_each"]):
            return False
        top_z = float(sim.built.rest_top)
        if np.any(top_z - np.asarray(tips)[:, 2] < .003):
            return False
        mid = np.mean(tips, axis=0)
        axis = tips[1] - tips[0]
        axis[2] = 0
        length = np.linalg.norm(axis)
        if length < 1e-8:
            return False
        axis /= length
        across = np.array([-axis[1], axis[0], 0.])
        delta = sim.d.flexvert_xpos[self.top] - mid
        inside = (np.abs(delta @ axis) <= length / 2 + .001) & (np.abs(delta @ across) <= .010)
        candidates = np.array(self.top)[inside].tolist()
        if not candidates:
            return False
        m, d = sim.m, sim.d
        for i in candidates:
            eq = self.equations[i]
            m.eq_data[eq, :3] = d.xmat[self.frame].reshape(3, 3).T @ (d.flexvert_xpos[i] - d.xpos[self.frame])
            m.eq_data[eq, 3:6] = 0.
            d.eq_active[eq] = True
        self.active = [self.equations[i] for i in candidates]
        self.segment, self.segments = candidates[0], candidates
        self.events.append(dict(t=float(d.time), event="bind", vertices=candidates))
        return True

    def check_load(self, dt, grip_force, force_vector=None):
        force = self.force_vector() if force_vector is None else force_vector
        # Equality force_vector is the force on body1 (palm). Downward load on
        # the palm means pulling material away from support; upward is pressing.
        load_vector = np.array([force[0], force[1], min(float(force[2]), 0.)])
        return super().check_load(dt, grip_force, load_vector)



def scene(cfg):
    size, mass, count = ((.24, .16, .05), .29, (7, 5, 4)) if cfg.category == "bendable" else ((.35, .25, .12), .5, (9, 7, 5))
    if cfg.surface_refinement not in (1,2):raise ValueError('surface_refinement must be 1 or 2')
    count=((count[0]-1)*cfg.surface_refinement+1,(count[1]-1)*cfg.surface_refinement+1,count[2])
    center = np.array([-.25, .04])
    top = size[2] + .002
    top_ids = [i for i in range(int(np.prod(count))) if i % count[2] == count[2] - 1]
    eq = ''.join(f'<connect name="pinch_{i}" body1="pinch_frame" body2="contents_{i}" anchor="0 0 0" '
                 'active="false" solref=".005 1" solimp=".99 .99 .001"/>' for i in top_ids)
    xml = f'''<mujoco><option timestep="{cfg.dt}" integrator="discrete" solver="Newton" iterations="100"
      tolerance="1e-8" ls_iterations="50" gravity="0 0 -9.81"/><size memory="128M"/>
      <worldbody><light pos="0 0 2"/>
        <geom name="table" type="box" pos="0 0 -.025" size="1 .22 .025" contype="1" conaffinity="6" friction=".8 .005 .0001"
          solref="{cfg.contact_time} 1" solimp=".99 .99 .001" rgba=".7 .7 .7 1"/>
        <flexcomp name="contents" type="grid" dim="3" dof="full" count="{' '.join(map(str, count))}"
          spacing="{vec(np.array(size)/(np.array(count)-1))}" pos="{vec([*center, .002+size[2]/2])}" mass="{mass}" radius=".001" rgba=".15 .65 .72 1">
          <elasticity young="{cfg.young}" poisson="{cfg.poisson}" damping="{cfg.damping}"/>
          <contact selfcollide="none" contype="2" conaffinity="5" condim="3" friction=".8 .005 .0001"
            solref="{cfg.contact_time} 1" solimp=".99 .99 .001"/>
        </flexcomp>
      </worldbody><contact/><equality>{eq}</equality></mujoco>'''
    if cfg.shape=='pillow':
        # A rounded, biconvex pillow, filled with tetrahedra throughout.
        # This changes only the reference mesh, not the constitutive law.
        points=[]
        for u in np.linspace(-1,1,count[0]):
            for v in np.linspace(-1,1,count[1]):
                norm=(abs(u)**4+abs(v)**4)**.25
                scale=max(abs(u),abs(v))/norm if norm else 1.
                half_height=.003+(size[2]/2-.003)*np.sqrt(max(0.,(1-u*u)*(1-v*v)))
                for w in np.linspace(-1,1,count[2]):
                    points.append([size[0]/2*u*scale,size[1]/2*v*scale,w*half_height])
        points=np.asarray(points)
        def index(q):return (q[0]*count[1]+q[1])*count[2]+q[2]
        elements=[]
        for i in range(count[0]-1):
            for j in range(count[1]-1):
                for k in range(count[2]-1):
                    origin=np.array([i,j,k])
                    for order in permutations(range(3)):
                        corners=[origin.copy()]
                        for axis in order:
                            q=corners[-1].copy();q[axis]+=1;corners.append(q)
                        ids=[index(q) for q in corners]
                        p=points[ids]
                        if np.linalg.det(np.stack([p[n]-p[0] for n in (1,2,3)]))<0:ids[1],ids[2]=ids[2],ids[1]
                        elements.append(ids)
        root=ET.fromstring(xml);flex=root.find('.//flexcomp');flex.set('type','direct')
        flex.attrib.pop('count');flex.attrib.pop('spacing')
        flex.set('point',vec(points));flex.set('element',' '.join(str(i) for t in elements for i in t))
        xml=ET.tostring(root,encoding='unicode')
    elif cfg.shape!='box':raise ValueError('shape must be box or pillow')
    built = SimpleNamespace(xml=xml, desc=SimpleNamespace(), cfg=cfg, top_vertices=top_ids, rest_top=top)
    return attach_robot(cfg,built,FlexSim,center,top,size,mass)



def measure(robot):
    return measure_contacts(robot,target_flexes={0})


def attachment(robot,cfg):
    return FlexPinch(robot,cfg)

def check_state(sim):
    value=float(sim.determinant().min())
    return ('tetrahedron inversion' if value<=0 else None),dict(min_determinant=value)

def summarize(trace):
    return dict(min_determinant=min((x['min_determinant'] for x in trace),default=None))

def dimensions(sim):
    return dict(nv=sim.m.nv,vertices=sim.m.nflexvert,tetrahedra=sim.m.nflexelem)

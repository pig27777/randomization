"""Rigid-model plumbing control; not a flex physics comparison."""
from types import SimpleNamespace
import numpy as np
from .protocol import MuJoCoSim,TrialConfig,attach_robot,vec,measure_contacts
Config=TrialConfig

class RigidSim(MuJoCoSim):
    def com(self):return self.d.xipos[self.m.body('content').id].copy()

class NoPinch:
    active=None
    load=0.
    events=()
    def try_bind(self,tips):return False
    def check_load(self,*args):return False
    def release(self,*args):pass

def scene(cfg):
    size,mass=((.24,.16,.05),.29) if cfg.category=='bendable' else ((.35,.25,.12),.5)
    center=np.array([-.25,.04]);top=size[2]+.002
    xml=f'''<mujoco><option timestep="{cfg.dt}" integrator="discrete" gravity="0 0 -9.81"/>
      <worldbody><geom name="table" type="box" pos="0 0 -.025" size="1 .22 .025"/>
      <body name="content" pos="{vec([*center,.002+size[2]/2])}"><freejoint/>
        <inertial pos="0 0 0" mass="{mass}" diaginertia="{vec(mass/12*(np.sum(np.array(size)**2)-np.array(size)**2))}"/>
        <geom name="content" type="box" size="{vec(np.array(size)/2)}" rgba=".8 .5 .2 1"/>
      </body></worldbody><contact/><equality/></mujoco>'''
    return attach_robot(cfg,SimpleNamespace(xml=xml,desc=SimpleNamespace(),cfg=cfg),RigidSim,center,top,size,mass)

def measure(robot):return measure_contacts(robot,target_geoms={robot.sim.m.geom('content').id})
def attachment(robot,cfg):return NoPinch()
def check_state(sim):return None,{}
def summarize(trace):return {}
def dimensions(sim):return dict(nv=sim.m.nv,rigid_bodies=1)

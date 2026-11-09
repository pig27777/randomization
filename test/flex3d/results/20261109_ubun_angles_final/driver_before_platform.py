"""Full YAM arm grasping native 3D flex from multiple orientations.

CPU behavior evidence only; no package installation and no writes to other lines.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bag_sim.mj_loose import fingertip_xml
from bag_sim.frozen_content import activate, RELEASE
RELEASE_SHA = activate()
sys.path.insert(0, str(ROOT / "policy"))
from yam_probe_robot import Yam, asset_fingerprint
from model import ANCHORS
from pinch import PinchAttachment, PinchSpring, LOAD_FILTER_TAU


def vec(x):
    return " ".join(format(float(v), ".17g") for v in np.asarray(x).ravel())


@dataclass
class Config:
    category: str = "bendable"
    yaw: float = 0.
    tilt: float = 0.
    young: float = 10000.
    poisson: float = .2
    damping: float = .02
    dt: float = .001
    pinch_coeff: float = 1.
    contact_time: float = .005
    wall_limit: float = 1800.
    smoke_seconds: float = 0.
    arm_height: float = .27


class FlexSim:
    def __init__(self, built):
        self.built = built
        root = ET.fromstring(built.xml)
        root.find(".//body[@name='arm']").set("pos", f"0 -.30 {built.cfg.arm_height}")
        pedestal = root.find(".//geom[@name='yam_pedestal']")
        pedestal.set("pos", f"0 -.30 {built.cfg.arm_height/2}")
        pedestal.set("size", f".06 .06 {built.cfg.arm_height/2}")
        # Preserve the full arm, rails, inertias and actuator limits. Replace only
        # the two pad contact faces by the task's 20x40 mm, 2 mm rounded geometry.
        for side in ("left", "right"):
            pad = root.find(f".//body[@name='{side}_pad']")
            old = pad.find("geom")
            center = np.fromstring(old.get("pos"), sep=" ")
            # Put the inner face at the upstream contact plane, and keep the
            # distal edge at the original tip edge (z=-144.4 mm).
            center[1], center[2] = -.004, -.1244
            pad.remove(old)
            geometry = ET.fromstring(fingertip_xml(side, 1, (0, 0, 0)))
            for i, geom in enumerate(geometry.findall("geom")):
                g = deepcopy(geom)
                g.set("name", f"{side}_pad" if i == 0 else f"{side}_round_{i}")
                g.set("pos", vec(np.fromstring(g.get("pos", "0 0 0"), sep=" ") + center))
                if g.get("fromto"):
                    ends = np.fromstring(g.get("fromto"), sep=" ").reshape(2, 3) + center
                    g.set("fromto", vec(ends))
                    g.attrib.pop("pos")
                g.set("contype", "4")
                g.set("conaffinity", "3")
                g.set("priority", "1")
                g.set("condim", "3")
                g.set("friction", ".8 .005 .0001")
                g.set("solref", f"{built.cfg.contact_time} 1")
                g.set("solimp", ".99 .99 .001")
                pad.append(g)
        link = root.find(".//body[@name='link_6']")
        ET.SubElement(link, "body", name="pinch_frame", pos="0 0 .1347")
        built.xml = ET.tostring(root, encoding="unicode")
        self.m = mujoco.MjModel.from_xml_string(built.xml)
        self.d = mujoco.MjData(self.m)
        self.desc = built.desc
        self.obj_geoms = set()
        self.table_geom = self.m.geom("table").id
        mujoco.mj_forward(self.m, self.d)
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
    yaw, tilt = np.deg2rad([cfg.yaw, cfg.tilt])
    rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    rx = np.array([[1,0,0],[0,np.cos(tilt),-np.sin(tilt)],[0,np.sin(tilt),np.cos(tilt)]])
    rotation = rz @ rx @ np.diag([1., -1., -1.])
    target = np.array([*center, top + .10])
    built = SimpleNamespace(xml=xml, desc=SimpleNamespace(), cfg=cfg, top_vertices=top_ids, rest_top=top)
    robot = Yam(ANCHORS[cfg.category], 0, fixture=dict(built=built, target=target, rotation=rotation, sim_factory=FlexSim))
    m,d = robot.sim.m,robot.sim.d
    robot.control_period_steps = max(1, round(.01 / cfg.dt))
    robot.pad_sets = [{g for g in range(m.ngeom) if m.geom(g).name == f"{s}_pad" or m.geom(g).name.startswith(f"{s}_round_")} for s in ("left", "right")]
    robot.all_finger_sets = [set(robot.sim.pad_geoms[s]) | robot.pad_sets[j] for j,s in enumerate(("left", "right"))]
    robot.pad_geom = [m.geom(f"{s}_pad").id for s in ("left", "right")]
    robot.nonfinger_mask &= ~np.isin(np.arange(m.ngeom), list(set.union(*robot.all_finger_sets)))
    robot.size, robot.mass = size,mass
    return robot


def tips(robot):
    m,d = robot.sim.m,robot.sim.d
    # Local pad z points toward the palm, so the distal edge lies at -20mm.
    return np.array([d.geom_xpos[g] - d.geom_xmat[g].reshape(3,3)[:,2]*.020 for g in robot.pad_geom])


def measure(robot):
    sim = robot.sim
    m,d = sim.m,sim.d
    result = np.zeros((2,3))
    top_contacts = [False,False]
    support = 0.
    penetration = 0.
    nonfinger = 0.
    buffer = np.zeros(6)
    for i,c in enumerate(d.contact):
        if c.efc_address < 0:
            continue
        gs = list(c.geom)
        if 0 not in c.flex:
            continue
        mujoco.mj_contactForce(m,d,i,buffer)
        world = c.frame.reshape(3,3).T @ buffer[:3]
        penetration = max(penetration, -float(c.dist)*1000)
        for j, group in enumerate(robot.all_finger_sets):
            for k in (0,1):
                if gs[k] in group:
                    result[j] += (1 if k==1 else -1)*world
                    # Top-facing contact normal pushing the finger upward.
                    top_contacts[j] |= bool((1 if k==1 else -1)*c.frame[2] > .5 and buffer[0] > .05)
        if sim.table_geom in gs:
            support += (1 if gs[0] == sim.table_geom else -1) * world[2]
        for g in gs:
            if g >= 0 and robot.nonfinger_mask[g]:
                nonfinger = max(nonfinger,-float(c.dist)*1000)
    axis = d.geom_xpos[robot.pad_geom[1]]-d.geom_xpos[robot.pad_geom[0]]
    axis /= max(np.linalg.norm(axis),1e-12)
    grip = [max(0.,float(-result[0]@axis)),max(0.,float(result[1]@axis))]
    return dict(grip_each_N=grip, force_on_fingers_N=result.tolist(), support_N=float(support),
                top_contact_each=top_contacts, penetration_mm=penetration, nonfinger_mm=nonfinger)


def run(cfg, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    (output/"config.json").write_text(json.dumps(asdict(cfg),indent=2))
    robot = scene(cfg)
    sim = robot.sim
    m,d = sim.m,sim.d
    (output/"model.xml").write_text(sim.built.xml)
    (output/"robot_assets.json").write_text(json.dumps(asset_fingerprint(),indent=2))
    pinch = FlexPinch(robot,cfg)
    spring = PinchSpring()
    dt = cfg.dt
    gap = .040
    filtered = np.zeros(2)
    grip20 = 0.
    phase = "settle"
    phase_start = 0.
    start = robot.initial_target.copy()
    target = start.copy()
    depth = 0.
    contacted = False
    baseline_com = sim.com()
    lift_site = None
    lift_object = None
    phases=[]
    trace=[]
    states=[]
    outcome="pending"
    reason=""
    peaks=dict(penetration_mm=0.,nonfinger_mm=0.,grip_N=0.,down_N=0.,tracking_mm=0.)
    min_det=1.
    loss=0.
    stable=0.
    sample_every=max(1,round(.02/dt))
    for step in range(round(30/dt)):
        elapsed=float(d.time)-phase_start
        current_tips=tips(robot)
        tip_mid=current_tips.mean(0)
        feedback=measure(robot)
        spring_force=spring.force(max(robot.gap(),0.)) if spring.reference_gap is not None else 0.
        raw=np.array(feedback["grip_each_N"])+spring_force
        filtered += dt/(.005+dt)*(raw-filtered)
        grip20 += dt/(.020+dt)*(float(raw.mean())-grip20)
        down=max(0.,sum(f[2] for f in feedback["force_on_fingers_N"]))
        peaks["grip_N"]=max(peaks["grip_N"],float(grip20))
        peaks["down_N"]=max(peaks["down_N"],down)
        for key in ("penetration_mm","nonfinger_mm"):
            peaks[key]=max(peaks[key],feedback[key])
        if grip20 > (25 if cfg.category=="bendable" else 15):
            outcome,reason="over_squeeze","v2.2 category grip threshold"
        if phase=="settle":
            if elapsed>=.3:
                baseline_com=sim.com().copy()
                next_phase="approach"
            else:next_phase=None
        elif phase=="approach":
            # Move downward until first physical contact, then insert no more
            # than 15mm at 10mm/s, with the v2.2 total 20N down-force stop.
            if max(feedback["grip_each_N"])>.05 or down>.05:
                contacted=True
            speed=.010 if contacted else .05
            if contacted:depth+=speed*dt
            target[2]-=speed*dt
            next_phase="close" if depth>=min(.015,.6*robot.size[2]) or (contacted and down>=20) else None
            if elapsed>8:
                outcome,reason="grasp_fail","no surface contact/approach timeout"
        elif phase in ("close","lift","lift_hold","carry","carry_hold"):
            gap=float(np.clip(gap+np.clip((filtered.mean()-10)*.002,-.05,.015)*dt,0.,.045))
            if phase=="close":
                if spring.reference_gap is None and contacted and all(feedback["top_contact_each"]):
                    spring.arm(max(robot.gap(),0.),cfg.pinch_coeff>0)
                stable=stable+dt if filtered.mean()>=9.5 else 0.
                next_phase="lift" if stable>=.05 or elapsed>=8 or gap<=0 else None
                if next_phase:
                    if cfg.pinch_coeff>0 and not pinch.try_bind(current_tips):
                        outcome,reason="grasp_fail","v1.3 pinch eligibility absent at end of close"
                    lift_site=target.copy(); lift_object=sim.com().copy()
            elif phase=="lift":
                target=lift_site+np.array([0.,0.,min(.25,.10*elapsed)])
                next_phase="lift_hold" if elapsed>=2.5 else None
            elif phase=="lift_hold":
                next_phase="carry" if elapsed>=.2 else None
                if next_phase and (feedback["support_N"]>.05 or sim.com()[2]<baseline_com[2]+.01):
                    outcome,reason="grasp_fail","object remains on table after lift and hold"
            elif phase=="carry":
                duration=np.sqrt(2*np.pi*.5/1.)
                a=min(1.,elapsed/duration)
                target=lift_site+np.array([.5*(a-np.sin(2*np.pi*a)/(2*np.pi)),0.,.25])
                next_phase="carry_hold" if a>=1 else None
            else:
                next_phase="lower" if elapsed>=.3 else None
            if phase!="close":
                present=pinch.active is not None or max(feedback["grip_each_N"])>.05
                loss=0. if present else loss+dt
                if loss>.050:
                    outcome,reason=("grasp_fail" if phase in ("lift","lift_hold") else "drop"),"bilateral contact/connection loss >50ms"
        elif phase=="lower":
            target[2]-=.2*dt
            next_phase="open" if feedback["support_N"]>=.3*robot.mass*9.81 or target[2]<=lift_site[2]+.005 else None
        elif phase=="open":
            pinch.release("open")
            spring.release()
            gap=min(.095,gap+.060*dt)
            next_phase="done" if elapsed>=1. else None
        else:
            raise RuntimeError(phase)
        if pinch.active is not None:
            pinch.check_load(dt,max(float(filtered.mean()),0.))
        # The reviewed gathering spring acts only between the real fingers.
        d.xfrc_applied[:]=0
        if spring_force:
            direction=d.geom_xpos[robot.pad_geom[1]]-d.geom_xpos[robot.pad_geom[0]]
            direction/=max(np.linalg.norm(direction),1e-12)
            for j,side in enumerate(("left","right")):
                d.xfrc_applied[m.body(f"{side}_finger").id,:3]=(1 if j else -1)*spring_force*direction
        robot.command(target,gap)
        sim.step()
        if step%sample_every==0:
            mujoco.mj_forward(m,d)
            det=float(sim.determinant().min())
            min_det=min(min_det,det)
            tracking=float(np.linalg.norm(d.site_xpos[robot.site]-target)*1000)
            robot.tracking.append(tracking)
            robot.torque_peaks=np.maximum(robot.torque_peaks,np.abs(d.actuator_force[robot.act]))
            peaks["tracking_mm"]=max(peaks["tracking_mm"],tracking)
            trace.append(dict(t=float(d.time),phase=phase,com=sim.com().tolist(),tip_mid=tips(robot).mean(0).tolist(),
                              gap=robot.gap(),command_gap=gap,grip_N=float(filtered.mean()),spring_N=spring_force,
                              attached=pinch.active is not None,pinch_load_N=pinch.load,min_det=det,
                              tracking_mm=tracking,**measure(robot)))
            if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
                outcome,reason="sim_fail","nonfinite state"
            elif det<=0:
                outcome,reason="sim_fail","tetrahedron inversion"
            if step%(sample_every*10)==0:
                states.append(d.qpos.copy())
            if time.perf_counter()-started>cfg.wall_limit:
                outcome,reason="compute_limit","wall budget reached; incomplete trial"
        if cfg.smoke_seconds and d.time>=cfg.smoke_seconds:
            outcome,reason="smoke_only","short entry check; no grasp inference"
        if outcome!="pending":break
        if next_phase:
            phases.append(dict(phase=phase,start_s=phase_start,end_s=float(d.time)))
            phase_start=float(d.time)
            if next_phase=="done":
                position=sim.com()
                success=bool(np.all(np.abs(position[:2]-(np.array([-.25,.04])+[.5,0]))<=.10))
                outcome,reason=("success","") if success else ("grasp_fail","final COM outside v2.2 target square")
                break
            phase=next_phase
    if outcome=="pending":outcome,reason="incomplete","simulation time budget reached"
    (output/"trace.json").write_text(json.dumps(trace,indent=2))
    np.savez_compressed(output/"trajectory.npz",qpos=np.array(states),time=np.arange(len(states))*.2,
                        final_qpos=d.qpos,final_qvel=d.qvel)
    result=dict(config=asdict(cfg),hostname=platform.node(),mujoco=mujoco.__version__,python=sys.executable,
                frozen_v13_sha256=RELEASE_SHA,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                robot_source_sha256=hashlib.sha256((ROOT/"policy/yam_probe_robot.py").read_bytes()).hexdigest(),
                outcome=outcome,reason=reason,phase=phase,sim_time_s=float(d.time),wall_s=time.perf_counter()-started,
                timing_is_throughput=False,peaks=peaks,min_determinant=min_det,phases=phases,
                initial_com=baseline_com.tolist(),final_com=sim.com().tolist(),pinch_events=pinch.events,
                pinch_frame="articulated palm; reactions load arm",robot=robot.diagnostics(),
                dimensions=dict(nv=m.nv,vertices=m.nflexvert,tetrahedra=m.nflexelem),
                warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(d.warning) if w.number})
    (output/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ("outcome","reason","phase","sim_time_s","peaks","min_determinant")}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--category",choices=("bendable","compressible"),default="bendable")
    for name,default in (("yaw",0.),("tilt",0.),("young",10000.),("dt",.001),("pinch-coeff",1.),("contact-time",.005),("wall-limit",1800.),("smoke-seconds",0.),("arm-height",.27)):
        parser.add_argument(f"--{name}",type=float,default=default)
    args=vars(parser.parse_args()); output=args.pop("output")
    try:
        run(Config(**args),output)
    except Exception as error:
        if output.exists():
            (output/"setup_error.json").write_text(json.dumps(dict(error=type(error).__name__,message=str(error)),indent=2))
        raise


if __name__=="__main__":main()

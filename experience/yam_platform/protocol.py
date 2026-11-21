"""Reusable full-YAM grasp protocol, independent of the content representation."""
from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from xml.etree import ElementTree as ET
import mujoco
import numpy as np
from .geometry import fingertip_xml
from .robot import Yam, asset_fingerprint
from .pinch import PinchSpring
from .provenance import source_files as platform_source_files

@dataclass
class TrialConfig:
    category: str='bendable'
    yaw: float=0.
    tilt: float=0.
    dt: float=.001
    pinch_coeff: float=1.
    contact_time: float=.005
    wall_limit: float=1800.
    smoke_seconds: float=0.
    arm_height: float=.27
    lift_height: float=.25
    gripper_overtravel: float=0.
    complete_motion: bool=False
    press_target_N: float | None=None
    press_timeout_s: float=12.
    press_start_offset_m: float=.10
    press_only: bool=False
    close_gap_hold_s: float=0.
    grasp_opening_m: float=.040
    lift_speed_m_s: float=.10
    pickup_only: bool=False
    pickup_release_seconds: float=0.
    side_insert_speed_m_s: float=.010
    closing_gain_m_per_N_s: float=.002
    lift_hold_seconds: float=.20


class NormalPress:
    """Approach quickly, then track film-normal force through the YAM pose servo."""

    support_limit_N = .05
    hold_s = .010

    def __init__(self, target_N, surface_z):
        self.target_N = target_N
        self.surface_z = surface_z
        self.filtered_N = 0.
        self.band_s = 0.
        self.reached = False
        self.peak_N = 0.
        self.peak_direct_support_N = 0.

    def move(self, z, tip_z, force_N, support_N, dt, approach):
        self.filtered_N += dt / (.010 + dt) * (force_N - self.filtered_N)
        if approach:
            self.peak_N = max(self.peak_N, force_N)
            self.peak_direct_support_N = max(self.peak_direct_support_N, support_N)
            qualified = .8 * self.target_N <= force_N <= 1.2 * self.target_N and support_N <= self.support_limit_N
            self.band_s = self.band_s + dt if qualified else 0.
            self.reached = self.band_s >= self.hold_s
        if support_N > self.support_limit_N:
            velocity = .004
        elif approach and tip_z > self.surface_z + .002 and force_N < .05:
            velocity = -.05
        else:
            velocity = float(np.clip(.002 * (self.filtered_N - self.target_N), -.002, .004))
        return z + dt * velocity

    def closure_rate(self, nominal_m_s, force_N, support_N):
        return nominal_m_s

def vec(x):
    return ' '.join(format(float(v),'.17g') for v in np.asarray(x).ravel())

class MuJoCoSim:
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
        self.m = getattr(built, "compile_model", mujoco.MjModel.from_xml_string)(built.xml)
        self.d = mujoco.MjData(self.m)
        self.desc = built.desc
        self.obj_geoms = set()
        self.table_geom = self.m.geom("table").id
        mujoco.mj_forward(self.m, self.d)

    def step(self):
        mujoco.mj_step(self.m,self.d)


def attach_robot(cfg,built,sim_factory,center,top,size,mass,approach=None):
    if approach is None:
        yaw, tilt = np.deg2rad([cfg.yaw, cfg.tilt])
        rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        rx = np.array([[1,0,0],[0,np.cos(tilt),-np.sin(tilt)],[0,np.sin(tilt),np.cos(tilt)]])
        rotation = rz @ rx @ np.diag([1., -1., -1.])
        offset = cfg.press_start_offset_m if cfg.press_target_N is not None else .10
        target = np.array([*center,top+offset])
    else:
        # Optional adapter-chosen approach (e.g. a horizontal side pinch), world frame: grasp-site z along the
        # inward direction n, y along the closing direction c; start 100 mm back from the contact point as
        # the default top approach does. yaw/tilt do not apply.
        n = np.asarray(approach["direction"], float)
        c = np.asarray(approach["closing"], float)
        if abs(np.linalg.norm(n)-1) > 1e-9 or abs(np.linalg.norm(c)-1) > 1e-9 or abs(n @ c) > 1e-9:
            raise ValueError("approach direction and closing must be orthonormal")
        close_advance = approach.get("close_advance_m")
        if close_advance is not None and (not np.isfinite(close_advance) or close_advance <= 0.):
            raise ValueError("side close advance must be positive and finite")
        lift_retract = approach.get("lift_retract_m", 0.)
        if not np.isfinite(lift_retract) or lift_retract < 0.:
            raise ValueError("side lift retraction must be finite and nonnegative")
        if lift_retract and (not cfg.pickup_only or n[2] >= 0. or -lift_retract * n[2] > cfg.lift_height):
            raise ValueError("side lift retraction requires pickup-only and an upward path within the lift height")
        rotation = np.column_stack([np.cross(c, n), c, n])
        target = np.asarray(approach["contact_point"], float) - .10*n
    robot = Yam(fixture=dict(built=built, target=target, rotation=rotation, sim_factory=sim_factory))
    m,d = robot.sim.m,robot.sim.d
    robot.control_period_steps = max(1, round(.01 / cfg.dt))
    robot.pad_sets = [{g for g in range(m.ngeom) if m.geom(g).name == f"{s}_pad" or m.geom(g).name.startswith(f"{s}_round_")} for s in ("left", "right")]
    robot.all_finger_sets = [set(robot.sim.pad_geoms[s]) | robot.pad_sets[j] for j,s in enumerate(("left", "right"))]
    robot.pad_geom = [m.geom(f"{s}_pad").id for s in ("left", "right")]
    robot.nonfinger_mask &= ~np.isin(np.arange(m.ngeom), list(set.union(*robot.all_finger_sets)))
    robot.size, robot.mass = size,mass
    if approach is not None:
        robot.approach, robot.approach_extent = n, float(abs(np.asarray(size) @ n))
        robot.close_advance_m = close_advance
        robot.lift_retract_m = lift_retract
    return robot



def tips(robot):
    m,d = robot.sim.m,robot.sim.d
    # Local pad z points toward the palm, so the distal edge lies at -20mm.
    return np.array([d.geom_xpos[g] - d.geom_xmat[g].reshape(3,3)[:,2]*.020 for g in robot.pad_geom])


def retracted_lift_displacement(approach, retraction_m, height_m, travel_m):
    """Follow the outward approach axis, then finish the commanded rise vertically."""
    withdrawal = min(retraction_m, travel_m)
    displacement = -withdrawal * approach
    displacement[2] += min(height_m + retraction_m * approach[2], travel_m - withdrawal)
    return displacement



def measure_contacts(robot, target_geoms=(), target_flexes=(), monitor_support=False):
    sim = robot.sim
    m,d = sim.m,sim.d
    result = np.zeros((2,3))
    top_contacts = [False,False]
    support = 0.
    penetration = 0.
    nonfinger = 0.
    direct_support = 0.
    buffer = np.zeros(6)
    approach = getattr(robot, "approach", None)
    for i,c in enumerate(d.contact):
        if c.efc_address < 0:
            continue
        gs = list(c.geom)
        if not (any(g in target_geoms for g in gs) or any(f in target_flexes for f in c.flex)):
            if monitor_support and sim.table_geom in gs:
                pad_slot = 1 - gs.index(sim.table_geom)
                if any(gs[pad_slot] in group for group in robot.all_finger_sets):
                    mujoco.mj_contactForce(m, d, i, buffer)
                    world = c.frame.reshape(3, 3).T @ buffer[:3]
                    direct_support += (1 if pad_slot == 1 else -1) * world[2]
            continue
        mujoco.mj_contactForce(m,d,i,buffer)
        world = c.frame.reshape(3,3).T @ buffer[:3]
        penetration = max(penetration, -float(c.dist)*1000)
        for j, group in enumerate(robot.all_finger_sets):
            for k in (0,1):
                if gs[k] in group:
                    result[j] += (1 if k==1 else -1)*world
                    # Top-facing contact normal pushing the finger upward (with an adapter
                    # approach: pushing the finger back against the approach direction).
                    facing = c.frame[2] if approach is None else -float(c.frame[:3] @ approach)
                    top_contacts[j] |= bool((1 if k==1 else -1)*facing > .5 and buffer[0] > .05)
        if sim.table_geom in gs:
            support += (1 if gs[0] == sim.table_geom else -1) * world[2]
        for g in gs:
            if g >= 0 and robot.nonfinger_mask[g]:
                nonfinger = max(nonfinger,-float(c.dist)*1000)
    axis = d.geom_xpos[robot.pad_geom[1]]-d.geom_xpos[robot.pad_geom[0]]
    axis /= max(np.linalg.norm(axis),1e-12)
    grip = [max(0.,float(-result[0]@axis)),max(0.,float(result[1]@axis))]
    measured = dict(grip_each_N=grip, contact_magnitude_each_N=np.linalg.norm(result,axis=1).tolist(),
                    force_on_fingers_N=result.tolist(), support_N=float(support),
                    top_contact_each=top_contacts, penetration_mm=penetration, nonfinger_mm=nonfinger)
    if monitor_support:
        measured["direct_support_N"] = max(0., float(direct_support))
    return measured





def run(backend, cfg, output, *, on_sample=None, on_penetration_peak=None, micro_lift=None):
    if not np.isfinite(cfg.lift_height) or cfg.lift_height <= 0:
        raise ValueError("lift_height must be finite and positive")
    if not np.isfinite(cfg.lift_speed_m_s) or cfg.lift_speed_m_s <= 0:
        raise ValueError("lift_speed_m_s must be finite and positive")
    if not np.isfinite(cfg.lift_hold_seconds) or cfg.lift_hold_seconds <= 0:
        raise ValueError("lift_hold_seconds must be finite and positive")
    if not np.isfinite(cfg.grasp_opening_m) or not 0 < cfg.grasp_opening_m <= .045:
        raise ValueError("grasp_opening_m must be in (0, .045] m")
    if not np.isfinite(cfg.side_insert_speed_m_s) or cfg.side_insert_speed_m_s <= 0:
        raise ValueError("side_insert_speed_m_s must be finite and positive")
    if not np.isfinite(cfg.closing_gain_m_per_N_s) or cfg.closing_gain_m_per_N_s <= 0:
        raise ValueError("closing_gain_m_per_N_s must be finite and positive")
    if not np.isfinite(cfg.gripper_overtravel) or cfg.gripper_overtravel < 0:
        raise ValueError("gripper_overtravel must be finite and nonnegative")
    if not np.isfinite(cfg.close_gap_hold_s) or cfg.close_gap_hold_s < 0:
        raise ValueError("close_gap_hold_s must be finite and nonnegative")
    if cfg.press_target_N is not None and (
            not np.isfinite(cfg.press_target_N) or cfg.press_target_N <= 0 or
            not np.isfinite(cfg.press_timeout_s) or cfg.press_timeout_s <= NormalPress.hold_s or
            not np.isfinite(cfg.press_start_offset_m) or cfg.press_start_offset_m <= 0.):
        raise ValueError("normal force target, start offset and approach timeout must be positive and finite")
    if cfg.press_only and cfg.press_target_N is None:
        raise ValueError("press-only trials require a normal force target")
    if not np.isfinite(cfg.pickup_release_seconds) or cfg.pickup_release_seconds < 0:
        raise ValueError("pickup_release_seconds must be finite and nonnegative")
    release_ramp = cfg.pickup_release_seconds > 0
    if release_ramp and (not cfg.pickup_only or cfg.pinch_coeff):
        raise ValueError("a pickup release requires pickup_only and native contact")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    (output/"config.json").write_text(json.dumps(asdict(cfg),indent=2))
    source_files = {**platform_source_files(), "platform": Path(__file__), "backend": Path(backend.__file__)}
    if hasattr(backend, "source_files"):
        source_files.update(backend.source_files())
    source_hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k,v in source_files.items()}
    for k,v in source_files.items():
        (output/(k+"_source.py")).write_bytes(v.read_bytes())
    robot = backend.scene(cfg)
    sim = robot.sim
    m,d = sim.m,sim.d
    if -cfg.gripper_overtravel < m.actuator_ctrlrange[robot.gripper, 0]:
        raise ValueError("gripper_overtravel exceeds the existing actuator range")
    minimum_gap = -2 * cfg.gripper_overtravel
    (output/"model.xml").write_text(sim.built.xml)
    if hasattr(backend, "save_model"):
        backend.save_model(sim, output)
    (output/"robot_assets.json").write_text(json.dumps(asset_fingerprint(),indent=2))
    pinch = backend.attachment(robot,cfg)
    spring = backend.gathering_spring(robot, cfg) if hasattr(backend, "gathering_spring") else PinchSpring()
    dt = cfg.dt
    gap = cfg.grasp_opening_m
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
    integration_states=[]
    state_times=[]
    contact_tip_z=None
    approach=getattr(robot,"approach",None)
    close_advance = getattr(robot, "close_advance_m", None)
    approach_origin = tips(robot).mean(0) if close_advance is not None else None
    lift_retract = getattr(robot, "lift_retract_m", 0.)
    lift_length = cfg.lift_height + lift_retract * (1. + approach[2]) if lift_retract else cfg.lift_height
    retried = False
    if micro_lift is not None and (approach is None or lift_retract or not cfg.pickup_only or cfg.lift_height <= micro_lift.height):
        raise ValueError("micro-lift requires a side pickup with a longer vertical lift")
    press = None
    if cfg.press_target_N is not None:
        if approach is not None:
            raise ValueError("normal force control requires the top approach")
        if "direct_support_N" not in backend.measure(robot):
            raise ValueError("normal force control requires separate direct support reactions")
        press = NormalPress(cfg.press_target_N, start[2] - cfg.press_start_offset_m)
    open_gap=None
    outcome="pending"
    reason=""
    grasp_failure=None
    motion_completed=False
    peaks=dict(penetration_mm=0.,nonfinger_mm=0.,grip_N=0.,down_N=0.,tracking_mm=0.)
    loss=0.
    stable=0.
    gap_hold=0.
    sample_every=max(1,round(.02/dt))
    release_sample_every=max(1,round(.002/dt))
    for step in range(round(getattr(cfg, "simulation_seconds", 30.)/dt)):
        elapsed=float(d.time)-phase_start
        current_tips=tips(robot)
        tip_mid=current_tips.mean(0)
        feedback=backend.measure(robot)
        spring_force=spring.force(max(robot.gap(),0.)) if spring.reference_gap is not None else 0.
        raw=np.array(feedback["grip_each_N"])+spring_force
        filtered += dt/(.005+dt)*(raw-filtered)
        grip20 += dt/(.020+dt)*(float(raw.mean())-grip20)
        if approach is None:
            down=max(0.,sum(f[2] for f in feedback["force_on_fingers_N"]))
        else:
            down=max(0.,-sum(float(np.dot(f,approach)) for f in feedback["force_on_fingers_N"]))
        peaks["grip_N"]=max(peaks["grip_N"],float(grip20))
        peaks["down_N"]=max(peaks["down_N"],down)
        if on_penetration_peak is not None and feedback["penetration_mm"] > peaks["penetration_mm"]:
            on_penetration_peak(robot, phase)
        for key in ("penetration_mm","nonfinger_mm"):
            peaks[key]=max(peaks[key],feedback[key])
        if grip20 > (25 if cfg.category=="bendable" else 15):
            outcome,reason="over_squeeze","v2.2 category grip threshold"
        if (press is not None or close_advance is not None) and feedback["penetration_mm"] > 1.:
            outcome,reason="sim_fail","film contact penetration exceeds 1 mm"
        if micro_lift is not None:
            micro_lift.observe(robot, phase, target, gap)
        if phase=="settle":
            if elapsed>=.3:
                baseline_com=sim.com().copy()
                next_phase="approach"
            else:next_phase=None
        elif phase=="approach":
            # Move downward until first physical contact, then insert no more
            # than 15mm at 10mm/s, with the v2.2 total 20N down-force stop.
            if press is not None:
                contacted = contacted or down > .05
                # The pose target can reach the surface before the arm has
                # followed it. Brake using the lower of the two tip heights.
                tracking_z = target[2] - d.site_xpos[robot.site, 2]
                lower_tip_z = tip_mid[2] + min(0., tracking_z)
                target[2] = press.move(target[2], lower_tip_z, down, feedback["direct_support_N"], dt, approach=True)
                next_phase = "close" if press.reached else None
                if press.reached and cfg.press_only and outcome == "pending":
                    outcome, reason = "press_ok", "normal force held in target band"
                elif not press.reached and elapsed >= cfg.press_timeout_s and outcome == "pending":
                    outcome, reason = "grasp_fail", "normal force target not held before approach timeout"
            elif approach is None:
                if max(feedback["contact_magnitude_each_N"])>.05:
                    if not contacted: contact_tip_z=float(tip_mid[2])
                    contacted=True
                speed=.010 if contacted else .05
                if contacted:depth=max(0.,contact_tip_z-float(tip_mid[2]))
                target[2]-=speed*dt
                next_phase="close" if depth>=min(.015,.6*robot.size[2]) or (contacted and down>=20) else None
            else:
                if max(feedback["contact_magnitude_each_N"])>.05:
                    if not contacted: contact_tip=tip_mid.copy()
                    contacted=True
                speed=cfg.side_insert_speed_m_s if contacted else .05
                if contacted:depth=max(0.,float((tip_mid-contact_tip)@approach))
                target+=approach*speed*dt
                if close_advance is None:
                    arrived = depth >= min(.015, .6 * robot.approach_extent)
                else:
                    arrived = contacted and float((tip_mid - approach_origin) @ approach) >= close_advance
                next_phase="close" if arrived or (contacted and down>=20) else None
            if press is None and elapsed>8:
                outcome,reason="grasp_fail","no surface contact/approach timeout"
        elif phase == "regrasp":
            target, gap, ready = micro_lift.retry_pose(elapsed, lift_site, retry_gap, cfg.grasp_opening_m, approach)
            next_phase = "close" if ready else None
        elif phase in ("close","lift","lift_hold","carry","carry_hold"):
            gap_rate = np.clip((filtered.mean()-10)*cfg.closing_gain_m_per_N_s,-.05,.015)
            if press is not None and phase == "close":
                gap_rate = press.closure_rate(gap_rate, down, feedback["direct_support_N"])
            gap=float(np.clip(gap+gap_rate*dt,minimum_gap,.045))
            if phase=="close":
                if press is not None:
                    target[2] = press.move(target[2], tip_mid[2], down, feedback["direct_support_N"], dt, approach=False)
                if spring.reference_gap is None and contacted and all(feedback["top_contact_each"]):
                    spring.arm(max(robot.gap(),0.),cfg.pinch_coeff>0)
                stable=stable+dt if filtered.mean()>=9.5 else 0.
                at_limit = gap <= minimum_gap
                gap_hold = gap_hold + dt if at_limit else 0.
                limit_hold_complete = at_limit and gap_hold >= cfg.close_gap_hold_s
                next_phase="lift" if stable>=.05 or elapsed>=8 or limit_hold_complete else None
                if next_phase:
                    if cfg.pinch_coeff>0 and not pinch.try_bind(current_tips):
                        outcome,reason="grasp_fail","v1.3 pinch eligibility absent at end of close"
                    lift_site=target.copy(); lift_object=sim.com().copy()
            elif phase=="lift":
                travel = cfg.lift_speed_m_s * elapsed
                retry_now = False
                if micro_lift is not None and not retried:
                    travel, retry_now = micro_lift.lift(elapsed, cfg.lift_speed_m_s, float(d.time))
                if lift_retract:
                    target = lift_site + retracted_lift_displacement(approach, lift_retract, cfg.lift_height, travel)
                else:
                    target=lift_site+np.array([0.,0.,min(cfg.lift_height,travel)])
                complete = elapsed >= lift_length / cfg.lift_speed_m_s if micro_lift is None else travel >= lift_length
                next_phase="lift_hold" if complete else None
                if retry_now:
                    next_phase = "regrasp"
                    retried = True
                    retry_gap = gap
                    stable = gap_hold = loss = 0.
                    pinch.release("regrasp")
                    spring.release()
            elif phase=="lift_hold":
                pickup_end = "open" if release_ramp else "done"
                next_phase=(pickup_end if cfg.pickup_only else "carry") if elapsed>=cfg.lift_hold_seconds else None
                if next_phase and feedback["support_N"]>.05:
                    outcome,reason="grasp_fail","object remains on table after lift and hold"
                elif next_phase == "open" and min(feedback["grip_each_N"]) < .05:
                    outcome,reason="grasp_fail","pickup release needs bilateral load after the hold"
            elif phase=="carry":
                duration=np.sqrt(2*np.pi*.5/1.)
                a=min(1.,elapsed/duration)
                target=lift_site+np.array([.5*(a-np.sin(2*np.pi*a)/(2*np.pi)),0.,cfg.lift_height])
                next_phase="carry_hold" if a>=1 else None
            else:
                next_phase="lower" if elapsed>=.3 else None
            if phase!="close":
                present=pinch.active is not None or max(feedback["contact_magnitude_each_N"])>.05
                loss=0. if present else loss+dt
                if loss>.050:
                    outcome,reason=("grasp_fail" if phase in ("lift","lift_hold") else "drop"),"both fingertip contacts/connection lost >50ms"
        elif phase=="lower":
            target[2]-=.2*dt
            supported=grasp_failure is None and feedback["support_N"]>=.3*robot.mass*9.81
            next_phase="open" if supported or target[2]<=lift_site[2]+.005 else None
        elif phase=="open":
            pinch.release("open")
            spring.release()
            if release_ramp:
                fraction=min(1.,elapsed/cfg.pickup_release_seconds)
                gap=open_gap+(cfg.grasp_opening_m-open_gap)*fraction
                present=max(feedback["contact_magnitude_each_N"])>.05
                loss=0. if present else loss+dt
                next_phase="done" if loss>=.050 else None
                if not next_phase and elapsed>=cfg.pickup_release_seconds+.10:
                    outcome,reason="release_fail","contact remains after the pickup release ramp"
            else:
                gap=min(.095,open_gap+.060)
                next_phase="done" if elapsed>=.5 else None
        else:
            raise RuntimeError(phase)
        if pinch.active is not None:
            if pinch.check_load(dt,max(float(filtered.mean()),0.)):
                spring.release()
                spring_force=0.
        # The reviewed gathering spring acts only between the real fingers.
        d.xfrc_applied[:]=0
        if spring_force:
            direction=d.geom_xpos[robot.pad_geom[1]]-d.geom_xpos[robot.pad_geom[0]]
            direction/=max(np.linalg.norm(direction),1e-12)
            for j,side in enumerate(("left","right")):
                d.xfrc_applied[m.body(f"{side}_finger").id,:3]=(1 if j else -1)*spring_force*direction
        robot.command(target,gap)
        previous_time=float(d.time)
        try:
            sim.step()
        except RuntimeError:
            model_failure, _ = backend.check_state(sim)
            if not model_failure:
                raise
            # Keep the last safely exported geometry and the backend's complete
            # failure state, even when a per-step validity check stops early.
            outcome, reason = "sim_fail", model_failure
            break
        if d.time < previous_time+dt/2:
            outcome,reason="sim_fail","MuJoCo automatic state reset"
        stride = release_sample_every if release_ramp and phase == "open" else sample_every
        if step%stride==0:
            if hasattr(backend, "refresh"):
                backend.refresh(sim)
            else:
                mujoco.mj_forward(m,d)
            model_failure,health=backend.check_state(sim)
            tracking=float(np.linalg.norm(d.site_xpos[robot.site]-target)*1000)
            robot.tracking.append(tracking)
            robot.torque_peaks=np.maximum(robot.torque_peaks,np.abs(d.actuator_force[robot.act]))
            peaks["tracking_mm"]=max(peaks["tracking_mm"],tracking)
            pressure_trace = {} if press is None else dict(down_N=down, press_band_s=press.band_s,
                                                           filtered_down_N=press.filtered_N)
            sampled_tip = tips(robot).mean(0)
            side_trace = {} if close_advance is None else dict(
                side_advance_m=float((sampled_tip - approach_origin) @ approach), side_close_advance_m=close_advance)
            trace.append(dict(t=float(d.time),phase=phase,com=sim.com().tolist(),tip_mid=sampled_tip.tolist(),
                              gap=robot.gap(),command_gap=gap,grip_N=float(filtered.mean()),spring_N=spring_force,
                              attached=pinch.active is not None,pinch_load_N=pinch.load,**health,
                              tracking_mm=tracking,**backend.measure(robot),**pressure_trace,**side_trace))
            if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
                outcome,reason="sim_fail","nonfinite state"
            elif model_failure:
                outcome,reason="sim_fail",model_failure
            states.append(d.qpos.copy())
            state_times.append(float(d.time))
            if hasattr(backend, "record_state"):
                integration_states.append(backend.record_state(sim))
            if on_sample is not None:
                on_sample(robot, trace[-1])
            if time.perf_counter()-started>cfg.wall_limit:
                outcome,reason="compute_limit","wall budget reached; incomplete trial"
        if cfg.smoke_seconds and d.time>=cfg.smoke_seconds:
            outcome,reason="smoke_only","short entry check; no grasp inference"
        if cfg.complete_motion and outcome in ("grasp_fail","drop") and phase in ("close","lift","lift_hold","carry","carry_hold"):
            # Complete the recorded arm motion while retaining the first failed
            # grasp verdict. Numerical and force-limit failures still stop.
            if grasp_failure is None:
                grasp_failure=dict(outcome=outcome,reason=reason,phase=phase,time_s=float(d.time))
            outcome,reason="pending",""
        if outcome!="pending":break
        if next_phase:
            phases.append(dict(phase=phase,start_s=phase_start,end_s=float(d.time)))
            phase_start=float(d.time)
            if next_phase=="done":
                motion_completed=True
                if cfg.pickup_only:
                    outcome,reason=("pickup_release_ok" if release_ramp else "pickup_ok"),""
                else:
                    position=sim.com()
                    success=bool(np.all(np.abs(position[:2]-(np.array([-.25,.04])+[.5,0]))<=.10))
                    outcome,reason=("success","") if success else ("grasp_fail","final COM outside v2.2 target square")
                if grasp_failure is not None:
                    outcome,reason=grasp_failure["outcome"],grasp_failure["reason"]
                break
            phase=next_phase
            if phase=="open":
                open_gap=gap
                if release_ramp:
                    loss=0.
    if outcome=="pending":outcome,reason="incomplete","simulation time budget reached"
    (output/"trace.json").write_text(json.dumps(trace,indent=2))
    extra_state = {}
    if integration_states:
        extra_state = dict(integration_state=np.asarray(integration_states),
                           integration_state_spec=np.asarray(backend.STATE_SPEC),
                           final_integration_state=backend.record_state(sim))
    np.savez_compressed(output/"trajectory.npz",qpos=np.array(states),time=np.array(state_times),
                        final_qpos=d.qpos,final_qvel=d.qvel,**extra_state)
    result=dict(config=asdict(cfg),hostname=platform.node(),mujoco=mujoco.__version__,python=sys.executable,
                frozen_v13_sha256=getattr(backend, "RELEASE_SHA", None),source_sha256=source_hashes,backend=backend.__name__,
                robot_source_sha256=hashlib.sha256(Path(__file__).with_name("robot.py").read_bytes()).hexdigest(),
                outcome=outcome,reason=reason,phase=phase,motion_completed=motion_completed,grasp_failure=grasp_failure,
                sim_time_s=float(d.time),wall_s=time.perf_counter()-started,
                timing_is_throughput=False,peaks=peaks,model_metrics=backend.summarize(trace),phases=phases,
                initial_com=baseline_com.tolist(),final_com=sim.com().tolist(),pinch_events=pinch.events,
                pinch_frame="articulated palm; reactions load arm",robot=robot.diagnostics(),
                dimensions=backend.dimensions(sim),
                warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(d.warning) if w.number})
    if micro_lift is not None:
        result["micro_lift"] = dict(decision=micro_lift.decision, prefix_sha256=micro_lift.prefix.hexdigest(),
                                    observations=micro_lift.samples)
    if press is not None:
        result["normal_press"] = dict(
            target_N=press.target_N, band_N=[.8 * press.target_N, 1.2 * press.target_N],
            required_hold_s=press.hold_s, achieved_hold_s=press.band_s, target_reached=press.reached,
            peak_N=press.peak_N, peak_direct_support_N=press.peak_direct_support_N)
    if lift_retract:
        result["side_lift_retraction_m"] = lift_retract
    (output/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ("outcome","reason","phase","sim_time_s","peaks","model_metrics")}),flush=True)
    return result

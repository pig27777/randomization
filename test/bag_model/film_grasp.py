"""Historical F7 grasp protocol for the film-pinch model."""
from pathlib import Path
import sys

BAG = Path(__file__).resolve().parents[2] / "bag_sim"
sys.path.insert(0, str(BAG))

import math
import time
from dataclasses import replace

import mujoco
import numpy as np

from loose_protocols import make, rotate_package, _rot, _cycloid, nearest_bag_body
from pinch import PinchAttachment, PinchSpring, DEPTH_MIN
from top_grasp import real_grip_force


from film_pinch import FilmPinch, contact_forces

def evaluate_film(k, bag, params, coefficient, seed=0, cache_dir=None, observer=None, binding_cls=None):
    """Frozen v2.2 controller and v1.3 spring/filter, with attachment targets on upper film only."""
    wall = time.perf_counter()
    u, v, psi, width, depth, force, lift_v, accel = params
    park = [('pA',1,(0,-1,.5)),('pB',1,(0,1,.5))]
    dt_scene = .0008 if bag.kind == 'flex' and k.rigid else .0004
    sim = make(k, replace(bag, pinch=True), (), big_table=True, dt=dt_scene, fingertips=park, cache_dir=cache_dir)
    cxy, size, h = sim.footprint()
    yaw = math.pi / 2 - psi
    rotate_package(sim, np.r_[cxy, 0.], yaw)
    g = cxy + _rot(yaw) @ (np.array([u, v]) * size)
    starts = {p: np.array([g[0], g[1]+sign*(width/2+.002), h+.005+.020])
              for p, sign in [('pA',-1),('pB',1)]}
    for p, x in starts.items(): sim.set_pad(p, x, np.zeros(3))
    mujoco.mj_forward(sim.m, sim.d)
    if bag.kind == 'effect':
        from effect_model import BagPinch
        # Use the observed grasp point to choose the target before the bag moves.
        local = (g - cxy) @ _rot(yaw)
        above_content = np.all(np.abs(local) <= np.asarray(k.size[:2])/2)
        if above_content:
            # The observed envelope must not block the frozen content-pinch contacts below it.
            sim.m.geom_conaffinity[sim.m.geom('bag_envelope').id] &= ~4
            mujoco.mj_forward(sim.m, sim.d)
            binding = PinchAttachment(sim, h)
            binding.coefficient = min(coefficient, k.pinch_coeff)
        else:
            binding = BagPinch(sim, coefficient)
    else:
        binding = (binding_cls or FilmPinch)(sim, coefficient)
    spring = PinchSpring()
    dt = sim.m.opt.timestep
    check = max(1, round(.010/dt))
    arm = np.zeros(3)
    prev = {p:x.copy() for p,x in starts.items()}
    phase, phase_t, t, n = 'settle', 0., 0., 0
    squeeze = ctrl = measured = maxforce = current = 0.
    real_max = spring_max = press_peak = 0.
    reactions = {p:np.zeros(3) for p in starts}
    first_contact = contact_height = None
    actual_depth = None
    effective_d, press_start, lift_start = depth, 0., 0.
    lost = pen = swing = 0.
    bag_lift = content_lift = lift_table = None
    outcome = reason = None
    phases, carry_positions = [], []
    rest_bag = rest_content = None
    angle = float(np.random.default_rng(seed).uniform(0,2*math.pi))
    carry = .5*np.array([math.cos(angle),math.sin(angle),0.])
    carry_t = math.sqrt(2*math.pi*.5/accel)

    def tips(): return np.array([sim.pad_pos(p)-[0,0,.020] for p in starts])
    def go(name):
        nonlocal phase, phase_t
        phase, phase_t = name, t
        phases.append(dict(t=t, phase=name))
    def end_press():
        nonlocal actual_depth
        actual_depth = max(0., press_start-arm[2])
        gap = sim.pad_pos('pB')[1]-sim.pad_pos('pA')[1]-.004
        # F12 的规则在压完时只看接触，聚拢在合拢结束时判；F7 的 FilmPinch 没有 press_eligible，行为不变。
        spring.arm(gap, getattr(binding, 'press_eligible', binding.candidate)(tips()) is not None)
        go('close')

    while outcome is None:
        tau = t-phase_t
        if phase == 'settle' and tau >= .3:
            rest_bag, rest_content = sim.bag_points().copy(), sim.com().copy()
            go('approach')
        elif phase == 'approach':
            if any(np.linalg.norm(f) > .05 for f in reactions.values()):
                first_contact = dict(t=t, tips=tips().tolist(), upper_contacts=[list(x) for x in binding.top_contacts()])
                contact_height = float(tips()[:,2].mean())
                binding.contact_height = tips()[:,2].copy()
                effective_d = min(depth, .6*max(contact_height,0.))
                press_start = arm[2]
                go('press')
            elif arm[2] < -h-.010:
                outcome, reason = 'grasp_fail', 'no film contact'
            else: arm[2] = -.010*tau
        elif phase == 'press':
            vertical = max(0.,sum(float(f[2]) for f in reactions.values()))
            press_peak = max(press_peak, vertical)
            if vertical >= 20. or tau >= effective_d/.010: end_press()
            else: arm[2] = press_start-min(.010*tau,effective_d)
        elif phase == 'close':
            if ctrl >= .95*force or squeeze >= width/2 or tau >= 8.:
                if current > 0 and binding.try_bind(tips()):
                    lift_start = arm[2]
                    go('lift')
                else: outcome, reason = 'grasp_fail', 'top-film pinch conditions or current force absent'
        elif phase == 'lift':
            arm[2] = lift_start+min(lift_v*tau,.250)
            if tau >= .250/lift_v+.2:
                bag_lift = float(np.mean(sim.bag_points()[:,2]-rest_bag[:,2])*1e3)
                content_lift = float((sim.com()[2]-rest_content[2])*1e3)
                lift_table = bool(any(c.efc_address >= 0 and c.dist <= 0 and
                    (c.geom1 == sim.table_geom or c.geom2 == sim.table_geom) for c in sim.d.contact))
                if lift_table: outcome, reason = 'grasp_fail', 'package still contacts table after 0.25 m lift'
                else: go('carry')
        elif phase == 'carry':
            arm = np.array([0.,0.,lift_start+.250])+_cycloid(tau,carry_t,1.)*carry
            if tau >= carry_t+.3:
                if sim.com()[2]-rest_content[2] < .030: outcome, reason = 'drop', 'content near table after carry'
                else: go('lower')
        elif phase == 'lower':
            arm = carry + [0,0,max(lift_start+.250-.2*tau,lift_start+.005)]
            if sim.table_force_pkg() >= .3*sim.pkg_mass*9.81 or arm[2] <= lift_start+.005:
                binding.release('open'); spring.release(); release_squeeze = squeeze; go('open')
        elif phase == 'open':
            squeeze = release_squeeze-min(.05*tau,.030)
            if tau >= .030/.05+.5:
                local = _rot(-angle) @ (sim.com()[:2]-rest_content[:2]-carry[:2])
                outcome = 'success' if np.all(np.abs(local) <= .100) else 'drop'
                break
        if phase in ('close','lift','carry','lower'):
            squeeze = float(np.clip(squeeze+np.clip(.0025*(force-ctrl),-.025,.025)*dt,0,width/2))
        targets = {p:x+[0,sgn*squeeze,0]+arm for (p,x),sgn in zip(starts.items(),[1,-1])}
        for p,x in targets.items():
            sim.set_pad(p,prev[p],(x-prev[p])/dt); prev[p]=x
        binding.set_frame(np.mean(list(targets.values()),axis=0)-[0,0,.020])
        gap = sim.pad_pos('pB')[1]-sim.pad_pos('pA')[1]-.004
        sf = spring.force(gap)
        for p,sgn in [('pA',-1),('pB',1)]: sim.d.xfrc_applied[sim.m.body(p).id,1] = sgn*sf
        sim.step(); t+=dt; n+=1
        reactions = contact_forces(sim)
        sf = spring.force(sim.pad_pos('pB')[1]-sim.pad_pos('pA')[1]-.004)
        real = real_grip_force(reactions)
        current = real+sf
        real_max,spring_max = max(real_max,real),max(spring_max,sf)
        ctrl += dt/(.005+dt)*(current-ctrl)
        measured += dt/(.020+dt)*(current-measured)
        maxforce = max(maxforce,measured)
        if binding.check_load(dt,current): spring.release()
        threshold = {'hard':50.,'bendable':25.,'compressible':15.}[k.category or ('hard' if k.rigid else 'bendable')]
        if measured > threshold and outcome is None: outcome,reason = 'over_squeeze',f'{measured:.3f} N in {phase}'
        if n%check == 0:
            pen = max(pen,sim.penetration())
            failure,_ = sim.bag_failure()
            if failure and outcome is None: outcome,reason = 'sim_fail',failure
            if phase in ('lift','carry'):
                contact = max(current if binding.active is not None else 0.,*[sim.pad_normal_force(p,a) for p in starts for a in range(3)])
                lost = lost+check*dt if contact < .05 else 0.
                if lost > .050 and outcome is None: outcome,reason = ('drop' if phase=='carry' else 'grasp_fail'),'contacts lost >50 ms'
            if phase == 'carry': carry_positions.append(sim.com()-tips().mean(0))
            if observer: observer(sim,binding,phase,t,current)
        if t>35 and outcome is None: outcome,reason = 'grasp_fail','execution timeout'
    if carry_positions:
        positions=np.array(carry_positions)
        swing=float(np.linalg.norm(positions[:,:2]-positions[-1,:2],axis=1).max()*1e3)
    return dict(outcome=outcome,reason=reason,params=params,seed=seed,bag_pinch_coeff=coefficient,
                max_grip_force_N=maxforce,max_real_force_N=real_max,max_spring_force_N=spring_max,
                bag_lift_mm=bag_lift,content_lift_mm=content_lift,content_in_bag=sim.outside_fraction()[0]==0,
                content_swing_mm=swing,lift_table_contact=lift_table,pinch_events=binding.events,
                first_contact=first_contact,actual_depth_mm=actual_depth*1e3 if actual_depth is not None else None,
                effective_depth_mm=effective_d*1e3,press_peak_N=press_peak,phases=phases,
                penetration_mm=pen*1e3,sim_s=t,wall_s=time.perf_counter()-wall,cache_hit=sim.cache_hit,
                max_ncon=sim.max_ncon,max_nefc=sim.max_nefc,pkg_weight_N=sim.pkg_mass*9.81,
                gather_mm=None if getattr(binding,'last_gather',None) is None else 1e3*binding.last_gather)

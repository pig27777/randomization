"""Protocol-v2.2 top grasp on frozen-v1 content plus the reviewed pinch effect.

evaluate(kappa_or_spec, TopGrasp_or_dict, seed) returns the task outcome.
measure_r4(kappa) uses the same descent/close controller and a fixed-bottom pull.
No content geometry, mass, connection coefficient or ordinary scene is modified.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

import mj_adapter as A
from model import G, Kappa, build
from pinch import PinchAttachment, PinchSpring, fragments

THRESHOLD = {'hard': 50., 'bendable': 25., 'compressible': 15.}
WIDTH, HEIGHT, RADIUS = .020, .040, .002
CLOSE_V, PRESS_V = .025, .010
LIFT_HEIGHT, PRESS_FORCE_LIMIT = .250, 20.
PROTOCOL_VERSION = 'v2.2'


@dataclass
class TopGrasp:
    u: float = 0.
    v: float = 0.
    psi: float = math.pi / 2
    w: float = .040
    d: float = .015
    F: float = 10.
    v_lift: float = .1
    a_max: float = 2.

    def validate(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('grasp parameters must be finite')
        bounds = dict(u=(-.5, .5), v=(-.5, .5), psi=(0, math.pi), w=(.020, .095),
                      d=(0, .040), F=(2, 50), v_lift=(.05, .3), a_max=(.5, 4))
        for name, (lo, hi) in bounds.items():
            if not lo <= getattr(self, name) <= hi:
                raise ValueError(f'{name} outside [{lo}, {hi}]')


def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def table_contacts(sim):
    """Active object/table contact pairs, independent of COM height or force cutoffs."""
    return [c for c in sim.d.contact if c.efc_address >= 0 and c.dist <= 0 and
            ((c.geom1 == sim.table_geom and c.geom2 in sim.obj_geoms) or
             (c.geom2 == sim.table_geom and c.geom1 in sim.obj_geoms))]


def real_grip_force(pad_forces):
    """Mean compressive reaction opposing the two fingers' closing directions."""
    return (max(0., -float(pad_forces['pA'][1])) + max(0., float(pad_forces['pB'][1]))) / 2


def fingertip_xml(name, position):
    v = A._v
    text = [f'<body name="{name}" pos="{v(position)}" gravcomp="1">',
            '<inertial pos="0 0 0" mass="100" diaginertia="1 1 1"/>']
    text += [f'<joint name="{name}_{an}" type="slide" axis="{v(np.eye(3)[a])}"/>' for a, an in enumerate('xyz')]
    attr = 'contype="0" conaffinity="0" rgba=".2 .6 .9 .6"'
    hx, hz = WIDTH / 2 - RADIUS, HEIGHT / 2 - RADIUS
    text.append(f'<geom name="{name}_core" type="box" size="{hx} {RADIUS} {hz}" {attr}/>')
    for axis, span, offset_axis, offset in ((0, hx, 2, hz), (2, hz, 0, hx)):
        for sign in (-1, 1):
            a, b = np.zeros(3), np.zeros(3)
            a[axis], b[axis] = -span, span
            a[offset_axis] = b[offset_axis] = sign * offset
            text.append(f'<geom name="{name}_e{axis}{sign}" type="capsule" fromto="{v(a)} {v(b)}" '
                        f'size="{RADIUS}" {attr}/>')
    return '\n'.join(text) + '</body>'


def task_scene(k, grasp, yaw, *, bottom_fixed=False):
    desc = build(k, omega_max=A.OMEGA_MAX)
    original_top = k.size[2] + 1e-4
    center = np.array([0., 0., k.size[2] / 2 + 1e-4])
    xy = rot(yaw) @ np.array([grasp.u * k.size[0], grasp.v * k.size[1]])
    starts = {p: np.array([xy[0], xy[1] + sign * (grasp.w / 2 + RADIUS), original_top + .005 + HEIGHT / 2])
              for p, sign in (('pA', -1), ('pB', 1))}
    frame, equality = fragments(desc)
    body = frame + ''.join(fingertip_xml(p, pos) for p, pos in starts.items())
    if bottom_fixed:
        bottom = ['obj'] if k.rigid else [f's{i}' for i, s in enumerate(desc.segments) if s.idx[2] == 0]
        equality += ''.join(f'<weld name="r4_fixed_{p}" body1="{p}" solref="0.001 1"/>' for p in bottom)
    built = A.scene(desc, center, worldbody=body, equality=equality)
    tree = ET.fromstring(built.xml)
    table = tree.find("./worldbody/geom[@name='table']")
    table.set('pos', '0 0 -0.025'); table.set('size', '2 2 0.025')
    contacts = tree.find('contact')
    # Only task fingertip pairs use 40 N/mm. Frozen object/table mechanics stay unchanged.
    for p in starts:
        for geom in tree.findall(f"./worldbody/body[@name='{p}']/geom"):
            for i, segment in enumerate(desc.segments):
                mass = k.mass if k.rigid else segment.mass
                kd = 40000. * (1 - A.CONTACT_D) / (A.CONTACT_D * mass)
                kd = min(kd, (.5 / A.BUSHING_DT) ** 2 / A.CONTACT_D)
                if k.skin_compliance:
                    stiffness = kd * A.CONTACT_D * mass / (1 - A.CONTACT_D)
                    kd /= 1 + stiffness * k.skin_compliance
                damping = 2 * math.sqrt(kd)
                bf = min(max(damping, 1000.), 1 / (2.5 * A.CONTACT_D * A.BUSHING_DT))
                ET.SubElement(contacts, 'pair', geom1=geom.attrib['name'], geom2=f's{i}', condim='3',
                    friction=f'{k.surface_friction} {k.surface_friction} .005 .0001 .0001',
                    solref=f'{-kd} {-damping}', solreffriction=f'{-kd} {-bf}', solimp='.99 .99 .001')
    built.xml = ET.tostring(tree, encoding='unicode')
    sim = A.Sim(built, starts)
    # Rotate only free content bodies; fingertips remain aligned with sim y.
    q = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    for j in range(sim.m.njnt):
        if sim.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        a = sim.m.jnt_qposadr[j]
        sim.d.qpos[a:a+2] = rot(yaw) @ sim.d.qpos[a:a+2]
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, q, sim.d.qpos[a+3:a+7].copy())
        sim.d.qpos[a+3:a+7] = quat
    if bottom_fixed and yaw:
        # R4 uses yaw zero below; a rotated fixed-bottom test needs transformed rest poses.
        raise ValueError('R4 fixed-bottom scene requires yaw zero')
    mujoco.mj_forward(sim.m, sim.d)
    return sim, starts, original_top


def evaluate(object_spec, grasp_params=None, seed=0, *, pull_test=False, trace=False, observer=None):
    wall = time.perf_counter()
    k = object_spec if isinstance(object_spec, Kappa) else Kappa.from_dict(object_spec.get('kappa', object_spec))
    grasp = grasp_params if isinstance(grasp_params, TopGrasp) else TopGrasp(**(grasp_params or {}))
    grasp.validate()
    category = k.category or ('hard' if k.rigid else 'bendable')
    yaw = 0. if pull_test else math.pi / 2 - grasp.psi
    sim, starts, original_top = task_scene(k, grasp, yaw, bottom_fixed=pull_test)
    binding = PinchAttachment(sim, original_top)
    spring = PinchSpring()
    dt = sim.m.opt.timestep
    check_every = max(1, round(.010 / dt))
    angle = float(np.random.default_rng(seed).uniform(0, 2 * math.pi))
    carry = .5 * np.array([math.cos(angle), math.sin(angle), 0.])
    carry_T = math.sqrt(2 * math.pi * .5 / grasp.a_max)
    arm, previous = np.zeros(3), {p: pos.copy() for p, pos in starts.items()}
    phase, phase_t, t, n = 'settle', 0., 0., 0
    squeeze = ctrl = measured = force_max = 0.
    press_start = lift_start = 0.
    effective_d, contact_height = grasp.d, None
    com_rest = lift_height = None
    outcome = reason = None
    lost = slip = peak_pull = peak_weld = 0.
    point_track = None
    phases, samples, diagnostics = [], [], []
    failure_state = None
    press_stop = None
    press_peak = 0.
    actual_depth = None
    rest_positions = rest_rotations = rest_gap = None
    hold_samples = []
    lift_table = None
    pad_forces = {p: np.zeros(3) for p in starts}
    current_force = real_force = spring_force = 0.
    real_force_max = spring_force_max = 0.
    first_contact = None

    def go(name):
        nonlocal phase, phase_t
        phase, phase_t = name, t
        phases.append(dict(t=t, phase=name))

    def tips_now():
        return np.array([sim.pad_pos(p) - [0, 0, HEIGHT / 2] for p in starts])

    def finish_press(force):
        nonlocal actual_depth, press_stop, press_peak
        actual_depth = max(0., press_start - arm[2])
        press_peak = max(press_peak, force)
        press_stop = dict(reason='force_limit' if force >= PRESS_FORCE_LIMIT else 'depth',
                          t=t, force_N=force, depth_m=actual_depth)
        gap = sim.pad_pos('pB')[1] - sim.pad_pos('pA')[1] - 2 * RADIUS
        spring.arm(gap, binding.candidate(tips_now()) is not None)
        go('close')

    while outcome is None:
        tau = t - phase_t
        if phase == 'settle':
            if tau >= .3:
                com_rest = sim.com().copy()
                rest_positions = sim.d.xpos.copy()
                rest_rotations = sim.d.xmat.copy().reshape(-1, 3, 3)
                go('approach')
        elif phase == 'approach':
            force = max(0., sum(float(f[2]) for f in pad_forces.values()))
            if any(np.linalg.norm(f) > .05 for f in pad_forces.values()):
                first_contact = dict(t=t, forces={p:f.tolist() for p,f in pad_forces.items()})
                contact_height = float(tips_now()[:, 2].mean())
                effective_d = min(grasp.d, .6 * max(contact_height, 0.))
                press_start = arm[2]; go('press')
                if force >= PRESS_FORCE_LIMIT:
                    finish_press(force)
            elif arm[2] < -original_top - .010:
                outcome, reason = 'grasp_fail', 'neither fingertip contacted the top'
            else:
                arm[2] = -PRESS_V * tau
        elif phase == 'press':
            press_force = max(0., sum(float(f[2]) for f in pad_forces.values()))
            press_peak = max(press_peak, press_force)
            if press_force >= PRESS_FORCE_LIMIT or tau >= effective_d / PRESS_V:
                finish_press(press_force)
            else:
                arm[2] = press_start - min(PRESS_V * tau, effective_d)
        elif phase == 'close':
            if ctrl >= .95 * grasp.F or squeeze >= grasp.w / 2 or tau >= 8.:
                gap = sim.pad_pos('pB')[1] - sim.pad_pos('pA')[1] - 2 * RADIUS
                if current_force > 0 and binding.try_bind(tips_now()):
                    lift_start = arm[2]
                    b = sim.seg_bodies[binding.segment]
                    local = sim.point_local(b, tips_now().mean(0))
                    point_track = (b, local)
                    rest_point = rest_positions[b] + rest_rotations[b] @ local
                    rest_gap = float(rest_point[2] - com_rest[2])
                    go('pull' if pull_test else 'lift')
                else:
                    binding.release('not_gripped')
                    outcome, reason = 'grasp_fail', 'top pinch conditions or current grip force absent'
        elif phase == 'lift':
            arm[2] = lift_start + min(grasp.v_lift * tau, LIFT_HEIGHT)
            if tau >= LIFT_HEIGHT / grasp.v_lift + .2:
                lift_height = float((sim.com()[2] - com_rest[2]) * 1000)
                contacts = table_contacts(sim)
                lift_table = dict(contact=bool(contacts), count=len(contacts),
                                  force_N=float(sim.contact_forces()[1][2]))
                if contacts:
                    outcome, reason = 'grasp_fail', 'object still contacts table after lift hold'
                else:
                    go('carry')
        elif phase == 'carry':
            z = tau / carry_T - math.sin(2 * math.pi * min(tau, carry_T) / carry_T) / (2 * math.pi)
            arm = np.array([0., 0., lift_start + LIFT_HEIGHT]) + min(z, 1.) * carry
            if tau >= carry_T + .3:
                if sim.com()[2] - com_rest[2] < .030:
                    outcome, reason = 'drop', 'COM returned near the table after carry'
                else:
                    go('lower')
        elif phase == 'lower':
            arm = carry + [0, 0, max(lift_start + LIFT_HEIGHT - .2 * tau, lift_start + .005)]
            if sim.contact_forces()[1][2] >= .3 * sim.mass * G or arm[2] <= lift_start + .005:
                binding.release('open'); spring.release(); go('open')
        elif phase == 'open':
            squeeze = release_squeeze - min(.05 * tau, .030)
            if tau >= .030 / .05 + .5:
                offset = rot(-angle) @ (sim.com()[:2] - com_rest[:2] - carry[:2])
                outcome = 'success' if np.all(np.abs(offset) <= .100) else 'drop'
                reason = None if outcome == 'success' else 'COM outside target square'
                break
        elif phase == 'pull':
            arm[2] = lift_start + .005 * tau
            if tau > 60:
                outcome, reason = 'grasp_fail', 'no pull-off within 60 s measurement window'

        if phase in ('close', 'lift', 'carry', 'lower', 'pull'):
            error = grasp.F - ctrl
            velocity = np.clip(.0025 * error, -CLOSE_V, CLOSE_V)
            squeeze = float(np.clip(squeeze + velocity * dt, 0., grasp.w / 2))
        if phase != 'open':
            release_squeeze = squeeze
        targets = {p: starts[p] + [0, sign * squeeze, 0] + arm for p, sign in (('pA', 1), ('pB', -1))}
        for p, target in targets.items():
            sim.set_pad(p, previous[p], (target - previous[p]) / dt)
            previous[p] = target
        midpoint = np.mean(list(targets.values()), axis=0) - [0, 0, HEIGHT / 2]
        binding.set_frame(midpoint)
        gap = sim.pad_pos('pB')[1] - sim.pad_pos('pA')[1] - 2 * RADIUS
        spring_force = spring.force(gap)
        # Equal/opposite reactions belong to the kinematic gripper, not the content bodies.
        for p, sign in (('pA', -1), ('pB', 1)):
            sim.d.xfrc_applied[sim.m.body(p).id, 1] = sign * spring_force
        sim.step()
        t += dt; n += 1
        gap = sim.pad_pos('pB')[1] - sim.pad_pos('pA')[1] - 2 * RADIUS
        pad_forces = sim.contact_forces()[0]
        spring_force = spring.force(gap)
        real_force = real_grip_force(pad_forces)
        current_force = real_force + spring_force
        real_force_max = max(real_force_max, real_force)
        spring_force_max = max(spring_force_max, spring_force)
        ctrl += dt / (.005 + dt) * (current_force - ctrl)
        measured += dt / (.020 + dt) * (current_force - measured)
        force_max = max(force_max, measured)
        weld_force = binding.force_vector()
        peak_weld = max(peak_weld, float(np.linalg.norm(weld_force)))
        if pull_test and phase == 'pull':
            # Wrist sensor sees the sum of attachment and real fingertip contact reactions.
            peak_pull = max(peak_pull, -float(weld_force[2] + sum(f[2] for f in pad_forces.values())), 0.)
        broke = binding.check_load(dt, current_force)
        if broke:
            spring.release()
            if pull_test:
                outcome, reason = 'pull_off', 'pinch capacity exceeded for 20 ms'
        if measured > THRESHOLD[category] and outcome is None:
            outcome, reason = 'over_squeeze', f'{measured:.6f} N in {phase}'
        if n % check_every == 0:
            failure = sim.failure()
            diagnostic = sim.velocity_diagnostic(np.array([0, 0, grasp.v_lift]), phase in ('lift', 'carry'))
            if max(diagnostic.values()) > 1 and not any(d['phase'] == phase for d in diagnostics):
                diagnostics.append(dict(t=t, phase=phase, **diagnostic, confirmed_failure=failure))
            if failure is not None and outcome is None:
                outcome, reason = 'sim_fail', failure
                failure_state = dict(t=t, phase=phase, qpos=sim.d.qpos.tolist(), qvel=sim.d.qvel.tolist(),
                                     warning_counts=[int(w.number) for w in sim.d.warning])
            if phase in ('lift', 'carry') and outcome is None:
                actual_touch = [max(sim.pad_normal_force(p, a) for a in range(3)) for p in starts]
                contact = max(current_force if binding.active is not None else 0., *actual_touch)
                lost = lost + check_every * dt if contact < .05 else 0.
                if lost > .050:
                    outcome, reason = ('grasp_fail' if phase == 'lift' else 'drop'), 'both contacts lost for >50 ms'
            if point_track is not None and phase in ('lift', 'carry'):
                b, local = point_track
                relative = sim.point_world(b, local) - tips_now().mean(0)
                slip = max(slip, float(np.linalg.norm(relative)))
                if phase == 'lift' and t - phase_t >= LIFT_HEIGHT / grasp.v_lift:
                    com_z = sim.com()[2]
                    material_z = sim.point_world(b, local)[2]
                    midpoint_z = tips_now()[:, 2].mean()
                    hold_samples.append(dict(t=t, gap_mm=float((midpoint_z-com_z)*1000),
                        extension_mm=float((midpoint_z-com_z-rest_gap)*1000),
                        material_extension_mm=float((material_z-com_z-rest_gap)*1000),
                        attachment_lag_mm=float((midpoint_z-material_z)*1000),
                        attachment_error_mm=float(np.linalg.norm(tips_now().mean(0)-sim.point_world(b,local))*1000),
                        static_bound=bool(binding.active is not None and t-phase_t >= LIFT_HEIGHT/grasp.v_lift+.1)))
            if trace:
                samples.append(dict(t=t, phase=phase, grip_force=current_force, measured_force=measured,
                                    real_grip_force=real_force, spring_force=spring_force,
                                    weld_force=weld_force.tolist(), pull_force=peak_pull,
                                    active=binding.active is not None, com=sim.com().tolist(),
                                    tips=tips_now().tolist(), top_contacts=[list(v) for v in binding.top_contacts()]))
            if observer is not None:
                observer(sim, binding, phase, t, current_force)
        if t > 90 and outcome is None:
            outcome, reason = 'grasp_fail', 'execution did not complete within 90 s'

    return dict(outcome=outcome, reason=reason, cls=category, max_grip_force_N=force_max,
                lift_height_mm=lift_height, slip_mm=slip * 1000, wall_s=time.perf_counter()-wall, sim_s=t,
                measured_pinch_coeff=peak_pull / 10 if pull_test else None, pull_force_N=peak_pull,
                peak_weld_force_N=peak_weld, pinch_events=binding.events, phases=phases,
                effective_depth_m=effective_d, contact_height_m=contact_height, trace=samples,
                diagnostics=diagnostics, failure_state=failure_state, protocol=PROTOCOL_VERSION,
                press_stop=press_stop, press_peak_vertical_force_N=press_peak,
                actual_depth_m=actual_depth, lift_table=lift_table, first_contact=first_contact,
                max_real_grip_force_N=real_force_max, max_spring_force_N=spring_force_max,
                rest_grip_to_com_gap_mm=None if rest_gap is None else rest_gap*1000,
                hold_samples=hold_samples,
                static_hold_sample_count=sum(s['static_bound'] for s in hold_samples),
                static_attachment_error_max_mm=max((s['attachment_error_mm'] for s in hold_samples if s['static_bound']),default=None),
                suspension_extension_mean_mm=float(np.mean([s['extension_mm'] for s in hold_samples])) if hold_samples else None,
                suspension_extension_max_mm=max((s['extension_mm'] for s in hold_samples), default=None),
                suspension_extension_final_mm=hold_samples[-1]['extension_mm'] if hold_samples else None,
                suspension_over_half_thickness=any(s['extension_mm'] > k.size[2]*500 for s in hold_samples))


def measure_r4(kappa, *, trace=True):
    return evaluate(kappa, TopGrasp(w=.040, d=.015, F=10.), 0, pull_test=True, trace=trace)

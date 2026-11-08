"""Full finite-actuator YAM scene and numerical IK for the probe experiment.

Upstream meshes, link inertias, joint limits, finger coupling and actuator limits
are retained. Joint positions are initialized once; every simulated step uses ctrl.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

import mj_adapter as A
from model import build
from top_grasp import contact_reactions, reaction_index

ASSET = Path('/home/ubuntu/workspace/gzl/workspace/asset/simulation-assets/external/i2rt-yam-menagerie/upstream')
ROTATION = np.array([[0.,-1.,0.],[-1.,0.,0.],[0.,0.,-1.]])


def asset_fingerprint():
    paths = [ASSET / 'yam_linear_4310.xml', ASSET / 'LICENSE', *sorted((ASSET / 'assets').rglob('*.stl'))]
    return {str(p.relative_to(ASSET)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def rotation_error(current, target):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, (target @ current.T).reshape(9))
    if q[0] < 0:
        q *= -1
    norm = np.linalg.norm(q[1:])
    return q[1:] * (2 * np.arctan2(norm, q[0]) / norm) if norm > 1e-12 else 2 * q[1:]


class Yam:
    def __init__(self, k, seed):
        # Opposite 0.5 m transfers are inside a common, explicitly checked arm workspace.
        self.carry = np.array([.5 if seed % 2 == 0 else -.5, 0., 0.])
        self.pick = -self.carry / 2
        self.initial_target = self.pick + [0., 0., k.size[2] / 2 + .0001]
        desc = build(k, omega_max=A.OMEGA_MAX)
        built = A.scene(desc, self.initial_target, pads=())
        root = ET.fromstring(built.xml)
        robot = ET.parse(ASSET / 'yam_linear_4310.xml').getroot()
        root.insert(0, ET.Element('compiler', angle='radian'))
        root.insert(1, deepcopy(robot.find('default')))
        assets = ET.SubElement(root, 'asset')
        for node in robot.find('asset'):
            node = deepcopy(node)
            if node.get('file'):
                node.set('file', str(ASSET / 'assets' / node.get('file')))
            assets.append(node)
        world = root.find('worldbody')
        table = world.find("geom[@name='table']")
        table.set('pos', '0 0 -0.025')
        table.set('size', '1 .22 .025')
        arm = deepcopy(robot.find("worldbody/body[@name='arm']"))
        arm.set('pos', '0 -.30 .12')
        arm.set('quat', '.707106781187 0 0 .707106781187')
        finger_names = {}
        for i, geom in enumerate(arm.iter('geom')):
            if geom.get('name') is None:
                geom.set('name', f'yam_geom_{i}')
        for p in ('left', 'right'):
            finger = arm.find(f".//body[@name='{p}_finger']")
            finger_names[p] = [g.get('name') for g in finger.iter('geom') if g.get('class') != 'visual']
        world.append(arm)
        ET.SubElement(world,'geom',name='yam_pedestal',type='box',pos='0 -.30 .06',
                      size='.06 .06 .06',rgba='.25 .3 .35 1',contype='0',conaffinity='0')
        root.find('contact').extend(deepcopy(list(robot.find('contact'))))
        root.find('equality').extend(deepcopy(list(robot.find('equality'))))
        root.append(deepcopy(robot.find('actuator')))
        # Explicit fingertip-content pairs preserve the experiment's 40 N/mm contact convention.
        for names in finger_names.values():
            for name in names:
                for i, segment in enumerate(desc.segments):
                    mass = k.mass if k.rigid else segment.mass
                    kd = min(40000. * (1-A.CONTACT_D)/(A.CONTACT_D*mass), (.5/A.BUSHING_DT)**2/A.CONTACT_D)
                    damping = 2*np.sqrt(kd)
                    bf = min(max(damping,1000.), 1/(2.5*A.CONTACT_D*A.BUSHING_DT))
                    ET.SubElement(root.find('contact'), 'pair', geom1=name, geom2=f's{i}', condim='3',
                        friction=f'{k.surface_friction} {k.surface_friction} .005 .0001 .0001',
                        solref=f'{-kd} {-damping}', solreffriction=f'{-kd} {-bf}', solimp='.99 .99 .001')
        built.xml = ET.tostring(root, encoding='unicode')
        self.sim = A.Sim(built)
        m, d = self.sim.m, self.sim.d
        # Package long axis is world y; its narrow side aligns with the actual finger closing axis.
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                a = m.jnt_qposadr[j]
                p = d.qpos[a:a+2] - self.pick[:2]
                d.qpos[a:a+2] = self.pick[:2] + [-p[1],p[0]]
                d.qpos[a+3:a+7] = [np.sqrt(.5),0,0,np.sqrt(.5)]
        self.sim.pad_names = ['left', 'right']
        self.sim.pad_geoms = {p: {m.geom(name).id for name in names} for p,names in finger_names.items()}
        self.index = reaction_index(self.sim)
        self.joints = np.array([m.joint(f'joint{i}').id for i in range(1,7)])
        self.qadr, self.dofs = m.jnt_qposadr[self.joints], m.jnt_dofadr[self.joints]
        self.act = np.array([m.actuator(f'joint{i}').id for i in range(1,7)])
        self.fingers = np.array([m.joint(f'{p}_finger').qposadr[0] for p in ('left','right')])
        self.gripper = m.actuator('gripper').id
        self.site = m.site('grasp_site').id
        self.pad_geom = [m.geom(f'{p}_pad').id for p in ('left','right')]
        self.kin = mujoco.MjData(m)
        self.jacp, self.jacr = np.zeros((3,m.nv)), np.zeros((3,m.nv))
        self.lo, self.hi = m.jnt_range[self.joints].T.copy()
        self.lo += 1e-5
        self.hi -= 1e-5
        self.qtarget = np.array([0.,1.5,1.5,0.,0.,0.])
        self.ik_errors, self.tracking, self.torque_peaks = [], [], np.zeros(6)
        self.control_clips = 0
        self.nonfinger_penetration_mm = 0.
        self.nonfinger_contact_samples = 0
        self.self_penetration_mm = 0.
        robot_bodies = {m.body('arm').id}
        for b in range(m.nbody):
            if m.body_parentid[b] in robot_bodies:
                robot_bodies.add(b)
        self.robot_mask = np.isin(m.geom_bodyid,list(robot_bodies))
        finger_geoms = set().union(*self.sim.pad_geoms.values())
        self.nonfinger_mask = self.robot_mask & ~np.isin(np.arange(m.ngeom),list(finger_geoms))
        self.target = self.initial_target.copy()
        self.solve(self.target, multistart=True)
        if self.ik_errors[-1]['position_mm'] > 1 or self.ik_errors[-1]['rotation_deg'] > 1:
            raise RuntimeError(f'YAM initial pose unreachable: {self.ik_errors[-1]}')
        d.qpos[self.qadr] = self.qtarget
        d.qpos[self.fingers] = .0475
        d.ctrl[self.act] = self.qtarget
        d.ctrl[self.gripper] = .0475
        mujoco.mj_forward(m,d)
        self.nominal_pad_offset = np.mean(d.geom_xpos[self.pad_geom],axis=0) - d.site_xpos[self.site]
        self.steps = 0

    def residual(self, q, target):
        self.kin.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.sim.m,self.kin)
        mujoco.mj_comPos(self.sim.m,self.kin)
        return np.r_[4*(target-self.kin.site_xpos[self.site]),
                     rotation_error(self.kin.site_xmat[self.site].reshape(3,3),ROTATION)]

    def solve(self, target, multistart=False):
        m = self.sim.m
        self.kin.qpos[:] = self.sim.d.qpos
        seeds = [self.qtarget]
        if multistart:
            seeds += [np.array([a,b,c,0.,e,0.]) for a in (-1.,0.,1.)
                      for b,c,e in ((1.,1.5,0.),(2.,2.,-.7),(.5,2.5,.7))]
        best = None
        for seed in seeds:
            q = np.clip(seed.copy(),self.lo,self.hi)
            for _ in range(180 if multistart else 15):
                error = self.residual(q,target)
                if np.linalg.norm(error[:3]) < .0004 and np.linalg.norm(error[3:]) < .002:
                    break
                mujoco.mj_jacSite(m,self.kin,self.jacp,self.jacr,self.site)
                J = np.vstack([4*self.jacp[:,self.dofs],self.jacr[:,self.dofs]])
                dq = J.T @ np.linalg.solve(J@J.T + .0001*np.eye(6),error)
                q = np.clip(q+np.clip(dq,-.15,.15),self.lo,self.hi)
            error = self.residual(q,target)
            score = np.linalg.norm(error)
            if best is None or score < best[0]:
                best = score,q.copy(),error.copy()
            if score < .001:
                break
        _,self.qtarget,error = best
        result = dict(position_mm=float(np.linalg.norm(error[:3])/4*1000),
                      rotation_deg=float(np.linalg.norm(error[3:])*180/np.pi))
        self.ik_errors.append(result)
        return result

    def command(self, position, gap_command):
        m,d = self.sim.m,self.sim.d
        self.target = np.asarray(position)
        if self.steps % 25 == 0:
            self.solve(self.target)
        ff = d.qfrc_bias[self.dofs] - d.qfrc_gravcomp[self.dofs]
        ctrl = self.qtarget + ff/m.actuator_gainprm[self.act,0]
        clipped = np.clip(ctrl,m.actuator_ctrlrange[self.act,0],m.actuator_ctrlrange[self.act,1])
        self.control_clips += int(np.any(clipped != ctrl))
        d.ctrl[self.act] = clipped
        d.ctrl[self.gripper] = np.clip(gap_command/2,-.005,.0475)
        self.steps += 1

    def observe(self):
        m,d = self.sim.m,self.sim.d
        self.torque_peaks = np.maximum(self.torque_peaks,np.abs(d.actuator_force[self.act]))
        forces,table = contact_reactions(self.sim,self.index)
        axis = d.geom_xpos[self.pad_geom[1]]-d.geom_xpos[self.pad_geom[0]]
        axis /= np.linalg.norm(axis)
        raw = {'pA': max(0.,-float(forces['left']@axis)),
               'pB': max(0.,float(forces['right']@axis))}
        if self.steps % 25 == 0:
            self.tracking.append(float(np.linalg.norm(d.site_xpos[self.site]-self.target)*1000))
            for c in d.contact:
                g1,g2 = c.geom
                if c.dist >= 0 or c.efc_address < 0:
                    continue
                if self.robot_mask[g1] and self.robot_mask[g2]:
                    self.self_penetration_mm = max(self.self_penetration_mm,-float(c.dist)*1000)
                if ((self.nonfinger_mask[g1] and (g2 in self.sim.obj_geoms or g2 == self.sim.table_geom)) or
                    (self.nonfinger_mask[g2] and (g1 in self.sim.obj_geoms or g1 == self.sim.table_geom))):
                    self.nonfinger_contact_samples += 1
                    self.nonfinger_penetration_mm = max(self.nonfinger_penetration_mm,-float(c.dist)*1000)
        return raw,table

    def gap(self):
        # Linear-4310 encoders: the coupled rail positions sum to the nominal inner gap.
        return float(np.sum(self.sim.d.qpos[self.fingers]))

    def diagnostics(self):
        return dict(robot='YAM linear_4310, full articulated dynamics',
                    ik_max_position_mm=max(r['position_mm'] for r in self.ik_errors),
                    ik_max_rotation_deg=max(r['rotation_deg'] for r in self.ik_errors),
                    tracking_max_mm=max(self.tracking,default=0.),
                    tracking_p95_mm=float(np.percentile(self.tracking,95)) if self.tracking else 0.,
                    actuator_peak_Nm=self.torque_peaks.tolist(),control_clip_steps=self.control_clips,
                    nonfinger_contact_samples=self.nonfinger_contact_samples,
                    nonfinger_max_penetration_mm=self.nonfinger_penetration_mm,
                    self_max_penetration_mm=self.self_penetration_mm,
                    original_force_limits_Nm=self.sim.m.actuator_forcerange[self.act].tolist())

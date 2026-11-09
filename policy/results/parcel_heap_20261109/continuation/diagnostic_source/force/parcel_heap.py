"""YAM mixed-parcel heap diagnostic; estimated materials, contact-only grasping.

The target shell is identical within each family before settling. Context parcels
are free rigid boxes. This is a small mixed heap, not a calibrated warehouse pile.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,dataclass,replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from xml.etree import ElementTree as ET

import numpy as np

from probe_feedback import prepare, load_backend, save, digest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'policy/results/parcel_heap_20261109'
CASES = ('book', 'clothes', 'empty_envelope', 'paper_envelope', 'card_envelope')
LAYOUTS = ('top', 'partial', 'gap')
DT = .0004
CENTER = np.array([-.20, .04])
SUPPORT_Z = .08
TOP_ROTATION = np.diag([1., -1., -1.])
# Rotate the overhead pose about world x: approach horizontally from negative y,
# with vertical finger closure. All rotations remain proper SO(3) matrices.
SIDE_ROTATION = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]) @ TOP_ROTATION


@dataclass(frozen=True)
class HeapSettings:
    stack_lift: float = 0.
    curvature_fix: bool = False
    solver: str = 'Newton'
    iterations: int = 100
    contact_time_s: float = .002
    timestep_scale: float = 1.
    adaptive_close: bool = False
    every_step_safety: bool = False
    wall_limit_s: float = 300.


SETTINGS = HeapSettings()
BACKEND_RUN = None
_CURVATURE_HELPERS = {}


def apply_curvature(sim,run):
    manifest=json.loads((run/'helper_manifest.json').read_text())
    source=run/'source_helpers/flex_curvature.py'
    checksum=digest(source)
    if checksum!=manifest['sha256']:
        raise RuntimeError('curvature helper changed')
    if checksum not in _CURVATURE_HELPERS:
        spec=importlib.util.spec_from_file_location('frozen_flex_curvature',source)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CURVATURE_HELPERS[checksum]=module.apply_curved_reference_fix
    apply=_CURVATURE_HELPERS[checksum]
    import mujoco
    changes=[apply(sim.m,sim.built.xml,mujoco.mj_id2name(sim.m,mujoco.mjtObj.mjOBJ_FLEX,i))
             for i in range(sim.m.nflex)]
    mujoco.mj_forward(sim.m,sim.d)
    return changes


def vector(values):
    return ' '.join(f'{x:.10g}' for x in np.asarray(values).ravel())


def shell_mesh(length, width, height, radius):
    """Two triangulated panels sharing a sealed rim, no artificial thick walls."""
    xs = np.linspace(-length / 2, length / 2, 11)
    ys = np.linspace(-width / 2, width / 2, 7)
    vertices, indices, faces = [], {}, []
    for side in (0, 1):
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                rim = i in (0, len(xs)-1) or j in (0, len(ys)-1)
                if side == 1 and rim:
                    indices[side, i, j] = indices[0, i, j]
                    continue
                taper = min(1., (length/2-abs(x))/(length*.1),
                            (width/2-abs(y))/(width*.15))
                z = radius + .0003
                if side:
                    z += 2*radius + height*taper
                elif rim:
                    z += radius
                indices[side, i, j] = len(vertices)
                vertices.append([x+CENTER[0], y+CENTER[1], z+SUPPORT_Z])
        for i in range(len(xs)-1):
            for j in range(len(ys)-1):
                a,b,c,d = [indices[side, *p] for p in ((i,j),(i+1,j),(i+1,j+1),(i,j+1))]
                faces.extend(((a,b,c),(a,c,d)) if side else ((a,c,b),(a,d,c)))
    return np.asarray(vertices), np.asarray(faces, dtype=int)


def flex_xml(name, vertices, faces, *, mass, radius, membrane, bending, rgba):
    # Match an estimated membrane modulus and bending rigidity. t_eff is a
    # constitutive parameter; collision thickness is independently 2*radius.
    effective_t = np.sqrt(12*(1-.3**2)*bending/membrane)
    return (f'<flexcomp name="{name}" type="direct" dim="2" radius="{radius}" '
            f'point="{vector(vertices)}" element="{vector(faces)}" mass="{mass}" rgba="{rgba}">'
            f'<elasticity young="{membrane/effective_t}" poisson=".3" thickness="{effective_t}" '
            'damping="0.0001" elastic2d="both"/>'
            '<contact contype="2" conaffinity="7" condim="3" priority="2" '
            'friction=".6 .005 .0001" solref=".002 1" solimp=".999 .999 .0001" selfcollide="auto"/>'
            '</flexcomp>')


def parcel_xml(name, center, half, mass, yaw=0.):
    return (f'<body name="{name}" pos="{vector(center)}" quat="{vector([np.cos(yaw/2),0,0,np.sin(yaw/2)])}">'
            f'<freejoint name="{name}_free"/><geom name="{name}" type="box" size="{vector(half)}" '
            f'mass="{mass}" contype="1" conaffinity="7" condim="3" friction=".6 .005 .0001" '
            'solref=".002 1" solimp=".99 .99 .0001" rgba=".48 .34 .22 1"/></body>')


def scene(case, layout, action='center', jitter=0., *, settings=None):
    import mujoco
    import mj_adapter as A
    from model import ANCHORS, build
    from yam_probe_robot import Yam

    settings = SETTINGS if settings is None else settings
    thin = 'envelope' in case
    radius = .000075 if thin else .00025
    length, width, height = (.22, .11, .0008) if thin else (.24, .11, .046)
    shell, faces = shell_mesh(length, width, height, radius)
    shell_mass = .006 if thin else .012
    fragments = [flex_xml('bag', shell, faces, mass=shell_mass, radius=radius,
                         membrane=30000. if thin else 1000.,
                         bending=.00002 if thin else .000004, rgba='.72 .56 .36 1')]
    k = replace(ANCHORS['bendable' if case == 'clothes' else 'hard'],
                size=(.18,.06,.04) if not thin else (.16,.07,.0003), mass=.3 if not thin else .012)
    has_segment_content = case in ('book', 'clothes', 'card_envelope')
    if not has_segment_content:
        # A.Sim requires an obj body. This noninteracting, gravcomp dummy is
        # outside the work area, invisible, and excluded from all package metrics.
        k = replace(ANCHORS['hard'], size=(.001,.001,.001), mass=.000001, segments=(1,1,1))
    desc = build(k, omega_max=A.OMEGA_MAX)
    origin = [*CENTER, SUPPORT_Z + 2*radius + .0006 + k.size[2]/2]
    if not has_segment_content:
        origin = [3.,3.,3.]
    if case == 'paper_envelope':
        px, py = np.linspace(-.08,.08,9), np.linspace(-.035,.035,5)
        paper = np.array([[x+CENTER[0],y+CENTER[1],SUPPORT_Z+.00065] for x in px for y in py])
        pf = []
        for i in range(8):
            for j in range(4):
                a = i*5+j
                pf.extend(((a,a+5,a+6),(a,a+6,a+1)))
        fragments.append(flex_xml('paper',paper,np.asarray(pf),mass=.012,radius=.00015,
                                 membrane=30000.,bending=.00006,rgba='.9 .9 .88 1'))
    neighbors = []
    if layout == 'gap':
        supports = [([CENTER[0]+s*.085,CENTER[1],.04],[.055,.075,.04],.65) for s in (-1,1)]
    else:
        supports = [([*CENTER,.04],[.135,.075,.04],1.3)]
    # Surrounding independently moving parcels make access and disturbance real
    # scene constraints. These are rigid context boxes, not soft target surrogates.
    supports += [([-.40,.21,.045],[.08,.07,.045],.7),
                 ([-.16,.23,.055],[.12,.065,.055],.9),
                 ([.04,.12,.035],[.065,.08,.035],.5),
                 ([-.40,.21,.115],[.055,.05,.025],.2)]
    if layout == 'partial':
        supports.append(([CENTER[0]+.085,CENTER[1]+.018,SUPPORT_Z+height+.022],
                         [.05,.038,.02],.15))
    for i, (pos, half, mass) in enumerate(supports):
        name = f'parcel_{i}'
        neighbors.append(name)
        fragments.append(parcel_xml(name,pos,half,mass,0. if i<len(supports)-1 else .08))
    dt=(.0001 if thin else DT)*settings.timestep_scale
    built = A.scene(desc,origin,pads=(),dt=dt,worldbody=''.join(fragments),
                    pad_contact=False,table_contact=False,solver=settings.solver)
    root = ET.fromstring(built.xml)
    root.find('option').set('integrator','discrete')
    root.find('option').set('iterations',str(settings.iterations))
    root.find('size').set('memory','128M')
    for flex in root.iter('flexcomp'):
        flex.find('contact').set('solref',f'{settings.contact_time_s} 1')
    if settings.stack_lift:
        # Lift only package objects, then support the whole heap with a free box.
        # Robot mounting and table remain unchanged; no body is pinned in place.
        for body in root.find('worldbody').findall('body'):
            pos=np.fromstring(body.get('pos','0 0 0'),sep=' ')
            if body.get('name')=='obj' and not has_segment_content:
                continue
            pos[2]+=settings.stack_lift
            body.set('pos',vector(pos))
        for flex in root.iter('flexcomp'):
            points=np.fromstring(flex.get('point'),sep=' ').reshape(-1,3)
            points[:,2]+=settings.stack_lift
            flex.set('point',vector(points))
        shell[:,2]+=settings.stack_lift
        foundation=parcel_xml('heap_foundation',[-.21,.10,settings.stack_lift/2],
                              [.34,.22,settings.stack_lift/2],3.)
        root.find('worldbody').append(ET.fromstring(foundation))
        neighbors.append('heap_foundation')
    if not has_segment_content:
        dummy = root.find("worldbody/body[@name='obj']")
        dummy.set('gravcomp','1')
        for geom in dummy.iter('geom'):
            geom.set('contype','0'); geom.set('conaffinity','0'); geom.set('rgba','0 0 0 0')
    built.xml = ET.tostring(root, encoding='unicode')
    built.heap_info = dict(case=case,layout=layout,thin=thin,radius=radius,
                           shell_initial=shell,faces=faces,neighbors=neighbors,
                           has_segment_content=has_segment_content,height=height,settings=asdict(settings),
                           shell_sha256=hashlib.sha256(shell.tobytes()+faces.tobytes()).hexdigest())
    z = SUPPORT_Z + (height*.52 if not thin else .010)
    grasp = np.array([*CENTER,z])
    if action == 'end':
        grasp[0] -= length*.37
    elif action == 'corner':
        grasp[:2] += [-length*.4,-width*.30]
    elif action == 'side':
        grasp[1] -= width*.48
        grasp[2] = SUPPORT_Z + height*.5 + .0005
        if settings.adaptive_close:
            # Visible local film mid-height at the exposed rim, not the hidden
            # content center: both fingers should straddle the same local patch.
            taper=min(1.,(width/2-abs(grasp[1]-CENTER[1]))/(width*.15))
            grasp[2]=SUPPORT_Z+2*radius+.0003+height*taper/2
    elif action != 'center':
        raise ValueError(action)
    grasp[0] += jitter
    grasp[2] += settings.stack_lift
    rotation = SIDE_ROTATION if action == 'side' else TOP_ROTATION
    start = grasp + ([0.,-.07,0.] if action == 'side' else [0.,0.,.09])
    robot = Yam(k,0,fixture=dict(built=built,sim_factory=HeapSim,target=start,rotation=rotation))
    robot.grasp_target = grasp
    robot.settings = settings
    return robot


def install_backend(run):
    global SETTINGS,BACKEND_RUN
    BACKEND_RUN=run
    path=run/'settings.json'
    SETTINGS=HeapSettings(**json.loads(path.read_text())) if path.exists() else HeapSettings()
    load_backend(run)
    # Define the subclass after selecting and verifying the frozen release.
    global HeapSim
    import mj_adapter as A

    class HeapSim(A.Sim):
        def __init__(self,built):
            super().__init__(built)
            self.info = built.heap_info
            self.curvature_changes = apply_curvature(self,BACKEND_RUN) if self.info['settings']['curvature_fix'] else []
            self.bag_bodies = np.flatnonzero(np.isin(np.arange(self.m.nbody),
                self.m.flex_vertbodyid[self.m.flex_vertadr[0]:self.m.flex_vertadr[0]+self.m.flex_vertnum[0]]))
            self.paper_bodies = (np.unique(self.m.flex_vertbodyid[self.m.flex_vertadr[1]:
                self.m.flex_vertadr[1]+self.m.flex_vertnum[1]]) if self.m.nflex>1 else np.array([],int))
            self.content_bodies = np.asarray(self.obj_bodies if self.info['has_segment_content'] else
                                             self.paper_bodies, dtype=int)
            self.package_bodies = np.r_[self.bag_bodies,self.content_bodies]
            self.neighbor_bodies = np.array([self.m.body(n).id for n in self.info['neighbors']])
            self.neighbor_geoms = {self.m.geom(n).id for n in self.info['neighbors']}

        def compaction_load(self,external):
            forces = external[self.seg_bodies,:3]
            total,resultant = np.linalg.norm(forces,axis=1).sum(),forces.sum(0)
            c = self.d.contact
            inside = np.zeros(c.geom.shape,bool)
            valid = c.geom>=0
            inside[valid] = self._obj_geom_mask[c.geom[valid]]
            selected = (inside[:,0]!=inside[:,1]) & (c.efc_address>=0)
            if selected.any():
                fn = np.maximum(self.d.efc_force[c.efc_address[selected]],0)
                total += fn.sum()
                resultant += (np.where(inside[selected,1],fn,-fn)[:,None]*c.frame[selected,:3]).sum(0)
            return max(0.,float(.5*(total-np.linalg.norm(resultant))))

        def package_com(self):
            ids = self.package_bodies
            masses = self.m.body_mass[ids]
            return np.sum(self.d.xipos[ids]*masses[:,None],axis=0)/masses.sum()

        def shell_points(self):
            return self.d.flexvert_xpos[:self.m.flex_vertnum[0]].copy()


def contacts(robot):
    """Decode only active native contacts, distinguishing flex from geom=-1."""
    import mujoco
    sim = robot.sim
    m,d = sim.m,sim.d
    normals = np.zeros(2)
    support = penetration = nonfinger = neighbor_touch = self_contact = 0.
    bypass = 0
    wrench = np.zeros(6)
    axis = d.geom_xpos[robot.pad_geom[1]] - d.geom_xpos[robot.pad_geom[0]]
    axis /= np.linalg.norm(axis)
    for i,c in enumerate(d.contact):
        if c.efc_address < 0:
            continue
        g = c.geom
        target = [(g[j] in sim.obj_geoms and sim.info['has_segment_content']) or c.flex[j]>=0
                  for j in range(2)]
        if not any(target):
            continue
        mujoco.mj_contactForce(m,d,i,wrench)
        world = c.frame.reshape(3,3).T @ wrench[:3]
        penetration = max(penetration,max(0.,-c.dist))
        for j in range(2):
            if g[j] < 0 or not target[1-j]:
                continue
            force = world if j==1 else -world
            for p,name in enumerate(('left','right')):
                if g[j] in sim.pad_geoms[name]:
                    normals[p] += abs(float(force@axis))
            if robot.nonfinger_mask[g[j]]:
                nonfinger = max(nonfinger,max(0.,-c.dist))
            if g[j] == sim.table_geom or g[j] in sim.neighbor_geoms:
                support -= force[2]
                if (g[1-j] in sim.obj_geoms and sim.info['has_segment_content']) or c.flex[1-j]>0:
                    bypass += 1
    # Robot-neighbor contact is independent of whether the target participates.
    for c in d.contact:
        a,b = c.geom
        if c.efc_address>=0 and a>=0 and b>=0:
            if robot.robot_mask[a] and robot.robot_mask[b]:
                self_contact = max(self_contact,max(0.,-c.dist))
            if ((robot.robot_mask[a] and b in sim.neighbor_geoms) or
                    (robot.robot_mask[b] and a in sim.neighbor_geoms)):
                neighbor_touch = max(neighbor_touch,max(0.,-c.dist))
            environment = sim.neighbor_geoms | {sim.table_geom}
            if ((robot.nonfinger_mask[a] and b in environment) or
                    (robot.nonfinger_mask[b] and a in environment)):
                nonfinger = max(nonfinger,max(0.,-c.dist))
    return dict(force=normals.tolist(),support_N=float(support),penetration_mm=float(penetration*1000),
                nonfinger_mm=float(nonfinger*1000),neighbor_contact_mm=float(neighbor_touch*1000),
                self_contact_mm=float(self_contact*1000),
                direct_content_support_contacts=bypass)


def outside_count(points, vertices, faces):
    """Solid-angle containment of content body centers in the closed target shell."""
    outside = 0
    for point in points:
        a,b,c = (vertices[faces]-point).transpose(1,0,2)
        na,nb,nc = (np.linalg.norm(x,axis=1) for x in (a,b,c))
        numerator = np.einsum('ij,ij->i',a,np.cross(b,c))
        denominator = (na*nb*nc + np.einsum('ij,ij->i',a,b)*nc +
                       np.einsum('ij,ij->i',b,c)*na + np.einsum('ij,ij->i',c,a)*nb)
        outside += int(abs(np.sum(2*np.arctan2(numerator,denominator))) < 2*np.pi)
    return outside


def evaluate(job):
    import mujoco
    wall = time.perf_counter()
    robot = scene(job['case'],job['layout'],job.get('action','center'),job.get('jitter',0.))
    sim = robot.sim
    m,d = sim.m,sim.d
    dt = m.opt.timestep
    initial_overlap_mm = max((max(0.,-c.dist)*1000 for c in d.contact),default=0.)
    initial_com = sim.package_com()
    initial_neighbors = d.xipos[sim.neighbor_bodies].copy()
    start = robot.initial_target.copy()
    gap = .095
    readings,states = [], []
    peaks = dict(penetration_mm=0.,nonfinger_mm=0.,neighbor_contact_mm=0.,self_contact_mm=0.)
    bypass_peak = 0
    preflight = job.get('preflight',False)
    outcome,reason = 'pending',''
    durations = [.25] if preflight else [.25,.8,1.8,1.5,1.,.3]
    names = ['settle'] if preflight else ['settle','approach','close','lift','carry','hold']
    last = dict(force=[0.,0.],support_N=0.)
    rest_com = initial_com.copy()
    for phase,duration in zip(names,durations):
        states.append(dict(phase=phase,t=float(d.time),qpos=d.qpos.tolist()))
        for step in range(round(duration/dt)):
            alpha = (step+1)/round(duration/dt)
            smooth = alpha*alpha*(3-2*alpha)
            target = start.copy()
            if phase == 'approach':
                target = start+(robot.grasp_target-start)*smooth
            elif phase in ('close','lift','carry','hold'):
                target = robot.grasp_target.copy()
                if phase in ('lift','carry','hold'):
                    target[2] += .15*(smooth if phase=='lift' else 1.)
                if phase in ('carry','hold'):
                    target[0] -= .1*(smooth if phase=='carry' else 1.)
                measured = min(last['force'])
                # One symmetric native gripper actuator; object label is not read.
                gap += np.clip((measured-6.)*.01,-.04,.015)*dt
                gap = float(np.clip(gap,-.010,.095))
            robot.command(target,gap)
            sim.step()
            if step%25==0:
                last = contacts(robot)
                bypass_peak = max(bypass_peak,last['direct_content_support_contacts'])
                for key in peaks:
                    peaks[key] = max(peaks[key],last[key])
                robot.tracking.append(float(np.linalg.norm(d.site_xpos[robot.site]-target)*1000))
                robot.torque_peaks = np.maximum(robot.torque_peaks,np.abs(d.actuator_force[robot.act]))
                readings.append(dict(t=float(d.time),phase=phase,com=sim.package_com().tolist(),
                                     gap=robot.gap(),**last))
                reason = sim.failure() or ''
                if reason:
                    outcome = 'sim_fail'
                    break
                if time.perf_counter()-wall > 300.:
                    outcome,reason = 'compute_limit','300 s wall limit reached; no success/failure inference'
                    break
                if np.min(sim.shell_points()[:,2]) < -.01:
                    outcome,reason = 'sim_fail','shell passed through the table'
                    break
                if not preflight and (last['penetration_mm']>(.15 if sim.info['thin'] else 1.) or
                                       last['direct_content_support_contacts']):
                    outcome,reason = 'invalid_physics','contact penetrates tolerance or bypasses shell'
                    break
        if phase=='settle':
            rest_com = sim.package_com().copy()
        if outcome!='pending':
            break
    states.append(dict(phase='end',t=float(d.time),qpos=d.qpos.tolist()))
    displacement = sim.package_com()-rest_com
    neighbors_delta = np.linalg.norm(d.xipos[sim.neighbor_bodies]-initial_neighbors,axis=1)
    # Geometric shell/content intersection supplements native contact penetration:
    # vertex-in-box detects a film node tunneling into a rigid content segment.
    inside_mm = 0.
    if sim.info['has_segment_content']:
        for g in sim.obj_geoms:
            local = (sim.shell_points()-d.geom_xpos[g]) @ d.geom_xmat[g].reshape(3,3)
            depth = np.min(m.geom_size[g]-np.abs(local),axis=1)
            inside_mm = max(inside_mm,float(max(0.,depth.max())*1000))
    warnings = {mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(d.warning) if w.number}
    escaped = outside_count(d.xipos[sim.content_bodies],sim.shell_points(),sim.info['faces'])
    mass = float(m.body_mass[sim.package_bodies].sum())
    gate_reasons = []
    if initial_overlap_mm > .15:
        gate_reasons.append('initial scene overlap exceeds 0.15 mm')
    if warnings:
        gate_reasons.append('MuJoCo warning')
    if bypass_peak:
        gate_reasons.append('content contacts support through shell')
    if escaped:
        gate_reasons.append('content centers outside shell')
    if peaks['penetration_mm']>1. or inside_mm>1.:
        gate_reasons.append('package penetration exceeds 1 mm')
    if sim.info['thin'] and (peaks['penetration_mm']>.15 or inside_mm>.15):
        gate_reasons.append('envelope penetration exceeds 0.15 mm film thickness')
    if preflight and not (.5*mass*9.81 <= last['support_N'] <= 1.5*mass*9.81):
        gate_reasons.append('settled support is not within 50% of package weight')
    if outcome=='pending':
        if preflight:
            outcome = 'preflight_pass' if not gate_reasons else 'preflight_fail'
        elif gate_reasons:
            outcome = 'invalid_physics'
        elif peaks['nonfinger_mm']>0. or peaks['self_contact_mm']>0.:
            outcome = 'robot_interference'
        elif displacement[2]<.10 or -displacement[0]<.07 or last['support_N']>.05*mass*9.81:
            outcome = 'grasp_fail'
        elif neighbors_delta.max()>.03:
            outcome = 'neighbor_disturbance'
        else:
            outcome = 'success'
        reason = '; '.join(gate_reasons)
    robot.nonfinger_penetration_mm = peaks['nonfinger_mm']
    robot.self_penetration_mm = peaks['self_contact_mm']
    robot.nonfinger_contact_samples = sum(r['nonfinger_mm']>0 for r in readings)
    return dict(**job,outcome=outcome,reason=reason,sim_time_s=float(d.time),wall_s=time.perf_counter()-wall,
                package_mass_kg=mass,displacement_mm=(displacement*1000).tolist(),
                timestep_s=dt,initial_overlap_mm=initial_overlap_mm,
                content_centers_outside=escaped,direct_content_support_contacts_peak=bypass_peak,
                initial_settle_displacement_mm=((rest_com-initial_com)*1000).tolist(),
                max_neighbor_displacement_mm=float(neighbors_delta.max()*1000),
                shell_vertex_inside_content_mm=inside_mm,peaks=peaks,warnings=warnings,
                last_contact=last,shell_sha256=sim.info['shell_sha256'],film_thickness_mm=sim.info['radius']*2000,
                shell_after=sim.shell_points().tolist(),robot=robot.diagnostics(),
                readings=readings,keyframes=states,dimensions=dict(nv=m.nv,nflex=m.nflex,nvert=m.nflexvert))


def worker(job):
    install_backend(Path(job.pop('run')))
    started = time.perf_counter()
    try:
        return evaluate(job)
    except RuntimeError as error:
        if not str(error).startswith('YAM initial pose unreachable:'):
            raise
        return dict(**job,outcome='setup_fail',reason=str(error),wall_s=time.perf_counter()-started,peaks={},
                    last_contact=dict(support_N=0.),readings=[],keyframes=[])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','preflight','contact','grasp','preview'))
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--case',choices=CASES)
    parser.add_argument('--layout',choices=LAYOUTS)
    parser.add_argument('--action',choices=('center','end','corner','side'))
    parser.add_argument('--jitter',type=float)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    if args.stage=='prepare':
        prepare(args.run)
        return
    install_backend(args.run)
    if args.stage=='preview':
        os.environ.setdefault('MUJOCO_GL','egl')
        import mujoco
        from render_probe import write_png
        robot = scene(args.case or 'book',args.layout or 'top')
        with mujoco.Renderer(robot.sim.m,600,800) as renderer:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [-.20,.05,.14]
            camera.distance=.85; camera.azimuth=135; camera.elevation=-35
            renderer.update_scene(robot.sim.d,camera=camera)
            write_png(args.run/f'preview_{args.case or "book"}_{args.layout or "top"}.png',renderer.render())
        print(json.dumps(dict(pad_offset=robot.nominal_pad_offset.tolist(),nv=robot.sim.m.nv)))
        return
    cases = [args.case] if args.case else CASES
    layouts = [args.layout] if args.layout else LAYOUTS
    jobs = [dict(case=c,layout=l,preflight=True) for c in cases for l in layouts]
    if args.stage in ('contact','grasp'):
        rows = [json.loads(line) for p in args.run.glob('preflight*.jsonl') for line in p.read_text().splitlines()]
        passed = {(r['case'],r['layout']) for r in rows if r['outcome']=='preflight_pass'}
        actions = ('center','side') if args.stage=='contact' else ('center','end','corner','side')
        jitters = (0.,) if args.stage=='contact' else (0.,.002)
        jobs = [dict(case=c,layout=l,action=a,jitter=j,preflight=False)
                for c in cases for l in layouts if (c,l) in passed
                for a in ([args.action] if args.action else actions)
                for j in ([args.jitter] if args.jitter is not None else jitters)]
    suffix = '_'.join(str(x) for x in (args.stage,args.case,args.layout,args.action,args.jitter) if x is not None)
    dest = args.run/f'{suffix}.jsonl'
    if dest.exists():
        raise FileExistsError(dest)
    from yam_probe_robot import asset_fingerprint
    save(args.run/'yam_asset_sha256.json',asset_fingerprint())
    files = [ROOT/'policy/parcel_heap.py',ROOT/'policy/yam_probe_robot.py',ROOT/'test/policy/HIDDEN_CONTENTS_REPORT.md']
    snapshot = args.run/'driver_source'/suffix
    snapshot.mkdir(parents=True)
    for file in files:
        (snapshot/file.name).write_bytes(file.read_bytes())
    import mujoco
    save(args.run/f'{suffix}_metadata.json',dict(host=platform.node(),jobs=len(jobs),bag_dt=DT,envelope_dt=.0001,
         wall_limit_s=300.,python=sys.version,mujoco=mujoco.__version__,numpy=np.__version__,
         stage=args.stage,started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
         files={str(f.relative_to(ROOT)):digest(f) for f in files}))
    with ProcessPoolExecutor(max_workers=args.workers) as pool, dest.open('w') as stream:
        futures = [pool.submit(worker,{**job,'run':str(args.run)}) for job in jobs]
        for job,future in zip(jobs,futures):
            result = future.result()
            stream.write(json.dumps(result,allow_nan=False)+'\n'); stream.flush()
            print(json.dumps({**job,'outcome':result['outcome'],'reason':result['reason'],
                  'wall_s':round(result['wall_s'],2),'peaks':result['peaks'],
                  'support_N':result['last_contact']['support_N']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

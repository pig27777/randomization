"""YAM mixed-parcel heap diagnostic; estimated materials, contact-only grasping.

The target shell is identical within each family before settling. Context parcels
are free rigid boxes. This is a small mixed heap, not a calibrated warehouse pile.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
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
SIDE_ROTATION = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]) @ TOP_ROTATION


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
            'friction=".6 .005 .0001" solref=".002 1" solimp=".99 .99 .0001" selfcollide="auto"/>'
            '</flexcomp>')


def parcel_xml(name, center, half, mass, yaw=0.):
    return (f'<body name="{name}" pos="{vector(center)}" quat="{vector([np.cos(yaw/2),0,0,np.sin(yaw/2)])}">'
            f'<freejoint name="{name}_free"/><geom name="{name}" type="box" size="{vector(half)}" '
            f'mass="{mass}" contype="1" conaffinity="7" condim="3" friction=".6 .005 .0001" '
            'solref=".002 1" solimp=".99 .99 .0001" rgba=".48 .34 .22 1"/></body>')


def scene(case, layout, action='center', jitter=0.):
    import mujoco
    import mj_adapter as A
    from model import ANCHORS, build
    from yam_probe_robot import Yam

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
    supports += [([-.40,.17,.045],[.08,.07,.045],.7),
                 ([-.16,.23,.055],[.12,.065,.055],.9),
                 ([.04,.12,.035],[.065,.08,.035],.5),
                 ([-.40,.17,.115],[.055,.05,.025],.2)]
    if layout == 'partial':
        supports.append(([CENTER[0]+.085,CENTER[1]+.018,SUPPORT_Z+height+.022],
                         [.05,.038,.02],.15))
    for i, (pos, half, mass) in enumerate(supports):
        name = f'parcel_{i}'
        neighbors.append(name)
        fragments.append(parcel_xml(name,pos,half,mass,0. if i<len(supports)-1 else .08))
    built = A.scene(desc,origin,pads=(),dt=DT,worldbody=''.join(fragments),
                    pad_contact=False,table_contact=False,solver='Newton')
    root = ET.fromstring(built.xml)
    root.find('option').set('integrator','discrete')
    root.find('option').set('iterations','100')
    root.find('size').set('memory','128M')
    if not has_segment_content:
        dummy = root.find("worldbody/body[@name='obj']")
        dummy.set('gravcomp','1')
        for geom in dummy.iter('geom'):
            geom.set('contype','0'); geom.set('conaffinity','0'); geom.set('rgba','0 0 0 0')
    built.xml = ET.tostring(root, encoding='unicode')
    built.heap_info = dict(case=case,layout=layout,thin=thin,radius=radius,
                           shell_initial=shell,faces=faces,neighbors=neighbors,
                           has_segment_content=has_segment_content,height=height,
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
    elif action != 'center':
        raise ValueError(action)
    grasp[0] += jitter
    rotation = SIDE_ROTATION if action == 'side' else TOP_ROTATION
    start = grasp + ([0.,-.07,0.] if action == 'side' else [0.,0.,.09])
    robot = Yam(k,0,fixture=dict(built=built,sim_factory=HeapSim,target=start,rotation=rotation))
    robot.grasp_target = grasp
    return robot


def install_backend(run):
    load_backend(run)
    # Define the subclass after selecting and verifying the frozen release.
    global HeapSim
    import mj_adapter as A

    class HeapSim(A.Sim):
        def __init__(self,built):
            super().__init__(built)
            self.info = built.heap_info
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
    support = penetration = nonfinger = neighbor_touch = 0.
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
    # Robot-neighbor contact is independent of whether the target participates.
    for c in d.contact:
        a,b = c.geom
        if c.efc_address>=0 and a>=0 and b>=0:
            if ((robot.robot_mask[a] and b in sim.neighbor_geoms) or
                    (robot.robot_mask[b] and a in sim.neighbor_geoms)):
                neighbor_touch = max(neighbor_touch,max(0.,-c.dist))
    return dict(force=normals.tolist(),support_N=float(support),penetration_mm=float(penetration*1000),
                nonfinger_mm=float(nonfinger*1000),neighbor_contact_mm=float(neighbor_touch*1000))


def evaluate(job):
    import mujoco
    wall = time.perf_counter()
    robot = scene(job['case'],job['layout'],job.get('action','center'),job.get('jitter',0.))
    sim = robot.sim
    m,d = sim.m,sim.d
    initial_com = sim.package_com()
    initial_neighbors = d.xipos[sim.neighbor_bodies].copy()
    start = robot.initial_target.copy()
    gap = .095
    readings,states = [], []
    peaks = dict(penetration_mm=0.,nonfinger_mm=0.,neighbor_contact_mm=0.)
    preflight = job.get('preflight',False)
    outcome,reason = 'pending',''
    durations = [.25] if preflight else [.25,.8,1.8,1.5,1.,.3]
    names = ['settle'] if preflight else ['settle','approach','close','lift','carry','hold']
    last = dict(force=[0.,0.],support_N=0.)
    rest_com = initial_com.copy()
    for phase,duration in zip(names,durations):
        states.append(dict(phase=phase,t=float(d.time),qpos=d.qpos.tolist()))
        for step in range(round(duration/DT)):
            alpha = (step+1)/round(duration/DT)
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
                gap += np.clip((measured-6.)*.01,-.04,.015)*DT
                gap = float(np.clip(gap,-.010,.095))
            robot.command(target,gap)
            sim.step()
            if step%25==0:
                last = contacts(robot)
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
                if np.min(sim.shell_points()[:,2]) < -.01:
                    outcome,reason = 'sim_fail','shell passed through the table'
                    break
        if phase=='settle':
            rest_com = sim.package_com().copy()
        if outcome=='sim_fail':
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
    mass = float(m.body_mass[sim.package_bodies].sum())
    gate_reasons = []
    if warnings:
        gate_reasons.append('MuJoCo warning')
    if peaks['penetration_mm']>1. or inside_mm>1.:
        gate_reasons.append('package penetration exceeds 1 mm')
    if sim.info['thin'] and (peaks['penetration_mm']>.15 or inside_mm>.15):
        gate_reasons.append('envelope penetration exceeds 0.15 mm film thickness')
    if preflight and not (.5*mass*9.81 <= last['support_N'] <= 1.5*mass*9.81):
        gate_reasons.append('settled support is not within 50% of package weight')
    if outcome!='sim_fail':
        if preflight:
            outcome = 'preflight_pass' if not gate_reasons else 'preflight_fail'
        elif gate_reasons:
            outcome = 'invalid_physics'
        elif peaks['nonfinger_mm']>0.:
            outcome = 'robot_interference'
        elif displacement[2]<.10 or -displacement[0]<.07 or last['support_N']>.05*mass*9.81:
            outcome = 'grasp_fail'
        elif neighbors_delta.max()>.03:
            outcome = 'neighbor_disturbance'
        else:
            outcome = 'success'
        reason = '; '.join(gate_reasons)
    return dict(**job,outcome=outcome,reason=reason,sim_time_s=float(d.time),wall_s=time.perf_counter()-wall,
                package_mass_kg=mass,displacement_mm=(displacement*1000).tolist(),
                initial_settle_displacement_mm=((rest_com-initial_com)*1000).tolist(),
                max_neighbor_displacement_mm=float(neighbors_delta.max()*1000),
                shell_vertex_inside_content_mm=inside_mm,peaks=peaks,warnings=warnings,
                last_contact=last,shell_sha256=sim.info['shell_sha256'],film_thickness_mm=sim.info['radius']*2000,
                shell_after=sim.shell_points().tolist(),robot=robot.diagnostics(),
                readings=readings,keyframes=states,dimensions=dict(nv=m.nv,nflex=m.nflex,nvert=m.nflexvert))


def worker(job):
    install_backend(Path(job.pop('run')))
    return evaluate(job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','preflight','grasp','preview'))
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--case',choices=CASES)
    parser.add_argument('--layout',choices=LAYOUTS)
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
    if args.stage=='grasp':
        rows = [json.loads(line) for p in args.run.glob('preflight*.jsonl') for line in p.read_text().splitlines()]
        passed = {(r['case'],r['layout']) for r in rows if r['outcome']=='preflight_pass'}
        jobs = [dict(case=c,layout=l,action=a,jitter=j,preflight=False)
                for c in cases for l in layouts if (c,l) in passed
                for a in ('center','end','corner','side') for j in (0.,.002)]
    suffix = '_'.join(x for x in (args.stage,args.case,args.layout) if x)
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
    save(args.run/f'{suffix}_metadata.json',dict(host=platform.node(),jobs=len(jobs),dt=DT,
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

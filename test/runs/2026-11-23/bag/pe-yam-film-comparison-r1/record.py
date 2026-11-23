"""Record right YAM shared joint commands and native two-sheet dynamics."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT/'experience'), str(ROOT/'test/bag_model')]
from bag_sim.physical_parcel import Config, Parcel, parcel_profile, vec
from experience.yam_platform.protocol import attach_robot
from experience.yam_platform.robot import asset_fingerprint
from physical_parcel_yam import Config as Trial, ParcelSim, compile_yam_model, measure

CENTER = np.array([-.03, -.225])
DURATION = 9.6
BOX = (.085, .016, .038)
SIZE = (.18, .14)
MASS = .065
DT = .0001
CONTACT_TIME = .0004
PAYLOAD_MARGIN = 0.0
BASE = json.loads((ROOT/'configs/route_b_yam_scene.json').read_text())['arm_bases_m'][1]

def make_scene():
    shell = parcel_profile('pe')['shell']
    film_mass = 2*np.prod(SIZE)*shell['areal_mass']
    p = Parcel(Config(mode='bendable', geometry='geometry/kraft_edge_sealed_35.npz',
        bag_size=SIZE, payload_size=BOX, payload_mass=MASS-film_mass, center=tuple(CENTER),
        timestep=DT, contact_time=CONTACT_TIME, friction=.35, **shell))
    assert abs(p.envelope['masses'].sum()-film_mass)<1e-8
    root = ET.fromstring(p.xml)
    # Fold segments follow the long side of the textile. Turn the entire packed asset by 90 degrees.
    rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    pivot=np.r_[CENTER,0.]
    def turn(points):
        return (points-pivot)@rot.T+pivot
    for key in ('reference','current','payload_origin'):
        p.envelope[key]=turn(p.envelope[key])
    root.find('.//flexcomp').set('point',vec(p.envelope['reference']))
    if PAYLOAD_MARGIN:
        for geom in root.findall('.//geom'):
            if geom.get('name','').startswith(('payload_geom_', 'plate_geom_')):
                geom.set('margin',str(PAYLOAD_MARGIN))
    first=root.find(".//body[@name='payload_0']")
    first.set('pos',vec(turn(np.fromstring(first.get('pos'),sep=' '))))
    first.set('quat',vec([np.sqrt(.5),0.,0.,np.sqrt(.5)]))
    world = root.find('worldbody')
    world.remove(world.find("body[@name='contact_probe']"))
    # Finite ABC tabletop. The YAM base stands beside its near edge.
    desk = world.find("geom[@name='table']")
    for k,v in dict(type='box',pos='0 0 -.005',size='.2975 .65 .005',rgba='.56 .58 .58 1',conaffinity='6').items():
        desk.set(k,v)
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    ET.SubElement(root.find('asset'),'mesh',name='table_box',vertex=vec(corners*np.array([.2975,.65,.005])))
    desk.set('type','mesh');desk.set('mesh','table_box');desk.attrib.pop('size')
    # Plane/triangle contact over the entire swept parcel area; the CAD arm still sees the finite desk.
    desk.set('contype','16');desk.set('conaffinity','5')
    ET.SubElement(world,'geom',name='floor',type='plane',size='1 1 .01',pos='0 0 0',
                  contype='0',conaffinity='2',friction='.35 .005 .0001',rgba='0 0 0 0')
    root.find('option').set('ccd_tolerance','1e-9')
    cfg = Trial(dt=DT, tilt=0., yaw=-90., contact_time=CONTACT_TIME, pickup_only=True)
    built = SimpleNamespace(xml=ET.tostring(root,encoding='unicode'),desc=SimpleNamespace(),cfg=cfg,parcel=p)
    def compile_workcell(xml):
        tree = ET.fromstring(xml)
        arm = tree.find(".//body[@name='arm']")
        arm.set('pos',vec(BASE))
        arm.set('quat','1 0 0 0')
        w = tree.find('worldbody')
        w.remove(w.find("geom[@name='yam_pedestal']"))
        return compile_yam_model(p,built,ET.tostring(tree,encoding='unicode'))
    built.compile_model = compile_workcell
    return attach_robot(cfg,built,ParcelSim,CENTER,.075,np.r_[SIZE,BOX[2]],MASS)

def pad_faces(robot,data):
    faces=[]
    for name in ('left_pad','right_pad'):
        body=data.body(name)
        faces.append(body.xpos+body.xmat.reshape(3,3)@np.array([0.,0.,-.1444]))
    return np.stack(faces)

def ease(t,start,end):
    u=np.clip((t-start)/(end-start),0.,1.)
    return u**3*(10.+u*(-15.+6.*u))

def run(out, seconds,depth, end_gap, replay, rail_servo=False):
    out.mkdir()
    r=make_scene()
    m,d=r.sim.m,r.sim.d
    p=r.sim.parcel
    faces0=pad_faces(r,d)
    z0=faces0[:,2].mean()
    print('tip positions',faces0,'start',r.initial_target,'q',r.qtarget,flush=True)
    count=round(DURATION/.01)
    q=[];gaps=[]
    down=r.rotation.copy()
    for i in range(count+1):
        t=i*.01
        target=r.initial_target.copy()
        target[2]+=(depth-z0)*ease(t,.15,1.55)+.285*ease(t,3.45,5.95)
        # Move toward the base as the elbow extends at the top.
        target[0]-=.045*ease(t,3.45,5.95)
        pitch=np.deg2rad(-36)*ease(t,3.45,5.95)
        c,s=np.cos(pitch),np.sin(pitch)
        r.rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])@down
        r.solve(target)
        q.append(r.qtarget.copy())
        gaps.append(.092+(end_gap-.092)*ease(t,1.65,3.2))
    ik=max(e['position_mm'] for e in r.ik_errors)
    if ik>1.:raise RuntimeError(f'IK error {ik:.3f} mm')
    q=np.array(q); gaps=np.array(gaps)
    if not replay:
        np.savez_compressed(out/'q_ref.npz',time=np.arange(count+1)*.01,q=q,gap=gaps)
    d.qpos[r.qadr]=q[0];d.qpos[r.fingers]=gaps[0]/2
    mj.mj_forward(m,d)
    body=np.arange(m.body('arm').id,m.nbody)
    film=m.flex_vertbodyid
    bodies=np.r_[film,r.sim.payload_bodies]
    weights=m.body_mass[bodies]
    weights=weights/weights.sum()
    values={k:[] for k in 'time V links link_quat com velocity clearance grip support'.split()}
    pad_path=[];q_path=[]
    faces=p.envelope['faces']
    mj.mj_saveModel(m,str(out/'model.mjb'))
    (out/'model.xml').write_text(r.sim.built.xml)
    start=time.perf_counter()
    n=round(seconds/DT)
    proxy_stride=round(.002/DT); frame_stride=round(.04/DT); ctrl_stride=round(.01/DT)
    if replay:
        locked=np.load(replay)
        joint=locked['q_path']
        exact='q_native' in locked
        if exact:
            locked_q=locked['q_native']; locked_v=locked['dq_native']; locked_control=locked['control']
        if len(joint)<1+round(seconds/.002):raise ValueError('incompatible shared clock')
        robot_qpos=np.r_[r.qadr,r.fingers]
        robot_dofs=np.r_[r.dofs,[m.joint(n+'_finger').dofadr[0] for n in ('left','right')]]
    from physical_parcel_checks import film_payload_geometry
    diagnostics=[]
    clearance_checks=[]
    joint_history=[]
    joint_rates=[]
    controls=[]
    stable=[]
    rigid_meshes=[]
    for geom in range(m.ngeom):
        if m.geom(geom).name.startswith(('payload_geom_','plate_geom_')):
            mesh=m.geom_dataid[geom]
            v=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
            rigid_meshes.append((geom,v.copy()))
    table=m.geom('table').id
    finger_masks=np.zeros((2,m.ngeom+1),dtype=bool)
    for j,group in enumerate(r.all_finger_sets):
        finger_masks[j,list(group)]=True
    for i in range(n+1):
        t=i*DT
        if replay:
            loc,rem=divmod(i,proxy_stride)
            delta=joint[min(loc+1,len(joint)-1)]-joint[loc]
            boundary=joint[loc]+rem/proxy_stride*delta
            dq=delta/.002
            if exact:
                boundary=locked_q[i]
                dq=locked_v[i]
            controlled=slice(None,6) if rail_servo else slice(None)
            d.qpos[robot_qpos[controlled]]=boundary[controlled]
            d.qvel[robot_dofs[controlled]]=dq[controlled]
        if replay:
            joint_history.append(d.qpos[robot_qpos].copy())
            joint_rates.append(d.qvel[robot_dofs].copy())
        if i%proxy_stride==0:
            mj.mj_kinematics(m,d)
            pad_path.append(pad_faces(r,d))
            q_path.append(d.qpos[np.r_[r.qadr,r.fingers]].copy())
        if i%frame_stride==0:
            mj.mj_forward(m,d)
            st=measure(r)
            bad_table=[];self_contact=[]
            for contact in d.contact:
                a,b=contact.geom
                if min(a,b)<0 or contact.efc_address<0 or contact.dist>=0: continue
                if table in (a,b) and r.robot_mask[b if a==table else a]:
                    bad_table.append(-contact.dist)
                if r.robot_mask[a] and r.robot_mask[b]:
                    self_contact.append(-contact.dist)
            bottom=min((points@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()
                       for g,points in rigid_meshes)
            diagnostics.append([t,st['penetration_mm'],st['nonfinger_mm'],max(bad_table,default=0.),
                                max(self_contact,default=0.),bottom])
            if replay:
                enclosed=film_payload_geometry(m,d,p.envelope['faces'])
                clearance_checks.append([t,enclosed['outside_corners'],enclosed['geometric_depth_upper_m']])
            verts=d.flexvert_xpos.copy()
            com=weights@d.xipos[bodies]
            velocities=np.empty((len(bodies),6))
            for j,b in enumerate(bodies):
                mj.mj_objectVelocity(m,d,mj.mjtObj.mjOBJ_BODY,int(b),velocities[j],0)
            for k,v in dict(time=t,V=verts,links=d.xpos[body],link_quat=d.xquat[body],com=com,
                    velocity=weights@velocities[:,3:], clearance=verts[:,2].min()-p.cfg.radius,
                    grip=st['grip_each_N'],support=st['support_N']).items():
                values[k].append(np.copy(v))
            if i==0:print('measure keys',st.keys(),flush=True)
            if i%round(1/DT)==0:print(t,'z,min, gap, forces',com[2],values['clearance'][-1],r.gap(),st,flush=True)
            if d.warning.number.any():raise ValueError(f'native warning at {t:.3f}')
        if i==n:break
        j=i//ctrl_stride; u=i%ctrl_stride/ctrl_stride
        target=q[j]+u*(q[j+1]-q[j])
        gap=gaps[j]+u*(gaps[j+1]-gaps[j])
        ff=(d.qfrc_bias[r.dofs]-d.qfrc_gravcomp[r.dofs])/m.actuator_gainprm[r.act,0]
        if replay:target=boundary[:6]
        d.ctrl[r.act]=np.clip(target+ff,*m.actuator_ctrlrange[r.act].T)
        d.ctrl[r.gripper]=end_gap/2 if rail_servo and t>2.93 else boundary[-1] if replay else gap/2
        if replay:
            if exact:d.ctrl[:]=locked_control[i]
            controls.append(d.ctrl.copy())
        r.sim.step()
        if replay and t>=5.12:
            previous=weights@d.xipos[bodies]
            mj.mj_kinematics(m,d)
            current=weights@d.xipos[bodies]
            speed=np.linalg.norm(current-previous)/DT
            ct=d.contact
            pos=ct.efc_address>=0
            live=(ct.flex>=0).any(1)&pos&(d.efc_force[np.maximum(ct.efc_address,0)]>0.) if d.nefc else pos
            touched=finger_masks[:,ct.geom[live]].any(axis=(1,2))
            stable.append([t+DT,d.xpos[film,2].min()-p.cfg.radius,speed,*touched])
    wall=time.perf_counter()-start
    np.savez_compressed(out/'shell.npz',F=faces,body_names=[m.body(int(b)).name for b in body],
       **{k:np.asarray(v) for k,v in values.items()},pad_path=pad_path,q_path=q_path,diagnostics=diagnostics,enclosure=clearance_checks,hold_audit=stable)
    if replay:
        np.savez_compressed(out/'measured_joints.npz',q=np.asarray(joint_history),dq=np.asarray(joint_rates),control=np.asarray(controls),dt=DT)
    summary=dict(engine='MuJoCo flex',dt=DT,contact_time=CONTACT_TIME,payload_margin=PAYLOAD_MARGIN,steps=n,wall_s=wall,steps_s=n/wall,mass_kg=float(m.body_mass[bodies].sum()),
        box_m=BOX,bag_sheet_m=SIZE,cfg=p.metadata,ik_max_mm=ik,robot_assets=asset_fingerprint())
    (out/'shell.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path);ap.add_argument('--seconds',type=float,default=DURATION)
    ap.add_argument('--depth',type=float,default=.005);ap.add_argument('--gap',type=float,default=-.006)
    ap.add_argument('--replay',type=Path)
    ap.add_argument('--rail-servo',action='store_true')
    ap.add_argument('--dt',type=float,default=DT);ap.add_argument('--contact-time',type=float,default=CONTACT_TIME)
    ap.add_argument('--payload-margin',type=float,default=0.)
    a=ap.parse_args();DT=a.dt;CONTACT_TIME=a.contact_time;PAYLOAD_MARGIN=a.payload_margin
    run(a.output,a.seconds,a.depth,a.gap,a.replay,a.rail_servo)

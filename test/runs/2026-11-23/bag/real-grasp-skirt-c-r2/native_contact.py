"""MuJoCo 4 dual-flex centre grasp with moving contact bodies and no grasp equality."""
import argparse
from itertools import permutations, product
from pathlib import Path
import sys
import time
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"umi-skirt-c-r1"))
from model_c import CENTER, CLOTH, MASS, BAG, lengths, vec
from limiter import compile_limit

HALF_PAD=np.array([.010,.004,.020])


def textile():
    width=np.array([-.1225,-.071,-.039,-.015,0.,.015,.039,.071,.1225])
    length=np.array([-.1725,-.058,-.010,0.,.010,.058,.1725])
    xx,yy,layer=np.meshgrid(width,length,[0.,1.],indexing="ij")
    rim=np.maximum((abs(xx)-.105)/.0175,(abs(yy)-.149)/.0235).clip(0,1)
    foot=.0008+rim*.0054
    thickness=.0190-rim*.0108
    p=np.column_stack((xx.ravel()+CENTER[0],yy.ravel()+CENTER[1],(foot+layer*thickness).ravel()))
    grid=np.arange(xx.size).reshape(xx.shape)
    tet=[]
    for cell in product(range(len(width)-1),range(len(length)-1),range(1)):
        for order in permutations(range(3)):
            bits=np.asarray(cell).copy()
            simplex=[grid[tuple(bits)]]
            for axis in order:
                bits[axis]+=1
                simplex.append(grid[tuple(bits)])
            tet.append(simplex)
    tet=np.asarray(tet)
    a,b,c,d=np.moveaxis(p[tet],1,0)
    inv=np.einsum("ij,ij->i",np.cross(b-a,c-a),d-a)<0
    tet[inv,1],tet[inv,2]=tet[inv,2].copy(),tet[inv,1].copy()
    tris=tet[:,[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]].reshape(-1,3)
    _,start,count=np.unique(np.sort(tris,axis=1),axis=0,return_index=True,return_counts=True)
    return p,tet,tris[start[count==1]]


def sheets():
    # The cloth is wider than the jaw path. Resolve material under each
    # approaching finger and use larger elements near the outer seal.
    x=np.array([-.145,-.125,-.079,-.039,-.019,0.,.019,.039,.079,.125,.145])
    y=np.array([-.195,-.174,-.058,-.012,0.,.012,.058,.174,.195])
    X,Y=np.meshgrid(x,y,indexing="ij")
    grid=np.arange(X.size).reshape(X.shape)
    seam=(abs(X)==.145)|(abs(Y)==.195)
    lower=grid.copy()
    lower[~seam]=X.size+np.arange((~seam).sum())
    bend=np.maximum(abs(X)-.123,abs(Y)-.173).clip(0,.022)
    sin=.01015/.022
    first=np.stack((X+CENTER[0]-np.sign(X)*bend*(1-np.sqrt(1-sin**2)),
                    Y+CENTER[1]-np.sign(Y)*bend*(1-np.sqrt(1-sin**2)),
                    .0207-bend*sin),axis=-1)
    other=first.copy()
    other[...,2]=.0004+bend*sin
    current=np.concatenate((first.reshape(-1,3),other[~seam]))
    flat=np.concatenate((np.stack((X+CENTER[0],Y+CENTER[1],X*0),-1).reshape(-1,3),
                         np.stack((X[~seam]+CENTER[0],Y[~seam]+CENTER[1],X[~seam]*0),-1)))
    top_faces=[]
    bottom_faces=[]
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            for u,g in ((top_faces,grid),(bottom_faces,lower)):
                a,b,c,d=g[i,j],g[i+1,j],g[i+1,j+1],g[i,j+1]
                if not (i+j)%2:
                    u.extend(((a,b,c),(a,c,d)))
                else:
                    u.extend(((a,b,d),(b,c,d)))
    faces=np.r_[np.array(top_faces),np.array(bottom_faces)[:,::-1]]
    edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[0,2]],faces[:,[1,2]])),axis=1),axis=0)
    return flat,current,faces,edges


def make_model(dt,mu,mass,young):
    ref,film,F,E=sheets()
    cloth,tets,C=textile()
    root=ET.Element("mujoco")
    ET.SubElement(root,"compiler",angle="radian",autolimits="true")
    ET.SubElement(root,"size",memory="600M")
    ET.SubElement(root,"option",timestep=str(dt),integrator="discrete",solver="Newton",
        jacobian="sparse",iterations="40",ls_iterations="14",tolerance="4e-7",ccd_tolerance="1e-7",ccd_iterations="25")
    world=ET.SubElement(root,"worldbody")
    contact_time=max(.00155,4*dt)
    contact=dict(solref=f"{contact_time} 1",solimp=".995 .999 .00035",condim="3")
    ET.SubElement(world,"geom",name="table",type="plane",size="1 1 .01",contype="1",conaffinity="6",
        friction=".38 .005 .0001",**contact)
    for side in ("left","right"):
        body=ET.SubElement(world,"body",name=side,gravcomp="1")
        ET.SubElement(body,"freejoint",name=f"{side}_boundary")
        ET.SubElement(body,"geom",name=f"{side}_rubber",type="box",size=vec(HALF_PAD),mass="100.",
            contype="8",conaffinity="6",priority="2",friction=f"{mu} .006 .00015",**contact)
    film_mass=2*np.prod(BAG)*.0552
    shell=ET.SubElement(world,"flexcomp",name="film",type="direct",dim="2",point=vec(film),element=vec(F),
                       mass=str(film_mass),radius=".00030")
    ET.SubElement(shell,"elasticity",young="20000000",poisson=".3",thickness=".00006",elastic2d="bend",damping=".000045")
    ET.SubElement(shell,"contact",contype="2",conaffinity="13",selfcollide="auto",friction=".66 .005 .0001",**contact)
    fill=ET.SubElement(world,"flexcomp",name="textile",type="direct",dim="3",dof="full",point=vec(cloth),
                      element=vec(tets),mass=str(mass-film_mass),radius=".00045")
    ET.SubElement(fill,"elasticity",young=str(young),poisson=".21",damping=".0015")
    ET.SubElement(fill,"contact",contype="4",conaffinity="11",selfcollide="none",
                 friction=".63 .005 .0001",**contact)
    model=mj.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    assert model.neq==0 and model.nmocap==0
    return model,ref,film,F,E,cloth,C


def replay(source,out,seconds,dt,mu,mass,young,pre_settled):
    recorded=np.load(source)
    centers=recorded["pad_box_centers"]
    rotations=recorded["pad_box_rotation"]
    orientation=np.empty(centers.shape[:2]+(4,))
    for i,q in enumerate(orientation):
        for j in range(2):
            mj.mju_mat2Quat(q[j],rotations[i,j].ravel())
    m,ref,initial,F,E,cloth,C=make_model(dt,mu,mass,young)
    d=mj.MjData(m)
    joint=[m.joint(f"{side}_boundary").id for side in ("left","right")]
    pos=m.jnt_qposadr[joint][:,None]+np.arange(7)
    vel=m.jnt_dofadr[joint][:,None]+np.arange(6)
    pads=[m.geom(f"{side}_rubber").id for side in ("left","right")]
    film_count=len(initial)
    vertex_bodies=m.flex_vertbodyid[:film_count]
    node_index=m.body_jntadr[vertex_bodies]
    qpos=m.jnt_qposadr[node_index][:,None]+np.arange(3)
    dof=m.jnt_dofadr[node_index][:,None]+np.arange(3)
    body_ids=m.flex_vertbodyid
    weights=m.body_mass[body_ids]
    L0=np.ascontiguousarray(np.minimum(lengths(ref,E),lengths(initial,E)))
    E=np.ascontiguousarray(E,dtype=np.int64)
    window=round(.002/dt)
    assert abs(window*dt-.002)<1e-10
    frames={x:[] for x in ("time","film","contents","normals","shear","support","clearance","com","ratio","pad_overlap_mm")}
    scratch,limit=compile_limit()
    mobility=np.ascontiguousarray(1/m.body_mass[vertex_bodies])
    num=round(seconds/dt)
    peak=0.
    begin=time.perf_counter()
    for step in range(num+1):
        # Before first possible jaw contact, an already settled specimen can
        # be replayed without integrating identical, unloaded time steps.
        if pre_settled and .48+dt/2<step*dt<2.12-dt/2:
            if step%round(.04/dt)==0:
                for key in frames:
                    frames[key].append(step*dt if key=="time" else np.copy(frames[key][-1]))
            continue
        idx,k=divmod(step,window)
        i=min(idx,len(centers)-2)
        rate=(k/window if idx==i else 1.)
        d.qpos[pos[:,:3]]=centers[i]*(1-rate)+centers[i+1]*rate
        d.qvel[vel[:,:3]]=(centers[i+1]-centers[i])/.002
        for j in range(2):
            angular=np.empty(3)
            mj.mju_subQuat(angular,orientation[i+1,j],orientation[i,j])
            d.qvel[vel[j,3:]]=angular/.002
            quat=orientation[i,j].copy()
            mj.mju_quatIntegrate(quat,angular,rate)
            d.qpos[pos[j,3:]]=quat
        v=np.ascontiguousarray(d.qpos[qpos]+initial)
        previous=v.copy()
        limit(v,E,L0,mobility,len(E))
        peak=max(peak,float((lengths(v,E)/L0).max()))
        correction=v-previous
        if np.any(correction):
            d.qpos[qpos]+=correction
            direct=correction/np.maximum(np.linalg.norm(correction,axis=1,keepdims=True),1e-15)
            qv=d.qvel[dof]
            d.qvel[dof]=qv-direct*np.minimum(np.sum(qv*direct,axis=1,keepdims=True),0.)
        if step%round(.04/dt)==0:
            mj.mj_forward(m,d)
            normals=np.zeros(2)
            shear=np.zeros(2)
            support=0.
            overlap=0.
            force=np.empty(6)
            for c_index,c in enumerate(d.contact):
                if c.efc_address<0:continue
                if not (c.flex>=0).any():continue
                mj.mj_contactForce(m,d,c_index,force)
                if 0 in c.geom:support+=force[0]
                for j,pad in enumerate(pads):
                    if pad in c.geom:
                        normals[j]+=force[0]
                        shear[j]+=abs(force[1:3]@c.frame.reshape(3,3)[1:,2])
                        overlap=max(overlap,-float(c.dist)*1000)
            com=np.average(d.xipos[body_ids],axis=0,weights=weights)
            values=dict(time=step*dt,film=d.flexvert_xpos[:film_count],contents=d.flexvert_xpos[film_count:],
                normals=normals,shear=shear,support=support,clearance=d.flexvert_xpos[:film_count,2].min(),
                com=com,ratio=(lengths(d.flexvert_xpos[:film_count],E)/L0).max(),pad_overlap_mm=overlap)
            for key in frames:frames[key].append(np.copy(values[key]))
            if step%round(.2/dt)==0:
                print(f"{values['time']:.2f} zmin={values['clearance']:.4f} com={com.round(5)} force={normals.round(2)} "
                    f"vertical_shear={shear.round(2)} support={support:.2f} ratio={values['ratio']:.4f} pad_overlap={overlap:.2f}",flush=True)
        if step==num:break
        d.qfrc_applied[dof]=-.0025*d.qvel[dof]
        mj.mj_step(m,d)
        if d.warning.number.any():
            raise RuntimeError(f"Native integration warning at {step*dt:.4f} s: {d.warning.number}")
    np.savez_compressed(out,**frames,faces=F,edges=E,ref=ref,initial=initial,content_faces=C,content_initial=cloth,
         wall_s=time.perf_counter()-begin,steps=num,dt=dt,max_ratio=peak,mu=mu,mass=mass,young=young,native_eq_count=m.neq,pre_settled=pre_settled)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--seconds",type=float,default=9.6)
    p.add_argument("--dt",type=float,default=.0004)
    p.add_argument("--mu",type=float,default=1.3)
    p.add_argument("--mass",type=float,default=.5)
    p.add_argument("--young",type=float,default=14500.)
    p.add_argument("--pre-settled",action="store_true")
    a=p.parse_args()
    replay(a.source,a.output,a.seconds,a.dt,a.mu,a.mass,a.young,a.pre_settled)

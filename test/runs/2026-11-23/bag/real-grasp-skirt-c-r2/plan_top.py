"""With the platform's native YAM IK, plan a vertical centre pinch at 2 ms."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"pe-yam-film-comparison-r1"))
from record import CENTER, ease, make_scene, pad_faces, mj

OPEN = .078
TOP = .020


def plan(out, final_gap, press_depth, yaw_trim=0.,lift_pitch=-32.):
    robot=make_scene()
    model,kin=robot.sim.m,robot.kin
    ids=np.r_[robot.qadr,robot.fingers]
    reference=robot.rotation.copy()
    phi=np.deg2rad(yaw_trim)
    c,s=np.cos(phi),np.sin(phi)
    robot.rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]]) @ reference
    robot.solve(robot.initial_target)
    kin.qpos[ids]=np.r_[robot.qtarget,OPEN/2,OPEN/2]
    mj.mj_kinematics(model,kin)
    offset=pad_faces(robot,kin).mean(0)-robot.initial_target
    vertical=robot.rotation.copy()
    initial=robot.initial_target+offset
    initial[:2]=CENTER+[-.042,-.024]
    initial[2]=.150
    ticks=np.arange(4801)*.002
    q=[]
    for t in ticks[::5]:
        approach=ease(t,0,1.5)
        press=ease(t,1.5,2.6)
        close=ease(t,2.6,4.)
        lift=ease(t,4.,6.2)
        target=initial+(np.r_[CENTER,TOP+.050]-initial)*approach
        target[2]-=(.050+press_depth)*press
        target[2]+=.300*lift
        tilt=np.deg2rad(lift_pitch)*ease(t,4.2,5.8)
        cosine,sine=np.cos(tilt),np.sin(tilt)
        frame=np.array([[cosine,0,sine],[0,1,0],[-sine,0,cosine]])
        robot.rotation=frame@vertical
        result=robot.solve(target-frame@offset)
        if result["position_mm"]>.8:
            raise RuntimeError(f"IK {t:.2f}s {result['position_mm']:.2f} mm")
        gap=OPEN+(final_gap-OPEN)*close
        q.append(np.r_[robot.qtarget,gap/2,gap/2])
    q=np.column_stack([np.interp(ticks,ticks[::5],col) for col in np.asarray(q).T])
    names=np.asarray([*(f"joint{j}" for j in range(1,7)),"left_finger","right_finger"])
    pads, centres, rotation, link_pos, link_quat=[],[],[],[],[]
    bid=np.arange(model.body("arm").id,model.nbody)
    for state in q:
        kin.qpos[ids]=state
        mj.mj_kinematics(model,kin)
        pp=pad_faces(robot,kin)
        rr=np.stack([kin.body(f"{k}_pad").xmat.reshape(3,3).copy() for k in ("left","right")])
        pads.append(pp)
        centres.append(np.stack([pp[k]+rr[k]@[0.,-.004,.020] for k in range(2)]))
        rotation.append(rr)
        link_pos.append(kin.xpos[bid].copy())
        link_quat.append(kin.xquat[bid].copy())
    pp=np.asarray(pads)
    print("pads at 0 / 1.5 / 2.5 / 3.6 / 4.8 / 8",pp[[0,750,1250,1800,2400,4000]].round(5))
    np.savez_compressed(out,time=ticks,q_path=q,joint_names=names,pads=pp,
        pad_box_centers=centres,pad_box_rotation=rotation,
        links=link_pos,quats=link_quat,names=[model.body(int(b)).name for b in bid])
    data={"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"pad_bottom_min_mm":float(pp[:,:,2].min()*1000),
          "gap_initial_mm":OPEN*1000,"gap_closed_mm":final_gap*1000,
          "max_horizontal_center_error_mm":float(np.linalg.norm(pp[750:].mean(1)[:,:2]-CENTER,axis=1).max()*1000),
          "press_depth_mm":press_depth*1000,"steps":4800}
    out.with_suffix(".json").write_text(json.dumps(data,indent=2)+"\n")
    print(data)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("--gap",type=float,default=.009)
    p.add_argument("--depth",type=float,default=.012)
    p.add_argument("--yaw-trim",type=float,default=0.)
    p.add_argument("--lift-pitch",type=float,default=-32.)
    a=p.parse_args()
    plan(a.out,a.gap,a.depth,a.yaw_trim,a.lift_pitch)

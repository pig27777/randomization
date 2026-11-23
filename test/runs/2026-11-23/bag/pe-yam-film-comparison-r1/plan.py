"""Generate a shared YAM kinematic boundary with platform IK and FK."""
import argparse
import numpy as np
from record import make_scene, pad_faces, ease, DURATION, Path
def plan(out, depth, gap, down_end, offset_y, carry_gap, lift_height, pitch_deg=-32.):
    r=make_scene()
    t=np.arange(round(DURATION/.002)+1)*.002
    rest=r.rotation.copy();target0=r.initial_target.copy()
    offset=pad_faces(r,r.sim.d).mean(0)-target0
    qc=[]; check=[]
    for ti in t[::5]:
        approach=ease(ti,.15,down_end)
        lift=ease(ti,3.15,5.15)
        pitch=np.deg2rad(pitch_deg)*lift
        c,s=np.cos(pitch),np.sin(pitch)
        tilt=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        r.rotation=tilt@rest
        center=target0+offset
        center[1]+=offset_y*ease(ti,.15,1.35)
        center[2]+=(depth-center[2])*approach+lift_height*lift
        center[0]-=.065*lift
        result=r.solve(center-tilt@offset)
        check.append(result['position_mm'])
        rail=.5*(.092+(gap-.092)*ease(ti,1.56,2.93))
        if carry_gap is not None:rail+=.5*(carry_gap-gap)*ease(ti,3.12,3.92)
        qc.append(np.r_[r.qtarget,rail,rail])
    if max(check)>1.:raise ValueError(max(check))
    q=np.stack([np.interp(t,t[::5],col) for col in np.asarray(qc).T],1)
    np.savez_compressed(out,time=t,q_path=q,joint_names=['joint1','joint2','joint3','joint4','joint5','joint6','left_finger','right_finger'])
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('output',type=Path);a.add_argument('--depth',type=float,default=.004);a.add_argument('--gap',type=float,default=.013)
    a.add_argument("--down-end",type=float,default=1.50)
    a.add_argument("--offset-y",type=float,default=0.)
    a.add_argument("--carry-gap",type=float);a.add_argument("--lift-height",type=float,default=.247)
    a.add_argument("--pitch-deg",type=float,default=-32.)
    v=a.parse_args();plan(v.output,v.depth,v.gap,v.down_end,v.offset_y,v.carry_gap,v.lift_height,v.pitch_deg)

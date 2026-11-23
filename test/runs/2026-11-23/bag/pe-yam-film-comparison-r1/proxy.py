"""Independent four-point parcel dynamics on the recorded YAM joint boundary."""
import argparse
import json
import time
from pathlib import Path
import mujoco as mj
import numpy as np
import torch
from record import ROOT, BOX, CENTER, MASS, make_scene, pad_faces, DT
from src.low_order_parcel.lumped_contact_proxy import LumpedContactProxy, ProxyParams, ContactState
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim, ParcelAction
from src.low_order_parcel.neck_damping import NeckDampingParams
from src.low_order_parcel.pad_torsion import PadTorsionParams
from src.low_order_parcel.unified_payload import PayloadParamsX, UnifiedPayloadTemplate
from src.low_order_parcel.yam_adapter import Config as ProxyYam

@torch.inference_mode()
def run(source,out,offset_y,cloth_xy=None,neck_length=None,direct_textile=False,cloth_stiffness=92.5):
    dt=.002
    mass=.50 if direct_textile else MASS
    # Both solvers take exactly this joint and rail position trace for their moving boundary.
    path=np.load(source)
    q=path['q_path']
    r=make_scene()
    template=r.sim.parcel.envelope
    dims=np.ptp(template['current'],axis=0)
    cfg=json.loads((ROOT/'configs/payload_template_X.yaml').read_text())
    envelope=json.loads((ROOT/'configs/proxy_material_envelopes.yaml').read_text())['materials']['pe']['nominal']['proxy']
    box=BOX
    if cloth_xy is not None:
        box=(*cloth_xy,.020)
        # The UMI calibration uses the garment volume for the four-block
        # payload; free film around that volume is represented by the ropes.
        dims=np.asarray(box)
    if direct_textile:
        envelope['L0']=[0.,0.]
        # Two load paths in the squeezed skirt share its weight.
        envelope['k']=cloth_stiffness
    elif neck_length is not None:
        envelope['L0']=[neck_length,neck_length]
    fill_volume=np.prod([.29,.39,.020]) if cloth_xy is not None else np.prod(dims)
    params=PayloadParamsX.from_presets(cfg,['folded_clothes'],bag_half_extents=torch.tensor(dims[None]/2,dtype=torch.float32),
            total_mass=torch.tensor([[mass]]),fill_ratio=torch.tensor([[float(np.prod(box)/fill_volume)]]))
    payload=UnifiedPayloadTemplate(params)
    torsion=PadTorsionParams()
    if direct_textile:
        torsion=PadTorsionParams(radius_m=.010,neck_length_m=.020,stiffness_Nm_rad=.001)
    elif neck_length is not None:
        # Tensioned central strip: small-angle couple T*r^2/L, unlike the
        # 35 mm reference neck of the old folded-edge case.
        radius=.010
        torsion=PadTorsionParams(radius_m=radius,neck_length_m=neck_length,
                                stiffness_Nm_rad=MASS*9.81*radius**2/neck_length)
    sim=MacroParcelSim(LumpedContactProxy(ProxyParams.from_scalars(envelope),dt),payload=payload,
                       neck_damping=NeckDampingParams(12.),torsion=torsion)
    spacing=.0015 if direct_textile else .024
    mount_xy=torch.tensor([[[-spacing,offset_y],[spacing,offset_y]]])
    sim.payload_mount_weights=payload.weights_at(mount_xy)
    state=sim.initial_state()
    shift=state.com.new_tensor([*CENTER,0])
    state.payload.pos.add_(shift); state.com.add_(shift)
    settings=ProxyYam()
    zeros=torch.zeros(1,2)
    keys='time centers frame half_height capture anchors reference clearance support velocity com tension normal pads'
    frames={k:[] for k in keys.split()}
    summary=[]
    kin=r.kin
    ids=np.r_[r.qadr,r.fingers]
    def fk(v):
        kin.qpos[ids]=v
        mj.mj_kinematics(r.sim.m,kin)
        return pad_faces(r,kin)
    prev=fk(q[0])
    start=time.perf_counter()
    for i,angles in enumerate(q):
        pad=fk(angles)
        vel=(pad-prev)/dt
        prev=pad
        axis=pad[1]-pad[0];gap=np.linalg.norm(axis);axis/=gap
        inward=-((vel-vel.mean(0))*np.stack([-axis,axis])).sum(1)
        tip=state.com.new_tensor(pad)[None]
        b=state.payload;R=payload.frame(b)
        local=torch.bmm(tip-b.pos.mean(1,keepdim=True),R)[0]
        in_film=(local[:,:2].abs()<=params.bag_half_extents[0,:2]).all(1)
        touch=in_film & (local[:,2]<=payload.half_height[0,0]) & (local[:,2]>-payload.half_height[0,0])
        captured=state.contacts[0,:,ContactState.CAPTURE]>.5
        normal=(touch|captured)*settings.fold_normal_stiffness_N_m*max(0.,(settings.fold_width_m-gap)/2)
        if i%20==0:
            anchors,_=payload.anchors(b,sim.payload_mount_weights,torch.zeros_like(b.d_surf))
            for k,v in dict(time=i*dt,centers=b.pos[0],frame=R[0],half_height=payload.half_thickness(b.d_surf)[0],
                    capture=captured,anchors=anchors[0],reference=state.contact_reference[0],pads=pad,normal=normal,
                    com=state.com[0],velocity=state.velocity[0],clearance=float(result.table_clearance) if i else 0.,
                    support=float(result.support_N) if i else 0.,tension=result.tension[0] if i else zeros[0]).items():
                frames[k].append(np.asarray(v).copy())
        if i==len(q)-1:break
        vel_tensor=tip.new_tensor(vel)[None]
        feed=alignment=zeros
        if cloth_xy is not None:
            # Measure insertion below the undented textile and lateral gathering
            # orientation from the actual two pad positions, without forcing capture.
            feed=torch.where(touch,payload.half_height-local[:,2],zeros).clamp_min(0)
            normal_component=tip.new_tensor(axis)@R[0,:,2]
            alignment=torch.where(touch,(1-normal_component.square()).clamp_min(0).sqrt(),zeros)
        action=ParcelAction(tip,vel_tensor,normal[None],tip.new_tensor(inward)[None],feed,alignment,
                            torch.zeros(1,3),torch.zeros(1,1))
        wrist=kin.body('link_6').xquat.copy()
        angular=np.zeros(3)
        if i:
            mj.mju_subQuat(angular,wrist,old_wrist)
            angular=kin.body('link_6').xmat.reshape(3,3)@angular/dt
        old_wrist=wrist
        result=sim.step(state,action,jaw_orientation=tip.new_tensor(wrist)[None],
                        jaw_angular_velocity=tip.new_tensor(angular)[None])
        state=result.state
        summary.append([i*dt,float(result.table_clearance),float(result.support_N),float(state.velocity.norm()),
                        float(state.contacts[0,:,ContactState.CAPTURE].min()),float(state.contacts[0,:,0].min())])
    wall=time.perf_counter()-start
    np.savez_compressed(out,**frames,audit_2ms=summary,weights=sim.payload_mount_weights[0].numpy())
    print('mass, size, final samples, wall, steps/s',mass,dims,summary[-1],wall,(len(q)-1)/wall)
    out.with_suffix('.json').write_text(json.dumps(dict(steps=len(q)-1,wall_s=wall,steps_s=(len(q)-1)/wall,
        mass=mass,direct_textile=direct_textile,nominal_L0=envelope["L0"],box_m=box,envelope_m=dims.tolist(),material='PE',content='folded_clothes',dt=dt)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--offset-y',type=float,default=0.)
    p.add_argument("--cloth-xy",nargs=2,type=float);p.add_argument("--neck-length",type=float)
    p.add_argument("--direct-textile",action="store_true")
    p.add_argument("--cloth-stiffness",type=float,default=92.5)
    a=p.parse_args();run(a.source,a.output,a.offset_y,a.cloth_xy,a.neck_length,a.direct_textile,a.cloth_stiffness)

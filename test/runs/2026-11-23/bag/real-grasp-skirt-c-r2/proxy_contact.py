"""Run the unchanged four-block payload and two-channel contact model at 2 ms."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import torch
import mujoco as mj

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
sys.path.insert(0,str(REPO))
from src.low_order_parcel.lumped_contact_proxy import ContactState, LumpedContactProxy, ProxyParams
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim, ParcelAction
from src.low_order_parcel.neck_damping import NeckDampingParams
from src.low_order_parcel.pad_torsion import PadTorsionParams
from src.low_order_parcel.unified_payload import PayloadParamsX, UnifiedPayloadTemplate

CLOTH=np.array([.245,.345,.020])
CENTER=np.array([-.03,-.225])
DT=.002


@torch.inference_mode()
def run(source,out,mass,stiffness):
    r=np.load(source)
    materials=json.loads((REPO/"configs/proxy_material_envelopes.yaml").read_text())
    p=dict(materials["materials"]["pe"]["nominal"]["proxy"])
    p.update(L0=[0.,0.],k=stiffness)
    model=json.loads((REPO/"configs/payload_template_X.yaml").read_text())
    as_tensor=lambda x:torch.as_tensor(x,dtype=torch.float32)
    params=PayloadParamsX.from_presets(model,["folded_clothes"],
       bag_half_extents=as_tensor(CLOTH[None]/2),total_mass=as_tensor([[mass]]),
       fill_ratio=as_tensor([[np.prod(CLOTH[:2])/(.29*.39)]]))
    payload=UnifiedPayloadTemplate(params)
    sim=MacroParcelSim(LumpedContactProxy(ProxyParams.from_scalars(p),DT),payload=payload,
       neck_damping=NeckDampingParams(12.),torsion=PadTorsionParams(radius_m=.010,neck_length_m=.020,
       stiffness_Nm_rad=.001))
    sim.payload_mount_weights=payload.weights_at(as_tensor([[[-.0015,0],[.0015,0]]]))
    state=sim.initial_state()
    shift=as_tensor([*CENTER,0.])
    state.payload.pos.add_(shift)
    state.com.add_(shift)
    wrist=r["quats"][:,list(r["names"]).index("link_6")]
    samples={key:[] for key in ("time","centers","frame","half","indent","pads","vertical_force",
            "pinch_normal","capture","tension","support","clearance","com","velocity","slack")}
    history=[]
    surface_press=np.zeros(2)
    tension=np.zeros(2)
    support=mass*9.81
    clearance=0.
    t0=perf_counter()
    pads=r["pads"]
    for i,pp in enumerate(pads):
        t=i*DT
        ppv=(pp-pads[max(0,i-1)])/DT
        line=pp[1]-pp[0]
        gap=np.linalg.norm(line)
        axis=line/gap
        inward=-((ppv-ppv.mean(0))*np.stack((-axis,axis))).sum(1)
        tip,tip_vel=as_tensor(pp[None]),as_tensor(ppv[None])
        st=state.payload
        R=payload.frame(st)
        local=torch.bmm(tip-st.pos.mean(1,keepdim=True),R)[0]
        inside=(local[:,:2].abs()<=params.bag_half_extents[0,:2]).all(1)
        previously=state.contacts[0,:,ContactState.CAPTURE]>.5
        nominal_depth=payload.half_height[0,0]-local[:,2]
        touch=inside & (nominal_depth>0) & (local[:,2]>-payload.half_height[0,0])
        # Calibrated finite gap / film compression. Capture is subject to
        # the existing gather, feed, normal and pull-out tests in the core.
        normal=(touch|previously) * (1600.*max(0.,(.024-gap)/2))
        feed=nominal_depth.clamp_min(0)*touch
        normal_component=as_tensor(axis)@R[0,:,2]
        align=(1-normal_component.square()).clamp_min(0).sqrt()*touch
        if i%20==0:
            for k,v in dict(time=t,centers=st.pos[0],frame=R[0],half=payload.half_thickness(st.d_surf)[0],
                   indent=st.d_surf[0],pads=pp,vertical_force=surface_press,pinch_normal=normal,
                   capture=previously,tension=tension,support=support,clearance=clearance,
                   com=state.com[0],velocity=state.velocity[0],slack=state.contacts[0,:,ContactState.SLACK]).items():
                samples[k].append(np.asarray(v).copy())
        if i==len(pads)-1:break
        omega=np.zeros(3)
        if i:
            mj.mju_subQuat(omega,wrist[i],wrist[i-1])
            rr=np.empty(9)
            mj.mju_quat2Mat(rr,wrist[i])
            omega=rr.reshape(3,3)@omega/DT
        contact_weights=payload.weights_at(local[None,:,:2])
        weights_z=(torch.bmm(contact_weights,st.pos)*R[:,None,:,2]).sum(dim=-1)
        world_z=(tip*R[:,None,:,2]).sum(dim=-1)
        depth=payload.half_height-(world_z-weights_z)
        demand=payload.predict_surface_demand(st,tip+DT*tip_vel,inside[None])
        surface=payload.surface_contact(st,depth,contact_weights,inside[None],demand)
        surface_press=surface.pad_force_N[0].numpy().copy()
        action=ParcelAction(tip,tip_vel,normal[None],as_tensor(inward[None]),feed[None],align[None],
                            torch.zeros(1,3),torch.zeros(1,1))
        result=sim.step(state,action,jaw_orientation=as_tensor(wrist[i:i+1]),jaw_angular_velocity=as_tensor(omega[None]))
        state=result.state
        tension=result.tension[0].numpy().copy()
        support=float(result.support_N)
        clearance=float(result.table_clearance)
        history.append([t,clearance,support,float(state.velocity.norm()),*surface_press,*normal.numpy(),
           *state.contacts[0,:,ContactState.CAPTURE].numpy()])
        if i%250==0: print(t,"clearance",clearance,"pressure",surface_press,"tension",tension,flush=True)
    np.savez_compressed(out,**samples,history=np.asarray(history),steps=len(pads)-1,wall_s=perf_counter()-t0,
                        mass=mass,stiffness=stiffness)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--mass",type=float,default=.5)
    p.add_argument("--stiffness",type=float,default=92.5)
    a=p.parse_args()
    run(a.source,a.out,a.mass,a.stiffness)

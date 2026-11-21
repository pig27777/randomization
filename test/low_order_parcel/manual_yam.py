"""Control a YAM over a Stage 2.5 parcel in the native MuJoCo viewer (CPU)."""

import argparse
import json
from pathlib import Path
from queue import SimpleQueue
import sys
import time
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experience.yam_platform.protocol import MuJoCoSim, attach_robot, tips
from src.low_order_parcel.lumped_contact_proxy import LumpedContactProxy, ProxyParams
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim, ParcelAction
from src.low_order_parcel.unified_payload import DT, PayloadParamsX, UnifiedPayloadTemplate
from src.low_order_parcel.yam_adapter import Config, ENVELOPES

CENTER = np.array([-.25, .04])
COLORS = [[.94,.72,.39,1], [.85,.61,.31,1], [.89,.65,.35,1], [.96,.76,.43,1]]


class ManualScene:
    def __init__(self, preset):
        config = json.loads((ROOT / 'configs/payload_template_X.yaml').read_text())
        x = PayloadParamsX.from_presets(config,[preset])
        self.template = UnifiedPayloadTemplate(x)
        pe = json.loads(ENVELOPES.read_text())['materials']['pe']['nominal']['proxy']
        self.sim = MacroParcelSim(LumpedContactProxy(ProxyParams.from_scalars(pe), DT), payload=self.template)
        self.size = 2*x.bag_half_extents[0].numpy()
        self.cfg = Config(dt=DT, yaw=90.)
        xml = """<mujoco model="Manual YAM - Stage 2.5 top grasp"><option timestep=".002" integrator="implicitfast"/>
        <worldbody><light pos="-.4 -.4 .8" dir=".3 .4 -1" directional="true"/>
        <geom name="table" type="box" pos="0 0 -.025" size=".85 .32 .025" rgba=".25 .30 .33 1"/>
        </worldbody><contact/><equality/></mujoco>"""
        built = SimpleNamespace(xml=xml,desc=SimpleNamespace(),cfg=self.cfg)
        pitch = np.deg2rad(40.)
        direction = np.array([0.,np.cos(pitch),-np.sin(pitch)])
        approach = dict(direction=direction, closing=[-1.,0.,0.],
                        contact_point=np.r_[CENTER,self.size[2]]-.02*direction)
        self.robot = attach_robot(self.cfg,built,MuJoCoSim,CENTER,self.size[2],self.size,
                                  float(x.total_mass[0,0]), approach)
        self.m,self.d = self.robot.sim.m,self.robot.sim.d
        self.fingers = [self.m.body(n+'_finger').id for n in ('left','right')]
        self.initial_qpos = self.d.qpos.copy()
        self.initial_ctrl = self.d.ctrl.copy()
        self.reset()

    def pads(self):
        pad = tips(self.robot)
        axis = pad[1]-pad[0]
        axis /= max(np.linalg.norm(axis),1e-8)
        return pad + np.array([.004,-.004])[:,None]*axis, axis

    def reset(self):
        mujoco.mj_resetData(self.m,self.d)
        self.d.qpos[:] = self.initial_qpos
        self.d.ctrl[:] = self.initial_ctrl
        mujoco.mj_forward(self.m,self.d)
        self.robot.qtarget = self.initial_qpos[self.robot.qadr].copy()
        self.robot.steps = 0
        self.robot.ik_errors.clear()
        self.target = self.robot.initial_target.copy()
        self.command = self.target.copy()
        self.gap = self.command_gap = .095
        self.state = self.sim.initial_state()
        shift = self.state.com.new_tensor([*CENTER,0.])
        self.state.payload.pos.add_(shift)
        self.state.com.add_(shift)
        self.pad,_ = self.pads()
        self.tip_offset = self.pad.mean(0)-self.target
        self.result = None
        self.normal = np.zeros(2)
        self.align_first = False

    def key(self,key):
        """Move the grasp site or jaw target using individual Cartesian commands."""
        move = {ord('A'):(-.01,0,0),ord('D'):(.01,0,0),ord('W'):(0,.01,0),ord('S'):(0,-.01,0),
                ord('U'):(0,0,.01),ord('J'):(0,0,-.01)}
        if key in move:
            self.target += move[key]
        elif key==ord('L'):
            self.target[2] = self.size[2]+.195-self.tip_offset[2]
        elif key==ord('G'):
            # Position commands only. Actual motion is rate limited and uses YAM IK/servos.
            center = self.state.payload.pos[0].mean(0).numpy()
            self.target[:2] = center[:2]-self.tip_offset[:2]
            self.target[2] = self.size[2]-.040-self.tip_offset[2]
            self.align_first = True
        elif key in (ord('C'),ord('O')):
            self.gap = .001 if key==ord('C') else .095
        elif key==ord('R'):
            self.reset()
        else:
            return False
        self.target[:] = np.clip(self.target,[-.40,-.20,.015],[-.10,.23,.47])
        return True

    @torch.inference_mode()
    def step(self):
        delta = self.target-self.command
        if self.align_first:
            self.align_first = np.linalg.norm(delta[:2]) > .0001
            if self.align_first:
                delta[2] = 0.
        self.command += delta*min(1.,.070*DT/max(np.linalg.norm(delta),1e-10))
        self.command_gap += np.clip(self.gap-self.command_gap,-.050*DT,.050*DT)
        robot = self.robot
        robot.command(self.command,self.command_gap)
        # Only the latest diagnostic is needed during an unbounded interactive session.
        del robot.ik_errors[:-1]
        pad,axis = self.pads()
        vel = (pad-self.pad)/DT
        self.pad = pad
        outward = np.stack((-axis,axis))
        inward = -((vel-vel.mean(0))*outward).sum(1)
        p, st = self.template, self.state.payload
        tip = st.pos.new_tensor(pad)[None]
        local = torch.bmm(tip-st.pos.mean(1,keepdim=True),p.frame(st))[0]
        inside = (local[:,:2].abs()<=self.sim.half_extents[0,:2]).all(1)
        touch = inside & (local[:,2]<=p.half_height[0,0]) & (local[:,2]>-p.half_height[0,0])
        captured = self.state.contacts[0,:,1]>.5
        self.normal = (touch|captured).numpy()*self.cfg.fold_normal_stiffness_N_m*max(
            0.,(self.cfg.fold_width_m-robot.gap())/2.)
        zeros = torch.zeros(1,2)
        action = ParcelAction(tip, tip.new_tensor(vel)[None],tip.new_tensor(self.normal)[None],
                              tip.new_tensor(inward)[None],zeros,zeros,torch.zeros(1,3),torch.zeros(1,1))
        r = self.sim.step(self.state,action)
        self.state,self.result = r.state,r
        # Map the template's measured wrench about the pad midpoint back to YAM.
        # Opposed fold normals are internal to this wrench but load the two jaw rails.
        force, torque = r.wrist_wrench[0].numpy().reshape(2,3)
        arm = pad[1]-pad[0]
        diff = np.cross(torque,arm)/max(np.dot(arm,arm),1e-8)
        reaction = force/2 + np.stack((-diff,diff)) + self.normal[:,None]*outward
        self.d.qfrc_applied[:] = 0
        for i,body in enumerate(self.fingers):
            mujoco.mj_applyFT(self.m,self.d,reaction[i],np.zeros(3),pad[i],body,self.d.qfrc_applied)
        mujoco.mj_step(self.m,self.d)
        mujoco.mj_kinematics(self.m,self.d)

    def draw(self,viewer):
        scn = viewer.user_scn
        state = self.state.payload
        frame = self.template.frame(state)[0].numpy()
        heights = self.template.half_thickness(state.d_surf)[0].numpy()
        # The four rendered patches follow the actual block centers and local compression.
        scn.ngeom = 0
        width = .25*self.size[:2]
        for pos,h,color in zip(state.pos[0].numpy(),heights,COLORS,strict=True):
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom],mujoco.mjtGeom.mjGEOM_BOX,
                                np.r_[width*.994,h],pos,frame.ravel(),np.asarray(color,dtype=np.float32))
            scn.ngeom += 1
        anchors,_ = self.template.anchors(state,self.sim.payload_mount_weights,torch.zeros_like(state.d_surf))
        for i,color in enumerate(([.12,.8,1.,1.],[1.,.33,.10,1.])):
            mount = anchors[0,i].numpy()
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.full(3,.003),mount,np.eye(3).ravel(),np.asarray(color,dtype=np.float32))
            scn.ngeom += 1
            if self.state.contacts[0,i,1]>.5:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),mount,np.eye(3).ravel(),np.asarray(color,dtype=np.float32))
                mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.0015,mount,self.pad[i])
                scn.ngeom += 1

    def status(self):
        r = self.result
        return dict(grip_mm=round(1000*self.robot.gap(),1),indent_mm=round(1000*float(self.state.payload.d_surf.max()),1),
                    capture=self.state.contacts[0,:,1].int().tolist(),
                    tension_N=[0.,0.] if r is None else r.tension[0].tolist(),
                    clearance_mm=0. if r is None else 1000*float(r.table_clearance[0,0]))


def manual(scene,preset):
    keys = SimpleQueue()
    paused = True
    with mujoco.viewer.launch_passive(scene.m,scene.d,key_callback=keys.put,show_left_ui=False,show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [-.17,-.09,.15]
            viewer.cam.distance = .95
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.opt.geomgroup[3] = False
        print(f'Manual YAM ready: {preset}. Click viewer; G lower, C close, L lift, O open. R reset.',flush=True)
        while viewer.is_running():
            before = time.monotonic()
            while not keys.empty():
                key = keys.get()
                if key==32:
                    paused = not paused
                elif scene.key(key):
                    paused = key==ord('R')
                    print(f'key {chr(key)}  target={np.round(scene.target,3)} gap={scene.gap:.3f}',flush=True)
            if not paused:
                for _ in range(10):
                    scene.step()
            s = scene.status()
            with viewer.lock():
                scene.draw(viewer)
            viewer.set_texts((None,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                'Manual YAM  |  Top grasp  |  PE / '+preset+'\nTarget\nJaw\nGrip / indentation\nLeft / right captured\nLeft / right tension\nBottom clearance',
                ('Paused' if paused else 'Live')+
                '\nW/S Y   A/D X   U/J up/down\nG lower to press   C close   L lift   O open   R reset   Space pause'+
                f"\n{s['grip_mm']:.1f} / {s['indent_mm']:.1f} mm\n{s['capture']}\n"+
                '   '.join(f'{f:.2f} N' for f in s['tension_N'])+f"\n{s['clearance_mm']:.1f} mm"))
            viewer.sync()
            time.sleep(max(.001,.034-(time.monotonic()-before)))


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--payload',choices=('fluffy_pillow','folded_clothes','rigid_box','heavy_quilt','loose_shifting'),default='fluffy_pillow')
    args = ap.parse_args()
    torch.set_num_threads(1)
    manual(ManualScene(args.payload),args.payload)

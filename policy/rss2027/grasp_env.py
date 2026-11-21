"""Two-decision pad-space environment over the frozen Stage 2.5 dynamics."""
import math
import torch
from torch.utils._pytree import tree_map

from .domain import HardwareGapFilter
from src.low_order_parcel.lift_probe import CAMERA_OFFSET, CAMERA_ROTATION, PROBE_RAMP_S, PROBE_SETTLE_S, probe_height
from src.low_order_parcel.pad_torsion import PadTorsionParams
from src.low_order_parcel.virtual_rgb import CameraPose, RGBParameters, VirtualRGBProjector
from src.low_order_parcel.lumped_contact_proxy import LumpedContactProxy
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim, ParcelAction
from src.low_order_parcel.unified_payload import DT, UnifiedPayloadTemplate

VARIANTS = ("ours","b0_heuristic","b1_rigid_snap","ablation_no_history","ablation_no_touchdown",
            "ablation_no_microlift","ablation_no_hw_dr","ablation_rigid_payload_only")


class SoftParcelGraspEnv:
    """One synchronized batch; retries preserve the physical parcel and hardware state."""
    def __init__(self, episode, variant="ours", training=False, open_loop=False):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant}")
        self.e, self.variant, self.open_loop = episode, variant, open_loop
        self.sim = MacroParcelSim(LumpedContactProxy(episode.proxy, DT,
                                  ablation_rigid_snap=training and variant=="b1_rigid_snap"),
                                  payload=UnifiedPayloadTemplate(episode.x), torsion=PadTorsionParams())
        self.state = tree_map(torch.Tensor.clone, self.sim.initial_state())
        self.count = len(self.state.com)
        self.ext = episode.x.bag_half_extents
        self.top = 2*self.ext[:,2:3]+episode.height_error
        yaw, tilt = episode.yaw, episode.tilt
        cy, sy, ct, st = yaw.cos(), yaw.sin(), tilt.cos(), tilt.sin()
        self.across = torch.cat((cy,sy,0*cy),1)
        self.along = torch.cat((-sy,cy,0*cy),1)
        rest = self.sim.payload.rest
        pos = (self.across[:,None,:]*rest[:,:,:1]*ct[:,None]+self.along[:,None,:]*rest[:,:,1:2]
               + self.sim.up[:,None,:]*(rest[:,:,:1]*st[:,None]+self.ext[:,None,2:3]))
        pos[:,:,2] += (self.ext[:,2:3]-pos[:,:,2].amin(1,keepdim=True))
        self.state.payload.pos.copy_(pos)
        self.state.com.copy_(self.sim.payload.com(self.state.payload))
        self.origin = self.state.com.new_zeros(self.count,3)
        self.like = self.top
        self.side_sign = torch.where(episode.clearance[:,:1]>=episode.clearance[:,1:2],1.,-1.)
        self.room = episode.clearance.amax(1,keepdim=True)
        self.width = self.ext[:,:1].clamp(max=.06)
        self.sign = self.like.new_tensor([-1.,1.])
        self.zero = torch.zeros_like(self.top)
        self.mode = self.zero.clone()
        self.route = self.zero.clone()
        self.a = self.zero.expand(-1,4).clone()
        self.b = self.zero.expand(-1,2).clone()
        self.request = torch.cat((self.zero,self.top+.004,self.width,self.zero,self.zero,self.zero,.022+self.zero),1)
        self.real = self.request.clone()
        self.start = self.real.clone()
        self.feedback = self.zero.clone()
        # Passive Cartesian wrist compliance, common to every training/test variant.
        self.deflection = torch.zeros_like(self.origin)
        self.deflection_v = torch.zeros_like(self.origin)
        self.wrist_stiffness = self.top.new_tensor([55.,55.,800.])
        self.wrist_damping = self.top.new_tensor([4.,4.,12.])
        self.last_force = torch.zeros_like(self.origin)
        self.probe_stop = self.zero.bool()
        self.hw = HardwareGapFilter(episode,self.request,enabled=not (training and variant=="ablation_no_hw_dr"))
        self.tick = self.zero.new_zeros(1,dtype=torch.long)
        self.phase = self.tick.clone()
        self.best = self.zero.clone()
        self.streak = self.zero.clone()
        self.airborne = self.zero.bool()
        self.dropped = self.zero.bool()
        self.finite = torch.ones_like(self.airborne)
        self.net = self.zero.clone()
        self.peak_press = self.zero.clone()
        self.valid_micro = self.zero.bool()
        self.samples = self.zero.new_zeros(32,self.count,8)
        self.probe_height = probe_height(self.top)
        self.probe_start_z = self.zero.clone()
        self.probe_start_load = self.zero.clone()
        # Capture physics at 2 ms. Expose/receive RGB outside the CUDA graph at 50 Hz.
        self.rgb_history = self.zero.new_zeros(32, self.count, 2)
        pose = CameraPose(self.top.new_tensor(CAMERA_OFFSET).expand(self.count,-1),
                          self.top.new_tensor(CAMERA_ROTATION).expand(self.count,-1,-1))
        camera = RGBParameters.sample(self.top)
        camera = camera._replace(delay_steps=(camera.delay_steps + 5) // 10)
        if not self.hw.enabled:
            camera = camera._replace(translation_m=camera.translation_m*0, rotation_rad=camera.rotation_rad*0,
                                     drop_probability=camera.drop_probability*0, mask_jitter=camera.mask_jitter*0,
                                     delay_steps=camera.delay_steps*0)
        self.rgb = VirtualRGBProjector(self.sim.payload, pose, camera, frame_period_steps=1,
                                       table_bounds=self.top.new_tensor([-.5,.5,-.5,.5]).expand(self.count,-1))
        self.physics_steps = 0
        self.attempt = 0
        self.graph = None
        if self.top.is_cuda:
            self.graph = torch.cuda.CUDAGraph()
            # Capture records operations without advancing the episode on the device.
            with torch.cuda.graph(self.graph):
                self._tick()

    def _pad(self, cmd):
        """[B,6] float32 SI command to [B,2,3] pad locations in m."""
        along,z,half,_,_,offset,cant = cmd.split(1,dim=1)
        xy = self.origin+self.along*along*self.side_sign+self.across*offset
        return xy[:,None]+self.across[:,None]*half[:,:,None]*self.sign[None,:,None]+self.sim.up[:,None]*(z+cant*(self.sign+1)*.5)[:,:,None]

    def _tick(self):
        """Advance every row by 2 ms; update float32 pose, wrench and success counters."""
        t = (self.tick+1)*DT
        ph = self.phase
        old_y,old_z,_,_,_,_,_ = self.request.split(1,1)
        sy,sz,sh,sg,sfeed,so,sc = self.start.split(1,1)
        a0,a1,a2,a3 = self.a.split(1,1)
        side = self.mode > .5
        # Smooth approach to the exposed side or to the top. Obstructions limit actual travel.
        entry = -.004+.023*a0
        stroke = .016+.045*a1
        speed = .035+.12*a3
        lip = -self.ext[:,1:2]-.004
        approach = ((t-.14)/.25).clamp(0.,1.)
        approach = .5-.5*torch.cos(math.pi*approach)
        travel = torch.minimum(stroke, ((t-.39)*speed).clamp_min(0.))
        blocked = self.room < .020
        travel = torch.where(blocked,0.,travel)
        side_y = torch.lerp(sy,lip,approach)+travel
        high = self.top+.030
        raised = torch.lerp(sz,high,(t/.14).clamp(max=1.))
        side_z = torch.lerp(raised,self.top+entry,((t-.39)/.12).clamp(0.,1.))
        close = ((t-.40)/.70).clamp(0.,1.)
        press_target = 4.+26.*a0
        top_z = old_z + DT*(.006*(self.feedback-press_target)).clamp(-.13,.065)
        if self.open_loop:
            top_z = torch.lerp(sz,(self.top-(.012+.050*a0)).clamp_min(.003), (t/.38).clamp(max=1.))
        probe_z = torch.maximum(old_z-DT*.10, self.top+.004-.015)
        self.probe_stop.logical_or_((ph==0)&((self.feedback>=2.) & (not self.open_loop)))
        probe_z = torch.where(self.probe_stop,old_z,probe_z)
        z = torch.where(ph==0,probe_z,torch.where(side,side_z,top_z.clamp_min(.003)))
        y = torch.where((ph==1)&side,side_y,sy*(1-approach))
        half = torch.where(ph==1, (self.width-close*stroke).clamp_min(.0015),sh)
        grip = torch.where(ph==1,(3.+30.*a2)*((t-.28)/.12).clamp(0.,1.),0.)
        side_depth = (travel-.004).clamp_min(0.)
        feed = torch.where((ph==1)&side,side_depth,0.)
        off = torch.where((ph==1)&~side, (.024*a3-.012)*approach, so)
        micro = (ph==2)
        lift = ph==3
        retry = self.route==2
        scale = torch.where(self.route==1,.35+.5*self.b[:,1:2],1.)
        u = (t*scale/1.20).clamp(0.,1.)
        lift_z = sz + .265*(.5-.5*torch.cos(math.pi*u))
        micro_z = sz + self.probe_height * (t / PROBE_RAMP_S).clamp(max=1.)
        # Carry the gathered side fold over the parcel before the main lift.
        centered = sy*(1-(.5-.5*torch.cos(math.pi*(t/.55).clamp(max=1.))))
        release = (ph==4)|(lift & retry)
        z = torch.where(micro,micro_z,torch.where(lift,lift_z,z))
        z = torch.where(release,torch.lerp(sz,self.top+.002,(t/.5).clamp(max=1.)),z)
        y = torch.where(release,sy*(1-(t/.6).clamp(max=1.)),torch.where(micro,sy,torch.where(lift,centered,y)))
        half = torch.where(release,torch.lerp(sh,self.width,(t/.25).clamp(max=1.)),torch.where(micro|lift,sh,half))
        boost = torch.where((self.route==1)&lift,2.+12.*self.b[:,:1],0.)
        grip = torch.where(release,0.,torch.where(micro|lift,sg+boost,grip))
        feed = torch.where(release,0.,torch.where(micro|lift,sfeed,feed))
        off = torch.where(micro|lift,so,off)
        cant = torch.where(ph==0,.022+self.zero,sc*(1-(t/.14).clamp(max=1.)))
        cant = torch.where(release,.022+self.zero,cant)
        cmd = torch.cat((y,z,half,grip,feed,off,cant),1)
        self.request.copy_(cmd)
        actual = self.hw.actuate(cmd)
        target_half = actual[:,2:3]
        actual_grip = actual[:,3:4]
        if self.hw.enabled:
            delta = target_half-self.real[:,2:3]
            target_half = torch.where(delta.abs()>self.e.deadband/2,target_half,self.real[:,2:3])
            actual_grip = actual_grip*self.e.current_gain
        actual = torch.cat((actual[:,:2],target_half,actual_grip,actual[:,4:]),1)
        # Implicit spring/damper update in SI; the open-loop target remains prescribed.
        k, c, mass = self.wrist_stiffness, self.wrist_damping, .12
        vf = (mass*self.deflection_v+DT*(self.last_force-k*self.deflection))/(mass+DT*c+DT*DT*k)
        offset = (self.deflection+DT*vf.clamp(-.18,.18)).clamp(-.06,.06)
        self.deflection_v.copy_((offset-self.deflection)/DT)
        self.deflection.copy_(offset)
        pos = self._pad(actual)+offset[:,None]
        vel = (self._pad(actual)-self._pad(self.real))/DT+self.deflection_v[:,None]
        inward = ((self.real[:,2:3]-target_half)/DT).expand(-1,2)
        # Side geometry comes from the achieved lip entry. Insufficient table clearance
        # and pushes feed through the existing contact geometry and side-wall solve.
        dep = actual[:,4:5].expand(-1,2)
        frame = self.sim.payload.frame(self.state.payload)
        local = torch.bmm(pos-self.state.payload.pos.mean(1,keepdim=True),frame)
        near_fold = ((local[:,:,:2].abs() <= self.ext[:,None,:2]+.004).all(2)
                     & ((local[:,:,2]-self.ext[:,2:3]).abs()<.028))
        eligible = near_fold | (self.state.contacts[:,:,1]>.5)
        alignment = ((dep>0)&eligible).to(pos.dtype)
        result = self.sim.step(self.state, ParcelAction(pos,vel,actual_grip.expand(-1,2),inward,
                                     dep,alignment,self.origin*0,self.zero))
        tree_map(lambda dst,src: dst.copy_(src),self.state,result.state)
        self.last_force.copy_(result.wrist_wrench[:,:3])
        self.peak_press.copy_(torch.maximum(self.peak_press,result.wrist_wrench[:,2:3]*(ph==1)))
        received = self.hw.observe(result.wrist_wrench)
        self.feedback.copy_(received[:,2:3])
        self.real.copy_(actual)
        clear = (result.table_clearance>.001)&(result.support_N==0)
        both = (result.state.contacts[:,:,1]>.5).all(1,keepdim=True)
        quiet = result.state.velocity.norm(dim=1,keepdim=True)<.020
        finite = torch.isfinite(result.wrist_wrench).all(1,keepdim=True)&torch.isfinite(self.state.payload.pos.flatten(1)).all(1,keepdim=True)
        self.finite.logical_and_(finite)
        self.dropped.logical_or_(self.airborne & ~clear & (micro|lift) & ~release)
        self.airborne.logical_or_(clear & both & (micro|lift))
        self.streak.copy_(torch.where(lift & ~retry & clear & both & quiet & (u>=1.) & self.finite,self.streak+1,0))
        self.best.copy_(torch.maximum(self.streak,self.best))
        self.net.copy_(torch.maximum(self.net, self.state.contacts[:,:,4].amin(1,keepdim=True)*(ph==1)))
        self.valid_micro.copy_(both & self.finite)
        sample = torch.cat((received[:,2:3],received[:,3:].norm(dim=1,keepdim=True),torch.cat((actual[:,1:2]+self.deflection[:,2:3],actual[:,2:3]),1),
                            self.request[:,3:4],self.room,self.ext[:,:2]),1)
        self.samples.index_copy_(0,(self.tick//10)%32,sample.unsqueeze(0))
        self.tick.add_(1)

    def run(self, phase, steps):
        """Advance a batch by integer ticks, return 50-Hz float32 [B,T,10] sensor history."""
        self.phase.fill_(phase)
        self.tick.zero_()
        self.start.copy_(self.request)
        if phase==0:
            self.probe_stop.zero_()
        if phase == 2:
            self.probe_start_z.copy_(self.real[:,1:2] + self.deflection[:,2:3])
            self.probe_start_load.copy_(self.feedback)
            self.rgb.reset(torch.ones(self.count, dtype=torch.bool, device=self.top.device))
            self._probe_rgb()  # establish a pre-lift exposure if visible
        for i in range(steps):
            if self.graph is None:
                self._tick()
            else:
                self.graph.replay()
            if phase == 2 and (i + 1) % 10 == 0:
                self.rgb_history[(i // 10) % 32].copy_(self._probe_rgb())
        self.physics_steps += self.count*steps
        n = min(steps//10,32)
        ids = torch.arange(steps//10-n,steps//10,device=self.top.device)%32
        seq = self.samples.index_select(0,ids).transpose(0,1)
        rgb = self.rgb_history.index_select(0,ids).transpose(0,1) if phase == 2 else seq.new_zeros(self.count,n,2)
        return torch.cat((seq, rgb), dim=2)

    def _probe_rgb(self):
        fk = self._pad(self.real).mean(1) + self.deflection
        wrist = CameraPose(fk + fk.new_tensor([0.,-.12,.12]), self.rgb.top_camera.world_from_camera)
        obs = self.rgb.step(self.state.payload, wrist, fk, torch.ones_like(self.zero, dtype=torch.bool),
                            exterior_height=self.top)
        return torch.cat((obs.vis_lift_ratio, obs.vis_lift_valid.to(self.top.dtype)), dim=1)

    def touch(self):
        """Return dimensionless [B,T,18] float32 tactile/geometry features."""
        self.attempt += 1
        self.best.zero_()
        self.airborne.zero_()
        self.dropped.zero_()
        self.net.zero_()
        self.peak_press.zero_()
        if self.variant=="ablation_no_touchdown":
            seq = self.run(4,100)
        else:
            seq = self.run(0,100)
        return self.features(seq,False)

    def grasp(self, mode, action):
        """Int64 [B] top/side, float32 [B,4] controls in [0,1]; return micro history."""
        self.mode.copy_(mode[:,None])
        self.a.copy_(action)
        close = self.run(1,600)
        if self.variant=="ablation_no_microlift":
            return self.features(close[:,-1:],False)
        # Hold long enough for the maximum 120 ms camera delay and a 300 ms decision window.
        steps = 10 * math.ceil((PROBE_RAMP_S + PROBE_SETTLE_S) / (10*DT))
        return self.features(self.run(2,steps),True)

    def finish(self, route, action):
        """Int64 [B] route and [B,2] float32 controls in [0,1]; return bool [B] success."""
        self.route.copy_(route[:,None])
        self.b.copy_(action)
        # Allow the slowest commanded lift to complete before the physical dwell.
        if (route!=2).any():
            self.run(3,4050)
        return (self.best[:,0]>=1500)&self.finite[:,0]

    def rewind(self):
        """Release both pads and bring them back above the current coarse box estimate."""
        self.run(4,750)
        self.origin[:,:2].copy_(self.state.payload.pos[:,:,:2].mean(dim=1))
        self.request[:,0].zero_()
        self.real[:,0].zero_()

    def features(self, seq, micro):
        """[B,T,10] float32 measured frames -> [B,T,18] dimensionless actor features."""
        f,tau,z,h,p,room,hx,hy,gamma,visible = seq.split(1,dim=2)
        gain_ref = torch.maximum(p,torch.ones_like(p)*2.)
        df = torch.diff(f,dim=1,prepend=f[:,:1])
        dz = z-(self.probe_start_z[:,None] if micro else z[:,:1])
        compress = ((-dz/.015)/((f-f[:,:1]).abs()/2.+.05)).clamp(-5,5)
        tension = (-f).clamp_min(0.)
        reference = tension[:,:min(seq.shape[1],20)].amax(dim=1,keepdim=True)
        keep = tension/(reference+.03*gain_ref)
        current_delta = (self.probe_start_load[:,None] - f)/gain_ref if micro else torch.zeros_like(f)
        if self.variant=="ablation_no_history":
            compress,df,keep,dz,current_delta,gamma = (torch.zeros_like(f) for _ in range(6))
        coarse_z = self.top[:,None].expand_as(f)/.16
        prior = (self.mode[:,None].expand_as(f) if self.attempt>1 else torch.full_like(f,-1.))
        feature = torch.cat((compress,df/(.02*gain_ref)*.1,keep,tension/gain_ref,
                           tau/(.03*gain_ref),h/hx.clamp_min(.01),
                           torch.diff(h,dim=1,prepend=h[:,:1])/.002,dz/(self.probe_height[:,None] if micro else .025),
                           hx/.19,hy/.15,room/.10,torch.full_like(f,float(micro)),coarse_z,prior,torch.full_like(f,self.attempt/3.),current_delta,gamma,visible),2).clamp(-8,8)
        return feature[:,-1:] if self.variant=="ablation_no_history" else feature

    def privileged(self):
        """Return float32 [B,70] critic-only normalized X, deformation, proxy and hardware."""
        scale = self.top.new_tensor([.02,1.,.04,8.,.06,30.])
        return torch.cat((self.sim.get_privileged_state(self.state),(self.state.contacts/scale).flatten(1),
                          self.e.force_gain,self.e.current_gain,self.e.sensor_delay[:,None]/25,
                          self.e.control_delay[:,None]/25),1)

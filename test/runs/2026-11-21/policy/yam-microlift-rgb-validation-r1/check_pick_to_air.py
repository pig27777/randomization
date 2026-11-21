import sys

import torch
from src.low_order_parcel.domain_randomizer import DomainRandomizer
from src.low_order_parcel.envs.pick_to_air_env import PickToAirEnv
from src.low_order_parcel.rl.asymmetric_ppo import MacroAction
from src.low_order_parcel.lift_probe import probe_height

class RecordedPick(PickToAirEnv):
    def _tick(self, cmd, active, **kwargs):
        if kwargs.get("lift") and not hasattr(self, "probe_start"):
            self.probe_start = self.command.center[:,0,2].clone()
        super()._tick(cmd, active, **kwargs)
    def _sequence(self, phase, mask, target):
        super()._sequence(phase, mask, target)
        if phase == 1:
            sensed_z = self.sensed.force[:,0,2] - self.tare[:,0,2]
            self.expected_delta = ((self.load_reference - sensed_z) / target.clamp_min(.25)).clamp(-3.,3.)
            self.expected_gamma = torch.where(self.rgb.vis_lift_valid, self.rgb.vis_lift_ratio, -1.).squeeze(-1)

torch.set_num_threads(1)
torch.manual_seed(17)
device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
with torch.no_grad():
    env = RecordedPick(DomainRandomizer(device,seed=17).sample(2))
    env.reset(forced_mode=torch.arange(2,device=device))
    result=env.step(MacroAction(torch.arange(2,device=device), torch.zeros((2,4),device=device)))
    travel=env.command.center[:,0,2]-env.probe_start
    torch.testing.assert_close(travel, probe_height(env.height))
    assert travel.min()>.025 and result.obs.seq.shape==(2,28,12)
    assert result.obs.valid_mask.all() and result.obs.phase.tolist()==[1,1]
    torch.testing.assert_close(result.obs.seq[:,-1,8],env.expected_delta)
    torch.testing.assert_close(result.obs.seq[:,-1,11],env.expected_gamma)
    print("height_mm", (env.height*1000).tolist(), "stroke_mm",(travel*1000).tolist())
    print("features",list(result.obs.seq.shape),"delta",env.expected_delta.tolist(),"gamma",env.expected_gamma.tolist())
    env.rgb = env.rgb._replace(vis_lift_valid=torch.zeros_like(env.rgb.vis_lift_valid))
    env._sequence(1,env.flags(True),env.grasp_force)
    assert (env.history[:,-1,11] == -1.).all()
    print(device, "invalid color observation marked -1")

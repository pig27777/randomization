"""Replay a requested edge target with the articulated Stage 4 pilot checkpoint.

Run with the pilot worktree on PYTHONPATH. Only the horizontal target and
the choice of top approach are prescribed. Record the actual robot and parcel.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.low_order_parcel.envs.yam_pick_to_air_env import YamPickToAirEnv
from src.low_order_parcel.lumped_contact_proxy import ContactState
from src.low_order_parcel.rl.asymmetric_ppo import AsymmetricPPO
from src.low_order_parcel.unified_payload import DT


class EdgeReplay(YamPickToAirEnv):
    def _grasp(self, action, active):
        self.start_xy = self.command.center[:, 0, :2].clone()
        body = self.state.payload
        self.edge_xy = body.pos.mean(1)[:, :2] - self.sim.payload.frame(body)[:, :2, 1] * (
            self.episode.payload.bag_half_extents[:, 1:2] - .004)
        self.close_step = 0
        self.stage = 'close'
        return super()._grasp(action, active)

    def _tick_eager(self, requested, active, lift):
        if self.stage == 'close' and not lift:
            t = min(1., self.close_step * DT / .38)
            center = requested.center.clone()
            center[:, 0, :2] = torch.lerp(self.start_xy, self.edge_xy, .5 - .5 * math.cos(math.pi * t))
            requested = requested._replace(center=center)
            self.close_step += 1
        super()._tick_eager(requested, active, lift)
        self.tick += 1
        phase = 'probe' if lift else self.stage
        b = self.state.payload
        sim = self.sim.payload
        # Every physics step for force, capture and pose. Downsample only for playback.
        frame = sim.frame(b)[0]
        row = torch.cat((self.robot.qpos[0], b.pos[0].flatten(), frame.flatten(),
                         sim.half_thickness(b.d_surf)[0], self.pads[0, 0].flatten(),
                         self.state.contacts_A[0, :, ContactState.CAPTURE], self.result.table_clearance[0],
                         self.result.support_N[0], self.result.press_N[0].flatten(),
                         self.command.center[0, 0]))
        self.samples.append(row)
        self.phases.append(phase)


@torch.inference_mode()
def run(checkpoint, out):
    settings = json.loads((checkpoint / 'config.json').read_text())
    training = json.loads((checkpoint / 'result.json').read_text())
    digest = hashlib.sha256((checkpoint / 'best.pt').read_bytes()).hexdigest()
    assert digest == training['checkpoint_sha256']
    dr = DomainRandomizer('cuda', 88003)
    episode = CornerSuite(dr).select(torch.tensor([4], device='cuda'))
    env = EdgeReplay(episode, urdf=settings['yam_urdf'], seed=88004, cuda_graph=False)
    actor = AsymmetricPPO(env.static_size, env.seq_size, env.priv_size, env.decisions).cuda().eval()
    weights = torch.load(checkpoint / 'best.pt', map_location='cuda', weights_only=True)
    actor.load_state_dict(weights['model'])
    env.stage, env.tick, env.samples, env.phases = 'touch', 0, [], []
    obs = env.reset(forced_mode=torch.zeros(1, device='cuda', dtype=torch.long))
    assert env.robot.assets == training['corners']['C5']['yam']['urdf']
    choices = []
    for _ in range(6):
        if env.done.item():
            break
        env.stage = 'verify' if obs.phase.item() else 'touch'
        act = actor.act(obs, deterministic=True).action
        choices.append([act.choice.item(), act.continuous[0].tolist()])
        obs = env.step(act).obs
    assert env.done.item()
    samples = torch.stack(env.samples).cpu().numpy()
    assert np.isfinite(samples).all()
    extents = episode.payload.bag_half_extents[0].tolist()
    np.savez_compressed(out / 'trajectory.npz', qpos=samples[:,:8], centers=samples[:,8:20].reshape(-1,4,3),
                        frame=samples[:,20:29].reshape(-1,3,3), half_height=samples[:,29:33],
                        pads=samples[:,33:39].reshape(-1,2,3), capture=samples[:,39:41],
                        clearance=samples[:,41], support=samples[:,42], command=samples[:,-3:],
                        phase=env.phases, half_extents=extents, dt=DT)
    result = dict(checkpoint_sha256=digest, update=weights['update'], model='YamPickToAirEnv',
                  urdf_sha256=env.robot.assets['yam.urdf'], corner='C5', half_extents_m=extents,
                  mass_kg=episode.payload.total_mass.item(), seed=88004, steps=env.tick,
                  success=env.success.item(), dropped=env.dropped.item(), torn=env.torn.item(),
                  choices=choices, overflow=env.overflow_bits.tolist(), edge_target_xy_m=env.edge_xy.tolist())
    (out / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    if (args.out / 'trajectory.npz').exists():
        raise FileExistsError(args.out)
    run(args.checkpoint, args.out)

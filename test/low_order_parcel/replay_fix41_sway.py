"""Replay the original fixed blind-test action vectors with optional neck damping."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.low_order_parcel.envs.pick_to_air_env import PickToAirEnv
from src.low_order_parcel.neck_damping import NeckDampingParams
from src.low_order_parcel.rl.asymmetric_ppo import AsymmetricPPO
from src.low_order_parcel.unified_payload import DT


class SwayReplay(PickToAirEnv):
    def __init__(self, episode, rows):
        super().__init__(episode, seed=88004, cuda_graph=True)
        self.rows = rows
        self.trace = []

    def _stable(self, arm):
        stable = super()._stable(arm)
        rows = self.rows
        body = self.state.payload
        position = self.sim.payload.com(body)[rows]
        speed = self.sim.payload.com_velocity(body)[rows].norm(dim=-1)
        clearance = self.result.table_clearance[rows, 0]
        self.trace.append(torch.cat((position, speed[:, None], clearance[:, None], stable[rows, None]), dim=1))
        return stable


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficient", type=float, default=8.)
    args = parser.parse_args()
    base = Path("test/runs/2026-11-22/policy/stage4-policy1-ours-s17/training/best.pt")
    rows = torch.tensor([0, 5, 6], device="cuda")
    ids = torch.arange(4096, device="cuda") % 8
    episode = CornerSuite(DomainRandomizer("cuda", 88003)).select(ids)
    model = AsymmetricPPO(PickToAirEnv.static_size, PickToAirEnv.seq_size, PickToAirEnv.priv_size,
                          PickToAirEnv.decisions).cuda().eval()
    model.load_state_dict(torch.load(base, map_location="cuda", weights_only=True)["model"])
    off = PickToAirEnv(replace(episode, neck_damping=NeckDampingParams()), seed=88004, cuda_graph=True)
    initial = off.reset()
    grasp = model.act(initial, deterministic=True).action
    verification = model.act(off.step(grasp).obs, deterministic=True).action
    print("Same actions for both runs", grasp.choice[rows].tolist(), grasp.continuous[rows].tolist(),
          verification.choice[rows].tolist(), flush=True)
    del off, initial
    params = episode.neck_damping._replace(coefficient_Ns_m=args.coefficient)
    env = SwayReplay(replace(episode, neck_damping=params), rows)
    env.reset()
    env.step(grasp)
    env.step(verification)
    trace = torch.stack(env.trace).cpu()
    for i, corner in enumerate(("C1", "C6", "C7")):
        data = trace[:, i]
        passed = bool(env.success[rows[i]])
        good = data[:, 5] > .5
        indices = torch.arange(len(data))
        streak = indices - torch.where(good, -1, indices).cummax(dim=0).values
        first_hold = (streak >= 1500).nonzero()
        stop = int(first_hold[0, 0]) + 1 if len(first_hold) else len(data)
        last = data[max(0, stop - 1500):stop]
        print(json.dumps(dict(corner=corner, row=int(rows[i]), success=passed, dropped=bool(env.dropped[rows[i]]),
                              torn=bool(env.torn[rows[i]]), longest_hold_s=min(1500., float(streak.max())) * DT,
                              episode_last3_speed_p50_p95_max_mm_s=(1000. * torch.quantile(
                                  last[:, 3], torch.tensor([.5, .95, 1.]))).tolist(),
                              episode_last3_com_x_span_mm=1000. * float(last[:, 0].max() - last[:, 0].min()),
                              episode_last3_min_clearance_mm=1000. * float(last[:, 4].min()))), flush=True)


if __name__ == "__main__":
    main()

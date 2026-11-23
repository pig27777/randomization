"""Run the archived pad-probe input on the C5 parcel used for PPO."""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.low_order_parcel.dual_arm_parcel_sim import DualArmAction
from src.low_order_parcel.lumped_contact_proxy import ContactState
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim, ParcelAction
from src.low_order_parcel.unified_payload import DT


@torch.inference_mode()
def run(out):
    e = CornerSuite(DomainRandomizer('cpu', 88003)).select(torch.tensor([4]))
    half = e.payload.bag_half_extents[0]
    y = -float(half[1])
    mount = torch.tensor([[[[-.024, y], [.024, y]], [[-.024, 0.], [.024, 0.]]]])
    sim = e.make_sim(mount_xy=mount)
    state = e.table_state(sim)
    probe = MacroParcelSim(sim.proxy_A, payload=sim.payload)
    times, _, _, inputs = probe.probe_actions('side', duration_s=6., grip_N=6.)
    position = inputs.pinch_position.clone()
    position[..., 1] += y
    inputs = inputs._replace(pinch_position=position)
    inactive = ParcelAction(*(torch.zeros_like(v[0]) for v in inputs))
    active = torch.tensor([[True, False]])
    orientations = torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]])
    angular = torch.zeros(1, 2, 3)
    values = []
    for i in range(len(times)):
        pad = ParcelAction(*(v[i] for v in inputs))
        result = sim.step(state, DualArmAction(pad, inactive, active, active),
                          jaw_orientation=orientations, jaw_angular_velocity=angular)
        state = result.state
        b = state.payload
        values.append(torch.cat((b.pos.flatten(), sim.payload.frame(b).flatten(),
                                 sim.payload.half_thickness(b.d_surf).flatten(), position[i].flatten(),
                                 state.contacts_A[0, :, ContactState.CAPTURE], result.table_clearance.flatten(),
                                 result.support_N.flatten(), state.torn_flag.flatten().float())))
    a = torch.stack(values).numpy()
    assert np.isfinite(a).all() and not (a[0,31:33] > .5).any()
    np.savez_compressed(out, centers=a[:,:12].reshape(-1,4,3), frame=a[:,12:21].reshape(-1,3,3),
                        half_height=a[:,21:25], pads=a[:,25:31].reshape(-1,2,3), capture=a[:,31:33],
                        clearance=a[:,33], support=a[:,34], torn=a[:,35:], dt=DT,
                        half_extents=half.numpy(), weights=sim.mount_weights[0,:2].numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run(args.output)

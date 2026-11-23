"""CPU checks for the optional YAM per-pad top-traction capacity."""

import torch

from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.low_order_parcel.dual_arm_parcel_sim import DualArmAction
from src.low_order_parcel.macro_parcel_sim import ParcelAction


def test_top_traction_load_uses_each_contact_once():
    dr = DomainRandomizer("cpu", 17)
    # Loaded cloth, one elevated finger, open jaws, stopped jaws, hard book,
    # no parcel contact, and a side-routed cloth contact.
    fixture = CornerSuite(dr).select(torch.tensor([4, 4, 4, 4, 3, 4, 4]))
    sim = fixture.make_sim()
    state = fixture.table_state(sim)
    pad = dr.tensor([[[-.043, 0., .035], [.043, 0., .035]]]).expand(7, -1, -1).clone()
    pad[1, 1, 2] = .18
    pad[5, :, 2] = .18
    pad[4, :, 2] = .026
    inward = dr.tensor([[.03, .03]]).expand(7, -1).clone()
    inward[2, 1] = -.03
    inward[3] = 0.
    zero = torch.zeros_like(inward)
    cmd = ParcelAction(pad, torch.zeros_like(pad), zero, inward, zero, zero,
                       torch.zeros_like(pad[:, 0]), zero[:, :1])
    enabled = torch.tensor([[True, False]]).expand(7, -1)
    side_route = torch.zeros_like(enabled)
    side_route[6, 0] = True
    action = DualArmAction(cmd, cmd, enabled, side_route)
    baseline = sim.step(state, action)
    normal = torch.empty((7, 2, 2))
    mapped = sim.step(state, action, map_top_pad_load=True, proxy_normal_out=normal)
    torch.testing.assert_close(normal[0, 0], baseline.press_N[0, 0].expand(2) * .5)
    assert normal[1, 0, 0] > 0. and normal[1, 0, 1] == 0.
    assert (normal[2:] == 0.).all() and (normal[:, 1] == 0.).all()
    # From the same uncaptured state this adds no force, torque, or motion.
    torch.testing.assert_close(mapped.wrist_wrench, baseline.wrist_wrench, rtol=0., atol=0.)
    torch.testing.assert_close(mapped.state.payload.pos, baseline.state.payload.pos, rtol=0., atol=0.)

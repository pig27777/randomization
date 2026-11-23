from types import SimpleNamespace

import torch

from src.isaac_yam_parcel.yam_parcel_base_env import DT, YAMParcelBaseEnv

gap = torch.tensor([.020, .014, .010, .005, .001, .010, .010]).unsqueeze(-1)
cases = len(gap)
axis = torch.tensor([1., 0., 0.])
pads = torch.stack((-gap * .5 * axis, gap * .5 * axis), dim=1)
vel = torch.zeros_like(pads)
vel[:, 0, 0] = .025
vel[:, 1, 0] = -.025
vel[5] *= -1.
jac = torch.zeros((cases, 2, 3, 8))
jac[:, 0, 0, 6] = 1.
jac[:, 1, 0, 7] = -1.
mass = torch.eye(8).expand(cases, -1, -1) * .069657
fold_work = torch.zeros(cases, 2, 1)
fold_reaction = torch.zeros(cases, 2, 2, 3)
model = SimpleNamespace(side_fold_damping=120., side_fold_gap=.015,
                        film_gap=.002, fold_work=fold_work, fold_reaction=fold_reaction)
robot = SimpleNamespace(data=SimpleNamespace(mass_matrix=SimpleNamespace(torch=mass)))
port_state = SimpleNamespace(pads=pads, pad_velocity=vel, pad_jacobian=jac)
active = torch.tensor([1, 1, 1, 1, 1, 1, 0], dtype=torch.bool).unsqueeze(-1)
force = YAMParcelBaseEnv._compress_side_fold(model, 0, robot, port_state, gap, active)
reaction = fold_reaction[:, 0]
total_force = reaction.sum(dim=1)
total_moment = torch.linalg.cross(pads, reaction).sum(dim=1)
assert (total_force == 0.).all() and (total_moment == 0.).all()
assert (fold_work <= 0.).all()
assert (force[[0, 4, 5, 6]] == 0.).all()
assert (force[1:4] > 0.).all()
assert ((2. / .069657) * DT * force[1:4] < .05).all()
print("effective force N", force.flatten().tolist())
print("discrete work mJ", (1000. * fold_work[:, 0]).flatten().tolist())
print("maximum net pair force N", total_force.abs().max().item())
print("maximum net pair moment Nm", total_moment.abs().max().item())

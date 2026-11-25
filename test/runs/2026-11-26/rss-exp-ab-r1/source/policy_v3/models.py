"""The frozen V3-MVP model architecture."""

import torch
from torch import nn
from torch.nn import functional as F

from .history import HISTORY_DIM


def mlp(input_dim, width, output_dim):
    return nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(), nn.Linear(width, output_dim))


class ProprioHistoryEncoder(nn.Module):
    def __init__(self, pooling="mean_max"):
        super().__init__()
        if pooling != "mean_max":
            raise ValueError("Only mean_max pooling is implemented in V3-MVP")
        self.convs = nn.ModuleList(nn.Conv1d(cin, 64, 3, dilation=dilation) for cin, dilation in
                                   ((HISTORY_DIM, 1), (64, 2), (64, 4)))
        self.action = nn.Sequential(nn.Linear(13, 32), nn.SiLU())
        self.project = nn.Sequential(nn.Linear(128 + 32, 64), nn.LayerNorm(64))

    def forward(self, history, padding_mask, a_try):
        mask = padding_mask[:, None]
        x = history.transpose(1, 2).masked_fill(mask, 0.)
        for layer in self.convs:
            x = F.silu(layer(F.pad(x, (2 * layer.dilation[0], 0)))).masked_fill(mask, 0.)
        mean = x.sum(dim=-1) / (~mask).sum(dim=-1).clamp_min(1)
        maximum = x.masked_fill(mask, -torch.inf).amax(dim=-1)
        maximum = torch.where(padding_mask.all(dim=1, keepdim=True), 0., maximum)
        return self.project(torch.cat((mean, maximum, self.action(a_try)), dim=-1))


class GraspScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(37 + 64 + 11, 96), nn.SiLU(), nn.Linear(96, 64),
                                 nn.SiLU(), nn.Linear(64, 1))
        self.register_buffer("grasp_scale", torch.tensor([12.] * 3 + [1.] * 6 + [12., 1.]))

    def logits(self, public, history, grasp):
        return self.net(torch.cat((public, history, grasp * self.grasp_scale), dim=-1)).squeeze(-1)

    def forward(self, public, history, grasp):
        return torch.sigmoid(self.logits(public, history, grasp))


def tanh_log_prob(distribution, u):
    # Reuse u even when tanh(u) has rounded to exactly -1 or +1.
    correction = torch.log(1. - torch.tanh(u).square() + 1e-6)
    return (distribution.log_prob(u) - correction).sum(dim=-1)


class ProbeActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(12, 64, 13)
        self.log_std = nn.Parameter(torch.full((13,), -1.))
        nn.init.uniform_(self.net[-1].weight, -.01, .01)
        nn.init.zeros_(self.net[-1].bias)

    def distribution(self, v0):
        return torch.distributions.Normal(self.net(v0), self.log_std.clamp(-5, 1).exp())

    def sample(self, v0, generator=None, *, deterministic=False):
        dist = self.distribution(v0)
        u = dist.mean if deterministic else dist.mean + dist.stddev * torch.randn(
            dist.mean.shape, generator=generator, dtype=v0.dtype, device=v0.device)
        return u, torch.tanh(u), tanh_log_prob(dist, u)


class PrivilegedCritic(nn.Module):
    def __init__(self, private_dim=14):
        super().__init__()
        self.net = mlp(12 + private_dim, 64, 1)

    def forward(self, s0, v0):
        return self.net(torch.cat((s0, v0), dim=-1)).squeeze(-1)

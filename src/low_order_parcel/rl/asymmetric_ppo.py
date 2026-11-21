"""Asymmetric recurrent PPO for the Stage 4 macro decisions."""

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


@dataclass(frozen=True)
class Decision:
    # One width per choice. A single choice has no categorical contribution.
    widths: tuple[int, ...]


PICK_DECISIONS = (Decision((4, 4)), Decision((0, 2, 0)))
HANDOVER_DECISIONS = (Decision((6,)), Decision((0, 2, 0)), Decision((2,)))


class Observation(NamedTuple):
    static: Tensor
    seq: Tensor
    valid_mask: Tensor
    priv: Tensor
    phase: Tensor
    forced_choice: Tensor  # -1 leaves the active categorical decision unlocked


class MacroAction(NamedTuple):
    choice: Tensor
    continuous: Tensor


class Sample(NamedTuple):
    action: MacroAction
    latent: Tensor
    log_prob: Tensor
    entropy: Tensor
    value: Tensor


class SequenceEncoder(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.gru = nn.GRU(features, 128, batch_first=True)

    def forward(self, seq, valid):
        # Stable compaction also supports missing frames inside the history.
        order = (~valid).to(torch.int32).argsort(dim=1, stable=True)
        ordered = seq.gather(1, order.unsqueeze(-1).expand_as(seq))
        values, _ = self.gru(ordered)
        length = valid.sum(dim=1)
        rows = torch.arange(len(seq), device=seq.device)
        last = values[rows, (length - 1).clamp_min(0)]
        return torch.where(length[:, None] > 0, last, 0.)


class AsymmetricPPO(nn.Module):
    def __init__(self, static_size, seq_size, priv_size, decisions):
        super().__init__()
        self.decisions = tuple(decisions)
        self.action_size = max(width for phase in self.decisions for width in phase.widths)
        self.actor_seq = SequenceEncoder(seq_size)
        self.critic_seq = SequenceEncoder(seq_size)
        self.actor = nn.Sequential(nn.Linear(128 + static_size, 256), nn.Tanh(),
                                   nn.Linear(256, 128), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(128 + static_size + priv_size, 256), nn.Tanh(),
                                    nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 128),
                                    nn.Tanh(), nn.Linear(128, 1))
        self.categorical = nn.ModuleDict()
        self.means = nn.ModuleDict()
        self.log_stds = nn.ParameterDict()
        for phase, decision in enumerate(self.decisions):
            if len(decision.widths) > 1:
                self.categorical[str(phase)] = nn.Linear(128, len(decision.widths))
            for branch, width in enumerate(decision.widths):
                if width:
                    key = f"{phase}_{branch}"
                    self.means[key] = nn.Linear(128, width)
                    self.log_stds[key] = nn.Parameter(torch.full((width,), -.5))

    def value(self, obs):
        features = self.critic_seq(obs.seq, obs.valid_mask)
        return self.critic(torch.cat((features, obs.static, obs.priv), dim=-1)).squeeze(-1)

    def act(self, obs, *, deterministic=False):
        return self._evaluate(obs, deterministic=deterministic)

    def evaluate_actions(self, obs, choice, latent):
        return self._evaluate(obs, choice=choice, latent=latent)

    def _evaluate(self, obs, *, choice=None, latent=None, deterministic=False):
        features = self.actor(torch.cat((self.actor_seq(obs.seq, obs.valid_mask), obs.static), dim=-1))
        log_prob = features.new_zeros(len(features))
        entropy = torch.zeros_like(log_prob)
        sampled = torch.zeros_like(obs.phase)
        params = features.new_zeros((len(features), self.action_size))
        for stage, decision in enumerate(self.decisions):
            active = obs.phase == stage
            forced = obs.forced_choice >= 0
            stage_key = str(stage)
            branch_choice = sampled.new_zeros(len(features))
            if stage_key in self.categorical:
                logits = self.categorical[stage_key](features)
                dist = Categorical(logits=logits, validate_args=False)
                if choice is None:
                    branch_choice = logits.argmax(dim=-1) if deterministic else dist.sample()
                else:
                    branch_choice = choice.clamp(0, len(decision.widths) - 1)
                branch_choice = torch.where(forced, obs.forced_choice.clamp(0, len(decision.widths) - 1),
                                            branch_choice)
                mask = (active & ~forced).to(log_prob.dtype)
                log_prob = log_prob + mask * dist.log_prob(branch_choice)
                entropy = entropy + mask * dist.entropy()
            sampled = torch.where(active, branch_choice, sampled)
            for branch, width in enumerate(decision.widths):
                if not width:
                    continue
                key = f"{stage}_{branch}"
                mean = self.means[key](features)
                normal = Normal(mean, self.log_stds[key].clamp(-5., 2.).exp(), validate_args=False)
                if latent is None:
                    raw = mean if deterministic else normal.rsample()
                else:
                    raw = latent[:, :width]
                mask = (active & (branch_choice == branch)).to(log_prob.dtype)
                log_prob = log_prob + mask * self._squashed_log_prob(normal, raw)
                entropy = entropy - mask * self._squashed_log_prob(normal, normal.rsample())
                params = params + F.pad(mask[:, None] * raw, (0, self.action_size - width))
        return Sample(MacroAction(sampled, params.tanh()), params, log_prob, entropy, self.value(obs))

    @staticmethod
    def _squashed_log_prob(normal, pre_tanh):
        log_jacobian = 2. * (math.log(2.) - pre_tanh - F.softplus(-2. * pre_tanh))
        return (normal.log_prob(pre_tanh) - log_jacobian).sum(dim=-1)

    def loss(self, obs, rollout, returns, advantages, *, clip=.2, value_weight=.5, entropy_weight=.01):
        current = self.evaluate_actions(obs, rollout.action.choice, rollout.latent.detach())
        ratio = (current.log_prob - rollout.log_prob.detach()).exp()
        advantages = advantages.detach()
        surrogate = torch.minimum(ratio * advantages, ratio.clamp(1. - clip, 1. + clip) * advantages)
        clipped_value = rollout.value.detach() + (current.value - rollout.value.detach()).clamp(-clip, clip)
        value_error = torch.maximum((current.value - returns.detach()).square(),
                                    (clipped_value - returns.detach()).square())
        return (-surrogate + value_weight * value_error - entropy_weight * current.entropy).mean()

"""C2 BCE and single-decision C3/C4 PPO updates."""

from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.nn import functional as F

from .data import ReplayBuffer
from .geometry import InfoMode
from .history import EFFORT, HistoryPreprocess, pad_histories
from .models import GraspScorer, PrivilegedCritic, ProbeActor, ProprioHistoryEncoder, tanh_log_prob


@dataclass(frozen=True)
class InformationProfile:
    mode: InfoMode | None = None
    history: str = "all"
    effort: bool = True

    def __post_init__(self):
        if self.mode is not None:
            object.__setattr__(self, "mode", InfoMode(self.mode))
        if self.history not in ("all", "last", "none"):
            raise ValueError("History profile must be all, last, or none")

    def frames(self, history):
        if self.history == "last":
            history = history[-1:]
        if not self.effort:
            history = history.copy()
            history[:, EFFORT] = 0
        return history


def score_candidates(encoder, scorer, context, history, grasps, profile, device="cpu"):
    if not len(grasps):
        return np.empty(0)
    with torch.no_grad():
        public = torch.as_tensor(context.features(), device=device)[None]
        g = torch.as_tensor(context.grasp_features(grasps), device=device)
        if profile.history == "none":
            z = torch.zeros((1, 64), device=device)
        else:
            h, mask = pad_histories([profile.frames(history)], device)
            z = encoder(h, mask, public[:, -13:])
        return scorer(public.expand(len(g), -1), z.expand(len(g), -1), g).cpu().numpy()


def _concordance(scores, labels):
    positive = scores[labels.astype(bool)]
    negative = scores[~labels.astype(bool)]
    if not len(positive) or not len(negative):
        return 0., 0
    ordered = np.sort(negative)
    below = np.searchsorted(ordered, positive)
    through = np.searchsorted(ordered, positive, side="right")
    return float(((below + through) * .5).sum()), len(positive) * len(negative)


class SupervisedTrainEntry:
    def __init__(self, encoder=None, scorer=None, *, device="cpu", lr=.002, profile=InformationProfile(),
                 preprocess=HistoryPreprocess(), seed=0):
        self.device = device
        self.encoder = (ProprioHistoryEncoder() if encoder is None else encoder).to(device)
        self.scorer = (GraspScorer() if scorer is None else scorer).to(device)
        self.profile = profile
        self.preprocess = preprocess
        self.rng = np.random.default_rng(seed)
        self.optimizer = torch.optim.Adam([*self.encoder.parameters(), *self.scorer.parameters()], lr=lr)

    def _forward(self, rows, *, augment=False):
        for row in rows:
            if self.profile.mode in (InfoMode.WORLD_COMMON, InfoMode.V0_ONLY):
                provenance = row.probe_info.get("candidate_info_mode", row.info_mode)
                if provenance == InfoMode.V1_ALLOWED:
                    raise ValueError("Recollect shared-world candidates for v1-disabled training")
        contexts = [r.public_context(self.profile.mode) for r in rows]
        public = torch.as_tensor(np.stack([c.features() for c in contexts]), device=self.device)
        grasps = torch.as_tensor(np.stack([c.grasp_features(r.grasp_parameters_world)
                                           for c, r in zip(contexts, rows, strict=True)]), device=self.device)
        if self.profile.history == "none":
            z = torch.zeros((len(rows), 64), device=self.device)
        else:
            h = [r.history() for r in rows]
            if augment:
                h = [self.preprocess(history, self.rng) for history in h]
            h, mask = pad_histories([self.profile.frames(history) for history in h], self.device)
            z = self.encoder(h, mask, public[:, -13:])
        y = torch.tensor([r.task_success for r in rows], dtype=torch.float32, device=self.device)
        return self.scorer.logits(public, z, grasps), y

    @torch.no_grad()
    def evaluate(self, rows, *, batch_size=32):
        rows = ReplayBuffer.bce_records(rows)
        if not rows:
            return dict(examples=0, bce=None, heldout_auc=None, paired_concordance=None, matched_pairs=0)
        logits, labels = zip(*(self._forward(rows[i:i + batch_size])
                               for i in range(0, len(rows), batch_size)), strict=True)
        logits, labels = torch.cat(logits), torch.cat(labels)
        scores, y = logits.cpu().numpy(), labels.cpu().numpy()
        concordant, comparable = _concordance(scores, y)
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[row.probe_state_id].append(i)
        pairs = 0
        matched = 0.
        for indices in groups.values():
            wins, count = _concordance(scores[indices], y[indices])
            matched += wins
            pairs += count
        return dict(examples=len(rows), positives=int(y.sum()), bce=float(F.binary_cross_entropy_with_logits(
                    logits, labels)), heldout_auc=concordant / comparable if comparable else None,
                    paired_concordance=matched / pairs if pairs else None, matched_pairs=pairs)

    def fit(self, splits, *, epochs=10, batch_size=32):
        from .data import assert_disjoint

        assert_disjoint(splits)
        rows = ReplayBuffer.bce_records(splits["train"])
        if not rows or epochs < 1 or batch_size < 1:
            raise ValueError("C2 requires valid training grasps and a positive update budget")
        initial = self.evaluate(rows, batch_size=batch_size)
        losses = []
        for _ in range(epochs):
            ids = self.rng.permutation(len(rows))
            running = 0.
            # Split evenly so a one-example tail cannot dominate an Adam step.
            for chunk in np.array_split(ids, math.ceil(len(ids) / batch_size)):
                batch = [rows[j] for j in chunk]
                logits, labels = self._forward(batch, augment=True)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite scorer BCE")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                running += loss.item() * len(batch)
            losses.append(running / len(rows))
        return dict(initial_train=initial, train=self.evaluate(rows, batch_size=batch_size), epoch_bce=losses,
                    val=self.evaluate(splits["val"], batch_size=batch_size),
                    test=self.evaluate(splits["test"], batch_size=batch_size))


class ContextualPPOEntry:
    def __init__(self, actor=None, critic=None, *, device="cpu", actor_lr=3e-4, critic_lr=.001,
                 clip=.2, seed=0):
        self.device = device
        self.actor = (ProbeActor() if actor is None else actor).to(device)
        self.critic = (PrivilegedCritic() if critic is None else critic).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.clip = clip
        self.rng = np.random.default_rng(seed)

    def update(self, buffer, *, epochs=4, batch_size=32):
        transitions = buffer.transitions
        metrics = dict(policy_version=buffer.policy_version, examples=len(transitions),
                       physics_invalid_rate=buffer.physics_invalid_rate,
                       ppo_excluded_physics_invalid=buffer.excluded_count)
        if not transitions:
            return metrics
        if epochs < 1 or batch_size < 1:
            raise ValueError("PPO update budgets must be positive")

        def field(name):
            return torch.as_tensor(np.stack([getattr(t, name) for t in transitions]),
                                   device=self.device, dtype=torch.float32)

        v0, priv, u, old, reward = (field(name) for name in ("v0", "private_s0", "u", "old_log_prob", "reward"))
        with torch.no_grad():
            pred = self.critic(priv, v0)
            advantage = reward - pred
            check = tanh_log_prob(self.actor.distribution(v0), u)
            if not torch.allclose(check, old, atol=2e-4, rtol=2e-4):
                raise ValueError("On-policy likelihood differs from the frozen actor")
        actor_total = critic_total = 0.
        for _ in range(epochs):
            ids = self.rng.permutation(len(transitions))
            for i in range(0, len(ids), batch_size):
                idx = torch.as_tensor(ids[i:i + batch_size], device=self.device)
                new = tanh_log_prob(self.actor.distribution(v0[idx]), u[idx])
                ratio = (new - old[idx]).clamp(-20, 20).exp()
                unclipped = ratio * advantage[idx]
                clipped = ratio.clamp(1 - self.clip, 1 + self.clip) * advantage[idx]
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                critic_loss = F.mse_loss(self.critic(priv[idx], v0[idx]), reward[idx])
                if not math.isfinite((actor_loss + critic_loss).item()):
                    raise FloatingPointError("Nonfinite PPO loss")
                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .5)
                self.actor_opt.step()
                self.critic_opt.step()
                actor_total += actor_loss.item() * len(idx)
                critic_total += critic_loss.item() * len(idx)
        denominator = epochs * len(transitions)
        return dict(metrics, actor_loss=actor_total / denominator, critic_loss=critic_total / denominator,
                    mean_reward=float(reward.mean()))

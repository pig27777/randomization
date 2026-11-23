"""Eight-frame joint actor, asymmetric critic, and BC/PPO updates."""

import torch
from torch import nn
from torch.distributions import Normal


HISTORY = 8


def network(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
                         nn.Linear(256, outputs))


class JointActor(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.body = network(HISTORY * obs_dim, 7)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def forward(self, history):
        normalized = ((history - self.obs_mean) / self.obs_std).clamp(-10., 10.)
        return self.body(normalized.flatten(1)).tanh()


class JointPolicy(nn.Module):
    def __init__(self, obs_dim, critic_dim):
        super().__init__()
        self.actor = JointActor(obs_dim)
        self.critic = network(critic_dim, 1)
        self.log_std = nn.Parameter(torch.full((7,), -2.))
        self.register_buffer("critic_mean", torch.zeros(critic_dim))
        self.register_buffer("critic_std", torch.ones(critic_dim))

    def distribution(self, history):
        return Normal(self.actor(history), self.log_std.clamp(-5., 0.).exp())

    def value(self, obs):
        return self.critic(((obs - self.critic_mean) / self.critic_std).clamp(-10., 10.)).squeeze(1)


class ActorHistory:
    def __init__(self, obs):
        self.tensor = obs.unsqueeze(1).expand(-1, HISTORY, -1).clone()

    def advance(self, obs, done):
        windows = torch.cat((self.tensor[:, 1:], obs.unsqueeze(1)), dim=1)
        self.tensor = torch.where(done[:, None, None], obs.unsqueeze(1), windows)


def demonstration_batch(data, size):
    steps, batch = data["policy"].shape[:2]
    device = data["policy"].device
    lane = torch.randint(batch, (size,), device=device)
    time = (torch.rand(size, device=device) * data["lengths"][lane].clamp(1, steps)).long()
    window = (time[:, None] - torch.arange(HISTORY - 1, -1, -1, device=device)).clamp_min(0)
    return (data["policy"][window, lane[:, None]].float(), data["critic"][time, lane].float(),
            data["action"][time, lane].float())


def behavior_clone(model, data, *, updates, minibatch):
    with torch.no_grad():
        obs, critic, _ = demonstration_batch(data, 32768)
        model.actor.obs_mean.copy_(obs[:, -1].mean(dim=0))
        model.actor.obs_std.copy_(obs[:, -1].std(dim=0).clamp_min(.005))
        model.critic_mean.copy_(critic.mean(dim=0))
        model.critic_std.copy_(critic.std(dim=0).clamp_min(.005))
    opt = torch.optim.Adam(model.actor.parameters(), lr=4e-4)
    loss = obs.new_zeros(())
    for _ in range(updates):
        history, _, target = demonstration_batch(data, minibatch)
        error = (model.actor(history) - target).square()
        loss = error[:, :6].mean() + error[:, 6].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
        opt.step()
    return loss.item()


@torch.inference_mode()
def rollout(env, model, obs, history, steps):
    storage = {name: [] for name in ("history", "critic", "action", "log_prob", "value", "reward", "done")}
    successes = torch.zeros((), dtype=torch.long, device=env.device)
    ends = torch.zeros_like(successes)
    for _ in range(steps):
        distribution = model.distribution(history.tensor)
        raw = distribution.sample()
        value = model.value(obs["critic"])
        critic_obs = obs["critic"]
        next_obs, reward, terminated, timeout, _ = env.step(raw.clamp(-1., 1.))
        done = terminated | timeout
        successes += (terminated & (reward > 0.)).sum()
        ends += done.sum()
        for name, tensor in zip(storage, (history.tensor, critic_obs, raw, distribution.log_prob(raw).sum(dim=-1),
                                         value, reward, done), strict=True):
            storage[name].append(tensor.clone())
        obs = next_obs
        history.advance(obs["policy"], done)
    trajectory = {name: torch.stack(items) for name, items in storage.items()}
    next_value = model.value(obs["critic"])
    accumulated = torch.zeros_like(next_value)
    advantage = torch.empty_like(trajectory["reward"])
    for index in reversed(range(steps)):
        live = ~trajectory["done"][index]
        td = trajectory["reward"][index] + .99 * next_value * live - trajectory["value"][index]
        accumulated = td + .99 * .95 * live * accumulated
        advantage[index] = accumulated
        next_value = trajectory["value"][index]
    trajectory["advantage"] = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
    trajectory["return"] = advantage + trajectory["value"]
    return {name: value.flatten(0, 1) for name, value in trajectory.items()}, obs, history, successes, ends


def ppo_update(model, optimizer, trajectory, *, minibatch, epochs, critic_only):
    # Rollout tensors were allocated during inference; gather fresh training tensors.
    size = len(trajectory["action"])
    for _ in range(epochs):
        order = torch.randperm(size, device=trajectory["action"].device)
        for index in order.split(minibatch):
            value = model.value(trajectory["critic"][index])
            loss = .5 * (value - trajectory["return"][index]).square().mean()
            if not critic_only:
                dist = model.distribution(trajectory["history"][index])
                log_ratio = dist.log_prob(trajectory["action"][index]).sum(dim=-1) - trajectory["log_prob"][index]
                ratio = log_ratio.exp()
                adv = trajectory["advantage"][index]
                loss += -torch.minimum(ratio * adv, ratio.clamp(.9, 1.1) * adv).mean()
                loss -= .001 * dist.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            if not critic_only and (ratio - 1. - log_ratio).mean().item() > .02:
                return loss.item()
    return loss.item()

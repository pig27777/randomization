"""Contextual bandit on grasp parameters: an ensemble of small MLPs predicts P(success | observation, action),
the policy picks the action with the highest predicted success (random candidates + CEM on the model).

Numpy only (no PyTorch in the shared environments). R1 and R3 use exactly this code and the same settings.
"""
from __future__ import annotations

import math

import numpy as np

from env import ACC_RANGE, F_RANGE, VLIFT_RANGE

OBS_REF = np.log([0.25, 0.18, 0.06])
HIDDEN = 64
ENSEMBLE = 5


# ------------------------------------------------------------------ action coding: z in [-1, 1]^6

def decode(z):
    """z (..., 6) -> grasp params (u, v, psi, F, v_lift, a_max)."""
    z = np.clip(np.asarray(z, float), -1, 1)
    lf = np.log(F_RANGE)
    return np.stack([z[..., 0] / 2, z[..., 1] / 2, (z[..., 2] + 1) / 2 * math.pi,
                     np.exp(lf[0] + (z[..., 3] + 1) / 2 * (lf[1] - lf[0])),
                     VLIFT_RANGE[0] + (z[..., 4] + 1) / 2 * (VLIFT_RANGE[1] - VLIFT_RANGE[0]),
                     ACC_RANGE[0] + (z[..., 5] + 1) / 2 * (ACC_RANGE[1] - ACC_RANGE[0])], -1)


def encode(params):
    p = np.asarray(params, float)
    lf = np.log(F_RANGE)
    return np.stack([p[..., 0] * 2, p[..., 1] * 2, p[..., 2] / math.pi * 2 - 1,
                     (np.log(p[..., 3]) - lf[0]) / (lf[1] - lf[0]) * 2 - 1,
                     (p[..., 4] - VLIFT_RANGE[0]) / (VLIFT_RANGE[1] - VLIFT_RANGE[0]) * 2 - 1,
                     (p[..., 5] - ACC_RANGE[0]) / (ACC_RANGE[1] - ACC_RANGE[0]) * 2 - 1], -1)


def features(obs, z):
    """Network input: log bounding box and the action, psi as (cos 2psi, sin 2psi) (period pi)."""
    obs = np.broadcast_to(obs, z.shape[:-1] + (3,))
    psi = (z[..., 2] + 1) / 2 * math.pi
    return np.concatenate([(np.log(obs) - OBS_REF) / 0.5, z[..., :2], np.cos(2 * psi)[..., None],
                           np.sin(2 * psi)[..., None], z[..., 3:]], -1)


def augment(obs, z, y):
    """Box symmetries (D2): mirror about either center line, which also mirrors psi. Valid because the COM offset
    and the carry direction are symmetric in both training distributions."""
    out_o, out_z, out_y = [obs], [z], [y]
    for su, sv in ((-1, 1), (1, -1), (-1, -1)):
        z2 = z.copy()
        z2[:, 0] *= su
        z2[:, 1] *= sv
        if su * sv < 0:
            z2[:, 2] = -z2[:, 2]                         # psi -> pi - psi
        out_o.append(obs)
        out_z.append(z2)
        out_y.append(y)
    return np.concatenate(out_o), np.concatenate(out_z), np.concatenate(out_y)


# ------------------------------------------------------------------ MLP ensemble

def _init(rng, sizes):
    return [(rng.normal(0, 1 / math.sqrt(a), (a, b)), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]


def _forward(params, x):
    hs = [x]
    for W, b in params[:-1]:
        hs.append(np.tanh(hs[-1] @ W + b))
    W, b = params[-1]
    return hs, (hs[-1] @ W + b)[:, 0]


def train_net(x, y, rng, epochs=300, batch=256, lr=3e-3, wd=1e-4):
    params = _init(rng, [x.shape[1], HIDDEN, HIDDEN, 1])
    m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for i in range(0, len(x), batch):
            idx = order[i:i + batch]
            hs, logit = _forward(params, x[idx])
            p = 1 / (1 + np.exp(-logit))
            g = ((p - y[idx]) / len(idx))[:, None]            # d BCE / d logit
            grads = []
            for li in range(len(params) - 1, -1, -1):
                W, b = params[li]
                grads.append((hs[li].T @ g + wd * W, g.sum(0)))
                if li > 0:
                    g = (g @ W.T) * (1 - hs[li] ** 2)
            grads.reverse()
            step += 1
            for li, ((W, b), (gW, gb)) in enumerate(zip(params, grads)):
                mW, mb = m[li]
                vW, vb = v[li]
                mW, mb = 0.9 * mW + 0.1 * gW, 0.9 * mb + 0.1 * gb
                vW, vb = 0.999 * vW + 0.001 * gW ** 2, 0.999 * vb + 0.001 * gb ** 2
                m[li], v[li] = (mW, mb), (vW, vb)
                c1, c2 = 1 - 0.9 ** step, 1 - 0.999 ** step
                params[li] = (W - lr * (mW / c1) / (np.sqrt(vW / c2) + 1e-8),
                              b - lr * (mb / c1) / (np.sqrt(vb / c2) + 1e-8))
    return params


class Model:
    def __init__(self, nets):
        self.nets = nets

    @staticmethod
    def fit(obs, z, y, seed=0, **kw):
        obs, z, y = augment(np.asarray(obs, float), np.asarray(z, float), np.asarray(y, float))
        x = features(obs, z)
        rng = np.random.default_rng(seed)
        nets = []
        for _ in range(ENSEMBLE):
            idx = rng.integers(0, len(x), len(x))             # bootstrap
            nets.append(train_net(x[idx], y[idx], rng, **kw))
        return Model(nets)

    def prob(self, obs, z):
        x = features(np.asarray(obs, float), np.asarray(z, float))
        return np.mean([1 / (1 + np.exp(-_forward(p, x)[1])) for p in self.nets], 0)

    def act(self, obs, seed=0, n0=4096, iters=3, n=512, elite=32):
        """Action z maximizing the predicted success for one observation (deterministic for a seed)."""
        rng = np.random.default_rng(seed)
        z = rng.uniform(-1, 1, (n0, 6))
        s = self.prob(obs, z)
        for _ in range(iters):
            e = z[np.argsort(-s)[:elite]]
            mu, sd = e.mean(0), e.std(0) + 0.02
            zn = np.clip(rng.normal(mu, sd, (n, 6)), -1, 1)
            z, s = np.concatenate([e, zn]), np.concatenate([s[np.argsort(-s)[:elite]], self.prob(obs, zn)])
        i = int(np.argmax(s))
        return z[i], float(s[i])

    def save(self, path):
        arrs = {f"n{i}_l{j}_{k}": a for i, net in enumerate(self.nets) for j, (W, b) in enumerate(net)
                for k, a in (("W", W), ("b", b))}
        np.savez(path, **arrs)

    @staticmethod
    def load(path):
        d = np.load(path)
        nets = []
        for i in range(ENSEMBLE):
            layers, j = [], 0
            while f"n{i}_l{j}_W" in d:
                layers.append((d[f"n{i}_l{j}_W"], d[f"n{i}_l{j}_b"]))
                j += 1
            nets.append(layers)
        return Model(nets)

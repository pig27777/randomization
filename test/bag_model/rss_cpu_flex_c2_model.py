"""C2-only FiLM scorers for replaying the archived CPU Flex experiment."""
from copy import deepcopy
from itertools import combinations
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from policy_v3 import geometry, history
from policy_v3.training import _concordance
from rss_cpu_flex_c1c2 import CANDIDATES, FINAL_PROPOSAL, PROBE_END, TICK, TRIES
from rss_cpu_flex_c1c2_policy import action_features, packet, v3_frames

EPS = 1e-6
Z_DIM = 32
N_TICKS = round(PROBE_END / TICK) + 1
# Distinguish the archived short and long sweep from the public action vector.
SPAN_INDEX = -10
SCAN_SPAN = (TRIES["Try_Left"][1] + TRIES["Try_Scan"][1]) / (.20 * 2)
# Preserve the location of contact transients along the public press and sweep schedule.
WINDOWS = tuple(math.ceil(t / TICK) for t in (0., .62, .96, 1.20, 1.54))


def inputs(path, row):
    h, _ = packet(path)

    def read_v1():
        raise AssertionError("C2 V0_ONLY must never request v1")

    public = geometry.PublicContext(geometry.InfoMode.V0_ONLY,
                                    geometry.VisualObservation(**row["v0"]), read_v1,
                                    action_features(row["probe"], row["probe_summary"]["clearance_offset_m"]))
    grasps = public.grasp_features(row["candidates"])
    grasps = np.concatenate((grasps, np.eye(len(CANDIDATES), dtype=np.float32)), axis=1)
    return public.features(), history.encoder_frames(v3_frames(h)), grasps


def training_example(path, row, candidate_names):
    if row["physics_validity"] is not True or type(row["task_success"]) is not bool:
        raise ValueError(f"Unlabeled or physically invalid BCE row: {path}")
    public, frames, candidates = inputs(path, row)
    return public, frames, candidates, candidate_names.index(row["grasp"]), row["task_success"]


class Normalization(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.register_buffer("indices", torch.as_tensor(indices, dtype=torch.long))
        self.register_buffer("frame_mean", torch.zeros(len(indices)))
        self.register_buffer("frame_scale", torch.ones(len(indices)))
        self.register_buffer("command_template", torch.zeros((2, N_TICKS, len(indices))))
        self.register_buffer("residual_scale", torch.ones(len(indices)))
        self.register_buffer("grasp_mean", torch.zeros(11))
        self.register_buffer("grasp_scale", torch.ones(11))

    @classmethod
    def fit(cls, train, mode):
        frames = np.concatenate([x[1] for x in train]).astype(np.float64)
        grasps = np.stack([candidates[index, :11] for _, _, candidates, index, _ in train]).astype(np.float64)
        active = np.any(frames != 0, axis=0)
        if mode == "Blind":
            active[:] = False
        elif mode == "Informed w/o Effort":
            active[history.EFFORT] = False
        out = cls(np.flatnonzero(active))
        frames = frames[:, active]
        for values, mean, scale in ((frames, out.frame_mean, out.frame_scale),
                                    (grasps, out.grasp_mean, out.grasp_scale)):
            mean.copy_(torch.from_numpy(values.mean(axis=0)))
            scale.copy_(torch.from_numpy(values.std(axis=0).clip(min=EPS)))
        if len(out.indices):
            residuals = []
            for command in (0, 1):
                traces = np.stack([x[1][:, active] for x in train
                                   if int(x[0][SPAN_INDEX] > SCAN_SPAN) == command]).astype(np.float64)
                baseline = traces.mean(axis=0)
                out.command_template[command].copy_((torch.from_numpy(baseline) - out.frame_mean)
                                                    / out.frame_scale)
                residuals.append((traces - baseline).reshape(-1, len(out.indices)))
            residual_std = np.concatenate(residuals).std(axis=0).clip(min=EPS)
            out.residual_scale.copy_(torch.from_numpy(residual_std) / out.frame_scale)
        return out


class TryEncoder(nn.Module):
    def __init__(self, input_channels, width):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(cin, width, 3, dilation=dilation)
                                   for cin, dilation in ((input_channels, 1), (width, 2), (width, 4)))
        self.project = nn.Sequential(nn.Linear(2 * width * len(WINDOWS), Z_DIM), nn.LayerNorm(Z_DIM))

    def forward(self, frames, padding_mask):
        mask = padding_mask[:, None]
        x = frames.transpose(1, 2).masked_fill(mask, 0.)
        for i, conv in enumerate(self.convs):
            filtered = F.silu(conv(F.pad(x, (2 * conv.dilation[0], 0)))).masked_fill(mask, 0.)
            x = x + filtered if i else filtered
        tick = torch.arange(x.shape[-1])
        phase = torch.bucketize(tick.contiguous(), torch.tensor(WINDOWS[1:]), right=True)
        valid = (~mask[:, None]) & (phase == torch.arange(len(WINDOWS))[:, None, None])
        phase_x = x[:, None]
        count = valid.sum(dim=-1)
        mean = (phase_x * valid).sum(dim=-1) / count.clamp_min(1)
        maximum = phase_x.masked_fill(~valid, -torch.inf).amax(dim=-1)
        maximum = torch.where(count > 0, maximum, 0.)
        return self.project(torch.cat((mean, maximum), dim=-1).flatten(1))


class GraspScorer(nn.Module):
    def __init__(self, width, informed, dropout):
        super().__init__()
        # V0 geometry and the public Try action are shared by both modalities.
        self.grasp = nn.Sequential(nn.Linear(12 + 13 + 12 + len(CANDIDATES), width), nn.SiLU(),
                                   nn.Linear(width, width))
        self.film = nn.Linear(Z_DIM, 2 * width, bias=False) if informed else None
        if self.film is not None:
            nn.init.normal_(self.film.weight, std=.15)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 2))

    def forward(self, public, z, grasp):
        context = torch.cat((public[:, :12], public[:, -13:]), dim=-1)
        context = context[:, None].expand(-1, grasp.shape[1], -1)
        h = self.grasp(torch.cat((context, grasp, grasp[..., :1].square()), dim=-1))
        h = h - h.mean(dim=1, keepdim=True)
        if self.film is not None:
            gain, beta = self.film(z).chunk(2, dim=-1)
            # Here gamma is gain minus one, so a zero embedding gives zero total gain.
            h = gain[:, None] * h + beta[:, None]
        return self.head(h).unbind(dim=-1)


class C2Model(nn.Module):
    def __init__(self, normalization, width=32, dropout=0.):
        super().__init__()
        self.normalization = normalization
        self.register_buffer("log_proposal", torch.as_tensor(FINAL_PROPOSAL, dtype=torch.float32).log())
        size = len(normalization.indices)
        self.encoder = TryEncoder(size, width) if size else None
        self.scorer = GraspScorer(width, size > 0, dropout)

    def embedding(self, histories, public):
        if self.encoder is None:
            return None
        frames, mask = history.pad_histories(histories)
        norm = self.normalization
        standardized = (frames[..., norm.indices] - norm.frame_mean) / norm.frame_scale
        command = torch.from_numpy(np.asarray(public)[:, SPAN_INDEX] > SCAN_SPAN).long()
        template = norm.command_template[command, :frames.shape[1]]
        residual = (standardized - template) / norm.residual_scale
        active, neutral = self.encoder(torch.cat((residual, torch.zeros_like(residual))),
                                       torch.cat((mask, mask))).chunk(2)
        return F.normalize(active - neutral, dim=-1, eps=EPS) * math.sqrt(Z_DIM)

    def logits(self, public, z, grasps):
        norm = self.normalization
        continuous = (grasps[..., :11] - norm.grasp_mean) / norm.grasp_scale
        return self.scorer(public, z, torch.cat((continuous, grasps[..., 11:]), dim=-1))

    @staticmethod
    def probabilities(success, support):
        # Relative support cancels the unidentifiable per-context acceptance rate.
        return success.sigmoid() * (support - support.amax(dim=-1, keepdim=True)).exp()

    def forward(self, batch):
        public, frames, grasps, selected, labels = zip(*batch, strict=True)
        z = self.embedding(frames, public)
        success, support = self.logits(torch.from_numpy(np.stack(public)), z, torch.from_numpy(np.stack(grasps)))
        return success, support, torch.tensor(selected), torch.tensor(labels, dtype=torch.float32)

    @torch.no_grad()
    def predict(self, sample):
        public, frames, candidates = sample
        z = self.embedding([frames], [public])
        success, support = self.logits(torch.from_numpy(public)[None], z, torch.from_numpy(candidates)[None])
        return self.probabilities(success, support)[0].tolist()

    @torch.no_grad()
    def evaluate(self, examples, batch_size):
        self.eval()
        scores, labels = [], []
        for i in range(0, len(examples), batch_size):
            success, support, selected, y = self(examples[i:i + batch_size])
            scores.append(self.probabilities(success, support)[torch.arange(len(y)), selected])
            labels.append(y)
        scores, labels = torch.cat(scores), torch.cat(labels)
        wins, pairs = _concordance(scores.numpy(), labels.numpy())
        return dict(examples=len(labels), positives=int(labels.sum()),
                    bce=F.binary_cross_entropy(scores, labels).item(),
                    heldout_auc=wins/pairs if pairs else None)

    def training_loss(self, batch, support_weight, weights=None):
        success, support, selected, labels = self(batch)
        selected_success = success[torch.arange(len(labels)), selected]
        supervised = F.binary_cross_entropy_with_logits(selected_success, labels, weight=weights)
        acceptance = F.cross_entropy(support + self.log_proposal, selected)
        return supervised + support_weight * acceptance

    def fit(self, splits, *, seed, epochs, batch_size, patience, lr, scorer_lr, decay, support_weight):
        train, val = splits["train"], splits["val"]
        groups = [{"params": self.scorer.parameters(), "lr": scorer_lr}]
        if self.encoder is not None:
            groups.append({"params": self.encoder.parameters()})
        optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=decay)
        rng = np.random.default_rng(seed)
        candidates = np.array([row[3] for row in train])
        weights_by_row = torch.from_numpy(np.bincount(candidates)[candidates] ** -.5).float()
        weights_by_row /= weights_by_row.mean()
        results = dict(initial_train=self.evaluate(train, batch_size), epochs=[])
        best, best_epoch, weights = (float("inf"), float("inf")), 0, None
        for epoch in range(1, epochs + 1):
            self.train()
            total = 0.
            for chunk in np.array_split(rng.permutation(len(train)), math.ceil(len(train) / batch_size)):
                loss = self.training_loss([train[i] for i in chunk], support_weight, weights_by_row[chunk])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training BCE")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(chunk)
            validation = self.evaluate(val, batch_size)
            if not math.isfinite(validation["bce"]):
                raise FloatingPointError("Nonfinite validation BCE")
            results["epochs"].append(dict(train_objective=total / len(train), val=validation))
            rank = validation["bce"], -validation["heldout_auc"]
            if rank < best:
                best, best_epoch, weights = rank, epoch, deepcopy(self.state_dict())
            if epoch - best_epoch >= patience:
                break
        self.load_state_dict(weights)
        return dict(results, best_epoch=best_epoch, train=self.evaluate(train, batch_size),
                    val=self.evaluate(val, batch_size))

    def diagnostics(self, train, matched, support_weight):
        self.eval()
        self.zero_grad(set_to_none=True)
        self.training_loss(train, support_weight).backward()

        def grad_norm(module):
            return torch.stack([p.grad.square().sum() for p in module.parameters()]).sum().sqrt().item()

        with torch.no_grad():
            z = self.embedding([x[1] for x in matched], [x[0] for x in matched])
            return dict(encoder_grad_l2=grad_norm(self.encoder), film_grad_l2=grad_norm(self.scorer.film),
                        scorer_grad_l2=grad_norm(self.scorer), z_dim=z.shape[-1],
                        z_cross_state_variance=z.var(dim=0, unbiased=False).mean().item(),
                        z_pair_l2=[torch.linalg.vector_norm(a-b).item() for a, b in combinations(z, 2)])

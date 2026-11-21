"""Episode distributions for the frozen two-pad proxy and four-block payload (SI)."""

from dataclasses import dataclass, fields, replace
import json
import math
from pathlib import Path

import torch

from src.low_order_parcel.lumped_contact_proxy import ProxyParams
from src.low_order_parcel.unified_payload import DT, PayloadParamsX

ROOT = Path(__file__).resolve().parents[2]
MATERIALS = ("pe", "kraft", "bubble", "paperboard")
PAYLOADS = ("rigid_box", "folded_clothes", "fluffy_pillow", "heavy_quilt", "loose_shifting")


@dataclass
class Episode:
    proxy: ProxyParams
    x: PayloadParamsX
    # float32 [B,1] except clearances [B,2] in m, and IDs/delays int64 [B].
    material: torch.Tensor
    content: torch.Tensor
    yaw: torch.Tensor
    tilt: torch.Tensor
    height_error: torch.Tensor
    clearance: torch.Tensor
    force_gain: torch.Tensor
    bias: torch.Tensor
    noise: torch.Tensor
    cutoff: torch.Tensor
    sensor_delay: torch.Tensor
    control_delay: torch.Tensor
    current_gain: torch.Tensor
    deadband: torch.Tensor

    def reset_rows(self, indices, sample):
        """Replace int64 [N] episode rows in place with N compatible sampled SI rows."""
        for group in (self, self.proxy, self.x):
            source = sample if group is self else sample.proxy if group is self.proxy else sample.x
            for field in fields(group):
                tensor = getattr(group, field.name)
                if torch.is_tensor(tensor):
                    tensor.index_copy_(0, indices, getattr(source, field.name))


class DomainRandomizer:
    def __init__(self, device, seed):
        self.device = device
        self.rng = torch.Generator(device=device).manual_seed(seed)
        config = json.loads((ROOT / "configs/proxy_material_envelopes.yaml").read_text())
        mats = [config["materials"][key] for key in MATERIALS]
        self.order = mats[0]["dependency"]["order"]
        self.chol = torch.linalg.cholesky(self.tensor([m["dependency"]["correlation"] for m in mats]))
        self.bounds = self.tensor([[m["tier1_ranges"][k] for k in self.order] for m in mats])
        # The file already includes 1.75x expansion. Extend its outer material tails.
        self.bounds[2, 2, 0], self.bounds[1, 2, 1] = 4., 2200.
        self.bounds[3, 1, 0] = .20
        self.bounds[2, 3:5, 0] = .02
        self.nominal = {f.name: self.tensor([m["nominal"]["proxy"][f.name] for m in mats]).reshape(4, -1)
                        for f in fields(ProxyParams)}
        self.xcfg = json.loads((ROOT / "configs/payload_template_X.yaml").read_text())
        self.presets = PayloadParamsX.from_presets(self.xcfg, PAYLOADS, device=device)

    def tensor(self, values):
        """Convert numeric SI array to a same-shaped float32 device tensor."""
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def sample(self, count, *, wide=True, rigid=False, bucket_weights=None, corners=None):
        """Return B=count episode rows. Optional int64 [B] corners use zero-based C1–C8 IDs."""
        if count < 1:
            raise ValueError("count must be positive")
        device, gen = self.device, self.rng
        u = torch.rand(count, 15, device=device, generator=gen)
        material = (4 * u[:, 0]).long()
        content = (5 * u[:, 1]).long()
        if bucket_weights is not None:
            bucket = torch.multinomial(bucket_weights.reshape(-1), count, replacement=True, generator=gen)
            material, content = bucket // 5, bucket % 5
        if corners is not None:
            material = corners.new_tensor([2, 0, 0, 3, 0, 1, 0, 0])[corners]
            content = corners.new_tensor([0, 0, 0, 0, 1, 4, 4, 3])[corners]
        if rigid:
            content = torch.zeros_like(content)
        x = PayloadParamsX.sample(self.xcfg, count, device=device, generator=gen)
        anchor = {f.name: getattr(self.presets, f.name)[content] for f in fields(x)}
        # Vary each archetype continuously, preserving the zero-compliance rigid subspace.
        mix = .55 if wide else .12
        material_fields = ("alpha_bend", "alpha_stretch", "theta_max", "slide_slack_max",
                           "max_indent_ratio", "table_friction", "tau_rebound", "internal_damping",
                           "k_soft", "hardening_gamma", "lateral_crush_k", "c_comp")
        values = {k: torch.where(anchor[k] == 0, 0., torch.lerp(anchor[k], getattr(x, k), mix))
                  for k in material_fields}
        if wide:
            values["mass_weights"] = torch.lerp(x.mass_weights, anchor["mass_weights"], .4)
        else:
            for k in ("bag_half_extents", "total_mass", "mass_weights", "fill_ratio"):
                values[k] = torch.lerp(anchor[k], getattr(x, k), mix)
        x = replace(x, **values)
        nominal = {name: table[material] for name, table in self.nominal.items()}
        z = torch.randn(count, len(self.order), 1, device=device, generator=gen)
        ranks = torch.special.ndtr(torch.bmm(self.chol[material], z).squeeze(2))
        lo, hi = self.bounds[material].unbind(-1)
        sample = dict(zip(self.order, torch.lerp(lo, hi, ranks).split(1, dim=1), strict=True))
        p = dict(nominal)
        ratio = .65 + .70 * ranks[:, 4:5]
        p.update({k: sample[k] for k in ("S_bag", "alpha", "beta", "s_crit", "h_min", "F_min")})
        p["mu_eff"] = torch.cat((sample["mu_L"], sample["mu_L"] * ratio), 1)
        p["k"] = .5 * sample["k"]
        if not wide:
            p = {k: torch.lerp(nominal[k], v, .12) for k, v in p.items()}
        p["s_max"] = torch.maximum(p["s_max"], p["s_crit"])
        p["F_lock"] = torch.maximum(p["F_lock"], p["F_min"] + .05)
        p["c_d"] = nominal["c_d"] * (p["k"] / nominal["k"]).sqrt()
        yaw, tilt, error = ((2 * u[:, j:j+1] - 1) * span * (1 if wide else .2)
                             for j, span in ((2, .6), (3, .08), (4, .005)))
        clearance = .10 * u[:, 5:7] if wide else .06 + .04 * u[:, 5:7]
        hw = Episode(ProxyParams(**p), x, material, content, yaw, tilt, error, clearance,
                     .7 + .6*u[:, 7:8], .16*(u[:, 8:9]-.5), .04*u[:, 9:10], 10.+50.*u[:, 10:11],
                     (5+21*u[:, 11]).long(), (5+21*u[:, 12]).long(), .8+.4*u[:, 13:14], .001*u[:, 14:15])
        if corners is not None:
            def lookup(data):
                return self.tensor(data)[corners].reshape(count, -1)
            x = PayloadParamsX(**anchor)
            x = replace(x, total_mass=lookup([.04,.20,1.5,.20,.50,1.10,.9,.65]),
                        fill_ratio=lookup([.6,.90,.6,.6,.60,.75,.30,.90]),
                        mass_weights=torch.where(((corners == 2) | (corners == 5))[:, None],
                                                 self.tensor([.08,.42,.08,.42]), x.mass_weights))
            p = dict(nominal)
            mu = lookup([[.26,.26],[.63,.55],[.63,.4095],[.66,.70],[.63,.55],[.40,.26],[.30,.26],[.63,.55]])
            p.update(mu_eff=mu, beta=torch.where(corners[:, None] == 0, .030, p["beta"]), k=torch.where(corners[:, None] == 3, 550., p["k"]),
                     alpha=torch.where(corners[:, None] == 3, .24, p["alpha"]))
            hw.x = x
            hw.yaw, hw.tilt, hw.height_error = (lookup(a) for a in (
                [0,0,0,0,0,.22,0,0],[0,0,0,0,0,.03,0,0],[0,0,0,0,.001,-.003,0,0]))
            hw.clearance = lookup([[.26,.26]]*5+[[.055,.012],[.10,.10],[.006,.006]])
            worst = ((corners == 4) | (corners == 5))[:, None]
            hw.force_gain = torch.where(worst, .70, 1.)
            hw.bias, hw.noise = worst.float()*.06, worst.float()*.025
            hw.cutoff = torch.where(worst, 10., 35.)
            hw.sensor_delay = hw.control_delay = torch.where(worst[:, 0], 25, 5)
            hw.current_gain, hw.deadband = torch.where(worst, .8, 1.), worst.float()*.001
        cap = x.fill_slack_capacity()
        p["S_bag"] = torch.minimum(p["S_bag"], cap)
        p["L0"] = nominal["L0"] * (x.fill_ratio/.75).pow(-1/3)
        hw.proxy = ProxyParams(**p)
        return hw


class HardwareGapFilter:
    """Per-world command and sensor rings at 500 Hz, including independent delays."""
    def __init__(self, episode, command, enabled=True):
        self.p, self.enabled = episode, enabled
        self.command = command.repeat(26, 1, 1)
        self.signal = command.new_zeros(26, len(command), 6)
        self.filtered = torch.zeros_like(self.signal[0])
        self.cursor = torch.zeros(1, dtype=torch.long, device=command.device)
        self.rows = torch.arange(len(command), device=command.device)
        self.coeff = 1 - torch.exp(-2*math.pi*episode.cutoff*DT)
        self.wrench_scale = command.new_tensor([1.,1.,1.,.02,.02,.02])

    def reset(self, mask, command):
        """Clear bool [B] rows with float32 [B,C] initial commands (m and N)."""
        self.command.copy_(torch.where(mask[None,:,None], command[None], self.command))
        self.signal.masked_fill_(mask[None,:,None], 0.)
        self.filtered.masked_fill_(mask[:, None], 0.)

    def actuate(self, target):
        """Delay [B,C] float32 Cartesian/jaw command (m, N), return the realized command."""
        if not self.enabled:
            return target
        self.command.index_copy_(0, self.cursor, target.unsqueeze(0))
        return self.command[(self.cursor-self.p.control_delay)%26, self.rows]

    def observe(self, wrench):
        """Input/output float32 [B,6] force (N) and torque (N m), one physics tick."""
        if not self.enabled:
            return wrench
        scale = self.wrench_scale
        noisy = self.p.force_gain*wrench + scale*(self.p.bias+self.p.noise*torch.randn_like(wrench))
        self.filtered.lerp_(noisy, self.coeff)
        self.signal.index_copy_(0, self.cursor, self.filtered.unsqueeze(0))
        received = self.signal[(self.cursor-self.p.sensor_delay)%26, self.rows]
        self.cursor.add_(1).remainder_(26)
        return received

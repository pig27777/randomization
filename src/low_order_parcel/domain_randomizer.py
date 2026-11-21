"""Stage 3 episode distributions and fixed evaluation corners in SI units."""

from dataclasses import dataclass, fields, replace
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from .dual_arm_parcel_sim import DualArmParcelSim
from .hardware_filters import YAMParameters
from .lumped_contact_proxy import LumpedContactProxy, ProxyParams
from .pad_torsion import PadTorsionParams
from .unified_payload import DT, PayloadParamsX, UnifiedPayloadTemplate
from .virtual_rgb import RGBParameters

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _interval(bounds, rank):
    low, high = bounds
    return low + (high - low) * rank


@dataclass(frozen=True)
class Stage3Episode:
    proxy: ProxyParams
    payload: PayloadParamsX
    F_half: Tensor
    T_tear: Tensor
    material: Tensor
    content: Tensor
    yaw_rad: Tensor
    clearance_m: Tensor  # [B,2], free travel from the two longitudinal sides
    swing_rad: Tensor
    fold_rad: Tensor
    yam: YAMParameters
    rgb: RGBParameters
    torsion: PadTorsionParams = PadTorsionParams()

    def make_sim(self, *, mount_xy=None):
        return DualArmParcelSim(LumpedContactProxy(self.proxy, DT), LumpedContactProxy(self.proxy, DT),
                                UnifiedPayloadTemplate(self.payload), F_half=self.F_half, T_tear=self.T_tear,
                                mount_xy=mount_xy, torsion=self.torsion)

    def table_state(self, sim):
        state = sim.initial_state()
        x, y, z = state.payload.pos.unbind(dim=-1)
        c, s = self.yaw_rad.cos(), self.yaw_rad.sin()
        pos = torch.stack((c * x - s * y, s * x + c * y, z), dim=-1)
        payload = state.payload._replace(pos=pos)
        return state._replace(payload=payload, side_origin=sim.payload.com(payload)[:, None].expand(-1, 2, -1))


class DomainRandomizer:
    def __init__(self, device="cpu", seed=None, config_path=CONFIG_DIR / "stage3_dr_and_corners.yaml"):
        self.device = torch.device(device)
        self.config = json.loads(Path(config_path).read_text())
        cfg = self.config
        self.rng = torch.Generator(device=self.device).manual_seed(cfg["seed"] if seed is None else seed)
        self.materials, self.contents = tuple(cfg["material_order"]), tuple(cfg["content_order"])
        materials = json.loads((CONFIG_DIR / "proxy_material_envelopes.yaml").read_text())["materials"]
        rows = [materials[name] for name in self.materials]
        self.nominal_rows = [row["nominal"]["proxy"] for row in rows]
        self.nominal = {f.name: self.tensor([row[f.name] for row in self.nominal_rows]).reshape(len(rows), -1)
                        for f in fields(ProxyParams)}
        self.order = tuple(rows[0]["dependency"]["order"])
        self.correlation = self.tensor([row["dependency"]["correlation"] for row in rows])
        self.chol = torch.linalg.cholesky(self.correlation)
        self.material_bounds = self.tensor([[row["tier1_ranges"][key] for key in self.order] for row in rows])
        self.payload_config = json.loads((CONFIG_DIR / "payload_template_X.yaml").read_text())
        self.presets = PayloadParamsX.from_presets(self.payload_config, self.contents, device=self.device)
        self.content_weights = self.tensor(cfg["content_probability"])
        self.stiffness_bounds = self.tensor(cfg["whole_bag_k_N_m"])
        ranges = cfg["whole_bag_k_N_m"]
        self.k_range = min(row[0] for row in ranges), max(row[1] for row in ranges)
        self.log_k_range = tuple(map(math.log, self.k_range))
        self.stiffness_column = self.order.index("k")

    def tensor(self, values, *, dtype=torch.float32):
        return torch.as_tensor(values, device=self.device, dtype=dtype)

    def _uniform(self, shape):
        return torch.rand(shape, device=self.device, dtype=torch.float32, generator=self.rng)

    def sample(self, count, *, curriculum_alpha=1.):
        if count < 1 or not 0. <= curriculum_alpha <= 1.:
            raise ValueError("count must be positive and curriculum_alpha in [0,1]")
        cfg, a = self.config, curriculum_alpha
        mix = cfg["narrow_mix"] + (1. - cfg["narrow_mix"]) * a
        k_bag = _interval(self.log_k_range, self._uniform((count, 1))).exp().clamp(*self.k_range)
        bounds_low, bounds_high = self.stiffness_bounds.unbind(dim=1)
        eligible = (k_bag >= bounds_low) & (k_bag <= bounds_high)
        material = torch.multinomial(eligible.float(), 1, generator=self.rng).squeeze(1)
        content = torch.multinomial(self.content_weights, count, replacement=True, generator=self.rng)
        nominal = {key: value[material] for key, value in self.nominal.items()}
        z = torch.randn((count, len(self.order), 1), device=self.device, generator=self.rng)
        normal = (self.chol[material] @ z).squeeze(-1)
        k_lo, k_hi = self.stiffness_bounds[material].log().split(1, dim=-1)
        k_rank = ((k_bag.log() - k_lo) / (k_hi - k_lo)).clamp(1e-6, 1. - 1e-6)
        # Condition the material copula on the already drawn log stiffness.
        column = self.stiffness_column
        delta = torch.special.ndtri(k_rank) - normal[:, column:column + 1]
        rank = torch.special.ndtr(normal + self.correlation[material, :, column] * delta)
        lo, hi = self.material_bounds[material].unbind(dim=-1)
        values = dict(zip(self.order, torch.lerp(lo, hi, rank).split(1, dim=-1), strict=True))
        quantile = dict(zip(self.order, rank.split(1, dim=-1), strict=True))
        # Two parallel ropes carry the specified whole-bag equivalent stiffness.
        stiffness = .5 * k_bag
        mu_lo, mu_hi = cfg["friction_left"]
        mu_left = mu_lo * (mu_hi / mu_lo) ** quantile["mu_L"]
        ratio = _interval(cfg["right_friction_ratio"], self._uniform((count, 1)))
        ratio = torch.lerp(nominal["mu_eff"][:, 1:] / nominal["mu_eff"][:, :1], ratio, mix)
        wide = {k: values[k] for k in ("S_bag", "beta", "s_crit", "F_min", "h_min")}
        wide["alpha"] = _interval(cfg["gather_alpha"], quantile["alpha"])
        wide["k"] = stiffness
        p = nominal | {key: torch.lerp(nominal[key], value, mix) for key, value in wide.items()}
        mu = torch.lerp(nominal["mu_eff"][:, :1], mu_left, mix)
        p["mu_eff"] = torch.cat((mu, mu * ratio), dim=1)
        p["F_lock"] = torch.maximum(p["F_lock"], p["F_min"] + .05)
        p["F_release"] = torch.minimum(p["F_release"], .8 * p["F_lock"])
        p["s_max"] = torch.maximum(p["s_max"], p["s_crit"])
        p["c_d"] = nominal["c_d"] * (p["k"] / nominal["k"]).sqrt()

        raw = PayloadParamsX.sample(self.payload_config, count, device=self.device, generator=self.rng)
        base = {f.name: getattr(self.presets, f.name)[content] for f in fields(raw)}
        x = {name: torch.lerp(value, getattr(raw, name), .55 * mix) for name, value in base.items()}
        for name in ("bag_half_extents", "fill_ratio", "total_mass", "table_friction"):
            x[name] = torch.lerp(base[name], getattr(raw, name), mix)
        x["mass_weights"] = torch.lerp(base["mass_weights"], raw.mass_weights, .7 * mix)
        x["max_indent_ratio"] = torch.where(content[:, None] == 0, 0., x["max_indent_ratio"])
        for name in ("alpha_bend", "theta_max", "alpha_stretch", "slide_slack_max"):
            x[name] = torch.where(content[:, None] == 0, 0., x[name])
        mass_lo, mass_hi = cfg["mass_kg"]
        x["total_mass"] = torch.lerp(base["total_mass"],
                                    mass_lo * (mass_hi / mass_lo) ** self._uniform((count, 1)), mix)
        for name in ("fill_ratio", "table_friction"):
            x[name] = torch.lerp(base[name], _interval(cfg[name], self._uniform((count, 1))), mix)
        # Loose contents have a larger fraction of their cavity available for internal travel.
        base_cavity = PayloadParamsX.cavity_slack(base["bag_half_extents"], base["fill_ratio"])
        raw_cavity = raw.cavity_slack(raw.bag_half_extents, raw.fill_ratio)
        fraction = torch.lerp((base["slide_slack_max"] / base_cavity.clamp_min(1e-8)).clamp_max(1.),
                              raw.slide_slack_max / raw_cavity.clamp_min(1e-8), .55 * mix)
        x["slide_slack_max"] = torch.where(content[:, None] == 0, 0., fraction) * PayloadParamsX.cavity_slack(
            x["bag_half_extents"], x["fill_ratio"])
        payload = PayloadParamsX(**x)
        p["S_bag"] = torch.minimum(p["S_bag"], payload.fill_slack_capacity())
        p["L0"] *= (payload.fill_ratio / .75).pow(-1. / 3.)

        draw = self._uniform((count, 7))
        yaw = mix * _interval(cfg["yaw_rad"], draw[:, :1])
        clearance = _interval(cfg["clearance_m"], draw[:, 1:3])
        blocked = draw[:, 3:4] < a * cfg["occlusion_probability"]
        side = draw[:, 4:5]
        obstruction = blocked & torch.cat((side < 2. / 3., side > 1. / 3.), dim=1)
        clearance = torch.where(obstruction, _interval(cfg["blocked_clearance_m"], draw[:, 5:7]), clearance)
        F_half = torch.lerp(torch.full_like(yaw, cfg["nominal_press_half_N"]),
                            _interval(cfg["press_half_N"], self._uniform((count, 1))), mix)
        tear = torch.lerp(torch.full_like(yaw, 40.), _interval(cfg["tear_N"], self._uniform((count, 1))), mix)
        yam, rgb = self._hardware(payload.total_mass, a)
        draw = self._uniform((count, 3)).split(1, dim=1)
        nominal = PadTorsionParams()
        torsion = PadTorsionParams(*(torch.lerp(torch.full_like(rank, center),
                                               _interval(cfg.get(key, (center, center)), rank), mix)
                                    for key, rank, center in zip(
                                        ("pad_radius_m", "neck_damping_ratio", "neck_length_m"), draw, nominal[:3],
                                        strict=True)))
        return Stage3Episode(ProxyParams(**p), payload, F_half, tear, material, content, yaw, clearance,
                             torch.zeros_like(yaw), torch.zeros_like(yaw), yam, rgb, torsion)

    def _hardware(self, like, alpha):
        yam = YAMParameters.sample(like, generator=self.rng)
        rgb = RGBParameters.sample(like, generator=self.rng)
        one = torch.ones_like(yam.sensor_gain)
        # Scale the low-pass time constant with the other degradation magnitudes.
        cutoff = yam.cutoff_hz / alpha if alpha > 0. else torch.full_like(yam.cutoff_hz, 2500.)
        clean = YAMParameters(one, 0. * one, 0. * one, cutoff, 0. * one,
                              torch.zeros_like(yam.delay_steps), 0. * one, one, 0. * one)
        yam = yam._replace(cutoff_hz=cutoff)
        yam = YAMParameters(*(torch.lerp(x, y, alpha) if x.is_floating_point()
                              else (alpha * y).round().long() for x, y in zip(clean, yam, strict=True)))
        rgb = rgb._replace(translation_m=alpha * rgb.translation_m, rotation_rad=alpha * rgb.rotation_rad,
                           mask_jitter=alpha * rgb.mask_jitter, drop_probability=alpha * rgb.drop_probability,
                           delay_steps=(alpha * rgb.delay_steps).round().long(),
                           wrist_min_range_m=torch.lerp(torch.full_like(rgb.wrist_min_range_m, .15),
                                                       rgb.wrist_min_range_m, alpha),
                           wrist_fov_rad=torch.lerp(torch.full_like(rgb.wrist_fov_rad, 1.30899694),
                                                  rgb.wrist_fov_rad, alpha))
        return yam, rgb

    def resample(self, episode, reset_mask, *, curriculum_alpha=1.):
        """Return an episode batch with only the bool [B] reset rows resampled."""
        if (reset_mask.shape != episode.material.shape or reset_mask.dtype != torch.bool
                or reset_mask.device != episode.material.device):
            raise ValueError("reset_mask must be a bool [B] tensor on the episode device")
        fresh = self.sample(len(reset_mask), curriculum_alpha=curriculum_alpha)

        def select(old, new):
            mask = reset_mask.reshape(-1, *([1] * (old.ndim - 1)))
            return torch.where(mask, new, old)

        def replace_fields(old, new):
            return type(old)(*(select(getattr(old, f.name), getattr(new, f.name)) for f in fields(old)))

        return replace(episode, proxy=replace_fields(episode.proxy, fresh.proxy),
                       payload=replace_fields(episode.payload, fresh.payload),
                       yam=YAMParameters(*(select(x, y) for x, y in zip(episode.yam, fresh.yam, strict=True))),
                       rgb=RGBParameters(*(select(x, y) for x, y in zip(episode.rgb, fresh.rgb, strict=True))),
                       torsion=PadTorsionParams(*(select(torch.as_tensor(x, device=self.device).expand_as(y), y)
                                                 for x, y in zip(episode.torsion[:3], fresh.torsion[:3], strict=True)),
                                                episode.torsion.stiffness_Nm_rad),
                       **{f.name: select(getattr(episode, f.name), getattr(fresh, f.name)) for f in fields(episode)
                          if isinstance(getattr(episode, f.name), Tensor)})


class CornerSuite:
    """Ten deterministic physical and sensor parameter rows, in C1–C10 order."""

    def __init__(self, randomizer):
        dr = self.dr = randomizer
        cfg = dr.config
        self.names = tuple(f"C{i}" for i in range(1, 11))
        corners = [cfg["corners"][name] for name in self.names]
        material = [dr.materials.index(c["material"]) for c in corners]
        content = [dr.contents.index(c["content"]) for c in corners]
        p = ProxyParams(*(dr.tensor([c.get("proxy", {}).get(f.name, dr.nominal_rows[i][f.name])
                                    for i, c in zip(material, corners, strict=True)]).reshape(10, -1)
                          for f in fields(ProxyParams)))
        payload_rows = [dr.payload_config["defaults"] | dr.payload_config["presets"][c["content"]]
                        | c.get("payload", {}) for c in corners]
        x = PayloadParamsX(*(dr.tensor([row[f.name] for row in payload_rows]).reshape(10, -1)
                            for f in fields(PayloadParamsX)))
        cavity = x.cavity_slack(x.bag_half_extents, x.fill_ratio)
        x = replace(x, slide_slack_max=torch.minimum(x.slide_slack_max, cavity))
        p = replace(p, S_bag=torch.minimum(p.S_bag, x.fill_slack_capacity()),
                    L0=p.L0 * (x.fill_ratio / .75).pow(-1. / 3.))
        yam_defaults = dict(zip(YAMParameters._fields, (1., 0., 0., 15., 0., 0, 0., 1., 0.), strict=True))
        rgb_defaults = dict(zip(RGBParameters._fields, ([0.] * 3, [0.] * 3, 0., 0., 0, .15, 1.30899694),
                                strict=True))
        yam_rows = [yam_defaults | (cfg["worst_yam"] if c.get("worst_hardware") else {}) for c in corners]
        rgb_rows = [rgb_defaults | (cfg["worst_rgb"] if c.get("worst_hardware") else {}) | c.get("rgb", {})
                    for c in corners]

        def channel(rows, key, shared=()):
            dtype = torch.long if key == "delay_steps" else torch.float32
            values = dr.tensor([row[key] for row in rows], dtype=dtype).reshape(10, -1)
            return values if key in shared else values[:, None].expand(-1, 2, -1).clone()

        yam = YAMParameters(*(channel(yam_rows, key) for key in YAMParameters._fields))
        rgb = RGBParameters(*(channel(rgb_rows, key, ("wrist_min_range_m", "wrist_fov_rad"))
                              for key in RGBParameters._fields))
        scalar = lambda name, default: dr.tensor([[c.get(name, default)] for c in corners])
        self.episodes = Stage3Episode(p, x, scalar("F_half", .30), scalar("T_tear", 40.),
                                      dr.tensor(material, dtype=torch.long), dr.tensor(content, dtype=torch.long),
                                      scalar("yaw_rad", 0.), dr.tensor([c.get("clearance_m", [.26, .26]) for c in corners]),
                                      scalar("swing_rad", 0.), scalar("fold_rad", 0.), yam, rgb)

    def select(self, indices):
        """Return rows for zero-based int64 [B] corner indices, preserving their order."""
        e = self.episodes
        return Stage3Episode(ProxyParams(*(getattr(e.proxy, f.name)[indices] for f in fields(e.proxy))),
                             PayloadParamsX(*(getattr(e.payload, f.name)[indices] for f in fields(e.payload))),
                             *(getattr(e, f.name)[indices] for f in fields(e)[2:-3]),
                             YAMParameters(*(value[indices] for value in e.yam)),
                             RGBParameters(*(value[indices] for value in e.rgb)), e.torsion)

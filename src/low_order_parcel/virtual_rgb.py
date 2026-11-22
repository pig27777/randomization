"""Zero-render RGB observations from the four parcel blocks (SI)."""

import math
from typing import NamedTuple

import torch
from torch import Tensor

from .lift_probe import monocular_lift_ratio
from .unified_payload import PayloadState, UnifiedPayloadTemplate


class CameraPose(NamedTuple):
    position: Tensor           # [B,3], camera center in world coordinates
    world_from_camera: Tensor  # [B,3,3], columns point right, down, forward


class RGBParameters(NamedTuple):
    translation_m: Tensor      # [B,top/wrist,3], world translation error
    rotation_rad: Tensor       # [B,top/wrist,3], local XYZ angle error
    mask_jitter: Tensor        # [B,top/wrist,1], fractional mask jitter
    drop_probability: Tensor   # [B,top/wrist,1]
    delay_steps: Tensor        # [B,top/wrist,1], at 2 ms
    wrist_min_range_m: Tensor  # [B,1]
    wrist_fov_rad: Tensor      # [B,1], horizontal

    @classmethod
    def sample(cls, like: Tensor, *, generator: torch.Generator | None = None):
        def uniform(shape, lo, hi):
            draw = torch.rand((len(like), *shape), dtype=like.dtype, device=like.device, generator=generator)
            return lo + (hi - lo) * draw

        angle = math.radians(1.5)
        delay = torch.randint(30, 61, (len(like), 2, 1), device=like.device, generator=generator)
        return cls(uniform((2, 3), -.005, .005), uniform((2, 3), -angle, angle),
                   uniform((2, 1), 0., .12), uniform((2, 1), .05, .15), delay,
                   uniform((1,), .10, .20), uniform((1,), math.radians(65), math.radians(85)))


class RGBObservation(NamedTuple):
    top_box: Tensor          # [B,6], x, y, L, W, cos(yaw), sin(yaw); yaw modulo pi
    top_clearance: Tensor    # [B,4], -X,+X,-Y,+Y, in m
    vis_lift_ratio: Tensor   # [B,1], synchronous image displacement during the active lift window
    top_valid: Tensor        # [B,1], a visible Top frame has arrived
    wrist_target_rel: Tensor  # [B,3], in the Wrist camera frame
    wrist_profile: Tensor     # [B,2], height and width in m at the FK depth
    wrist_sway_est: Tensor    # [B,2], apparent inter-frame displacement, camera right/down
    wrist_vis_valid: Tensor   # [B,1], latest delivered frame geometrically valid
    vis_lift_valid: Tensor    # [B,1], same delivered Top frame has a usable lift ratio


def _rotation_xyz(angles: Tensor) -> Tensor:
    x, y, z = angles.cos().unbind(dim=-1)
    sx, sy, sz = angles.sin().unbind(dim=-1)
    return torch.stack((z * y, z * sy * sx - sz * x, z * sy * x + sz * sx,
                        sz * y, sz * sy * sx + z * x, sz * sy * x - z * sx,
                        -sy, y * sx, y * x), dim=-1).unflatten(-1, (3, 3))


class VirtualRGBProjector:
    def __init__(self, payload: UnifiedPayloadTemplate, top_camera: CameraPose,
                 params: RGBParameters, *, table_bounds: Tensor, table_z: float = 0.,
                 top_horizontal_fov: float = math.radians(75.), aspect_ratio: float = 4. / 3.,
                 frame_period_steps: int = 10, generator: torch.Generator | None = None):
        """Call step at 500 Hz. By default both cameras expose at 50 Hz."""
        like = payload.params.total_mass
        batch = len(like)
        if like.dtype != torch.float32:
            raise ValueError("Use a float32 payload with the virtual RGB cameras")
        for value, shape in zip(top_camera, ((batch, 3), (batch, 3, 3)), strict=True):
            if (value.shape != shape or value.device != like.device or value.dtype != like.dtype
                    or not value.isfinite().all()):
                raise ValueError("The Top camera pose must match the payload batch and device")
        shapes = ((batch, 2, 3), (batch, 2, 3), *((batch, 2, 1),) * 3, *((batch, 1),) * 2)
        for name, value, shape in zip(params._fields, params, shapes):
            dtype = torch.long if name == "delay_steps" else like.dtype
            if (value.shape != shape or value.dtype != dtype or value.device != like.device
                    or not value.isfinite().all()):
                raise ValueError(f"Invalid RGB parameter {name}; expected finite {shape} and {dtype}")
        if ((params.delay_steps < 0) | (params.delay_steps > 60) |
                (params.drop_probability < 0.) | (params.drop_probability > 1.) |
                (params.mask_jitter < 0.) | (params.mask_jitter >= 1.)).any():
            raise ValueError("Invalid RGB delay, drop probability or mask jitter")
        if ((params.wrist_min_range_m < 0.) | (params.wrist_fov_rad <= 0.) |
                (params.wrist_fov_rad >= math.pi)).any():
            raise ValueError("Invalid Wrist camera range or field of view")
        if (not 0. < top_horizontal_fov < math.pi or not math.isfinite(aspect_ratio) or aspect_ratio <= 0.
                or frame_period_steps < 1 or not math.isfinite(table_z)):
            raise ValueError("Invalid camera field of view, aspect ratio, exposure period or table height")
        if (table_bounds.shape != (batch, 4) or table_bounds.device != like.device
                or table_bounds.dtype != like.dtype or not table_bounds.isfinite().all()
                or (table_bounds[:, ::2] >= table_bounds[:, 1::2]).any()):
            raise ValueError("table_bounds must contain finite [xmin,xmax,ymin,ymax] for each environment")
        self.payload, self.params, self.top_camera = payload, params, top_camera
        self.table_bounds, self.table_z = table_bounds, table_z
        self.frame_period_steps, self.aspect, self.generator = frame_period_steps, aspect_ratio, generator
        self.top_limit = math.tan(top_horizontal_fov / 2.)
        self.wrist_limit = (params.wrist_fov_rad / 2.).tan()
        self._rotation = _rotation_xyz(params.rotation_rad)
        self._signs = like.new_tensor([(x, y, z) for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
        self._ring = like.new_zeros((61, batch, 2, 13))
        self._exposed = torch.zeros((61, batch, 2), dtype=torch.bool, device=like.device)
        self._last = like.new_zeros((batch, 2, 13))
        self._age = torch.zeros((batch, 1), dtype=torch.long, device=like.device)
        self._batch = torch.arange(batch, device=like.device)[:, None]
        self._camera = torch.arange(2, device=like.device)
        self._previous_wrist = like.new_zeros((batch, 3))
        self._wrist_seen = torch.zeros_like(self._age, dtype=torch.bool)
        self._lift_origin = like.new_zeros((batch, 3))  # Top mask y, FK image y, FK world z
        self._reference_box = like.new_zeros((batch, 3))
        self._gamma = torch.zeros_like(like)
        self._lifting = torch.zeros_like(self._age, dtype=torch.bool)
        self._index = self._frame_clock = 0

    def reset(self, reset_mask: Tensor, *, params: RGBParameters | None = None) -> None:
        """Clear selected histories and optionally install new episode draws."""
        mask = reset_mask[:, None]
        if params is not None:
            self.params = RGBParameters(*(torch.where(mask if old.ndim == 2 else mask.unsqueeze(-1), new, old)
                                          for new, old in zip(params, self.params, strict=True)))
            self._rotation = _rotation_xyz(self.params.rotation_rad)
            self.wrist_limit = (self.params.wrist_fov_rad / 2.).tan()
        self._age = torch.where(mask, 0, self._age)
        self._last = torch.where(mask.unsqueeze(-1), 0., self._last)
        self._wrist_seen = self._wrist_seen & ~mask
        self._lifting = self._lifting & ~mask
        self._gamma = torch.where(mask, 0., self._gamma)

    def vertices(self, state: PayloadState) -> Tensor:
        """World corners [B,32,3] from four locally compressed block envelopes."""
        xy = (.5 * self.payload.params.bag_half_extents[:, None, :2]).expand(-1, 4, -1)
        half = torch.cat((xy, self.payload.half_thickness(state.d_surf).unsqueeze(-1)), dim=-1)
        offset = half.unsqueeze(2) * self._signs
        frame = self.payload.frame(state)
        return (state.pos.unsqueeze(2) + offset @ frame[:, None].transpose(-1, -2)).flatten(1, 2)

    @staticmethod
    def _project(vertices, position, orientation):
        local = (vertices - position.unsqueeze(-2)) @ orientation
        return local[..., :2] / local[..., 2:].clamp_min(1e-6), local[..., 2]

    def _top_features(self, pixels, lo, hi):
        """Use the table homography and a 2-D principal axis of the projected mask."""
        camera = self.top_camera
        points = torch.cat((pixels.clamp(lo[:, None], hi[:, None]), .5 * (lo + hi)[:, None]), dim=1)
        ray = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
        ray = ray @ camera.world_from_camera.transpose(1, 2)
        ray_z = ray[..., 2:3]
        divisor = torch.where(ray_z.abs() >= 1e-6, ray_z, -1e-6)
        plane = camera.position[:, None, :2] + (self.table_z - camera.position[:, None, 2:]) * ray[..., :2] / divisor
        outline, center = plane[:, :-1], plane[:, -1]
        delta = outline - outline.mean(dim=1, keepdim=True)
        dx, dy = delta.unbind(dim=-1)
        yaw = .5 * torch.atan2(2. * (dx * dy).sum(dim=1), (dx.square() - dy.square()).sum(dim=1))
        c, s = yaw.cos(), yaw.sin()
        axes = torch.stack((c, -s, s, c), dim=1).unflatten(-1, (2, 2))
        aligned = delta @ axes
        dimensions = aligned.amax(dim=1) - aligned.amin(dim=1)
        box = torch.cat((center, dimensions, c[:, None], s[:, None]), dim=1)
        lower, upper = outline.amin(dim=1), outline.amax(dim=1)
        clearance = torch.stack((lower[:, 0] - self.table_bounds[:, 0],
                                 self.table_bounds[:, 1] - upper[:, 0],
                                 lower[:, 1] - self.table_bounds[:, 2],
                                 self.table_bounds[:, 3] - upper[:, 1]), dim=1).clamp_min(0.)
        valid = ((self.table_z - camera.position[:, None, 2:]) * ray_z > 0.).all(dim=1)
        return box, clearance, valid

    def step(self, state: PayloadState, wrist_camera: CameraPose, arm_A_fk: Tensor,
             lift_active: Tensor, *, exterior_height: Tensor | None = None,
             lift_travel_m: float | Tensor | None = None) -> RGBObservation:
        """Project one sample. Without height calibration, the default lift window is 25 mm."""
        p = self.params
        vertices = self.vertices(state)
        positions = torch.stack((self.top_camera.position, wrist_camera.position), dim=1)
        orientations = torch.stack((self.top_camera.world_from_camera, wrist_camera.world_from_camera), dim=1)
        pixels, depth = self._project(vertices[:, None], positions + p.translation_m, orientations @ self._rotation)
        lo, hi = pixels.amin(dim=-2), pixels.amax(dim=-2)
        noise = torch.rand((*lo.shape, 2), dtype=lo.dtype, device=lo.device, generator=self.generator)
        stretch, shift = ((2. * noise - 1.) * p.mask_jitter.unsqueeze(-1)).unbind(dim=-1)
        center = .5 * (lo + hi + (hi - lo) * shift)
        half = .5 * (hi - lo) * (1. + stretch)
        pixels = center.unsqueeze(-2) + (pixels - .5 * (lo + hi).unsqueeze(-2)) * (1. + stretch.unsqueeze(-2))
        limit_x = torch.cat((torch.full_like(self.wrist_limit, self.top_limit), self.wrist_limit), dim=1)
        limit = torch.stack((limit_x, limit_x / self.aspect), dim=-1)
        lo, hi = (center - half).clamp(-limit, limit), (center + half).clamp(-limit, limit)
        exposure = self._frame_clock == 0
        self._frame_clock = (self._frame_clock + 1) % self.frame_period_steps
        drops = torch.rand(p.drop_probability.shape, dtype=lo.dtype, device=lo.device, generator=self.generator)
        fresh = (drops >= p.drop_probability).squeeze(-1) & exposure
        in_image = (depth > 1e-6).all(dim=-1) & (hi > lo).all(dim=-1)

        top_box, top_clearance, table_visible = self._top_features(pixels[:, 0], lo[:, 0], hi[:, 0])
        top_valid = in_image[:, :1] & table_visible
        fk_image, _ = self._project(arm_A_fk[:, None], *self.top_camera)
        mask_y = .5 * (lo[:, 0, 1:2] + hi[:, 0, 1:2])
        lift_active = lift_active.reshape(-1, 1)
        baseline_needed = lift_active & ~self._lifting
        top_exposed = top_valid & fresh[:, :1]
        start = baseline_needed & top_exposed
        working = torch.cat((mask_y, fk_image[:, 0, 1:2], arm_A_fk[:, 2:3]), dim=1)
        self._lift_origin = torch.where(start, working, self._lift_origin)
        self._lifting = lift_active & (self._lifting | start)
        delta_y, ee_y, lift_z = (working - self._lift_origin).split(1, dim=1)
        ratio = (delta_y * ee_y.sign() / ee_y.abs().clamp_min(1e-6)).clamp(0., 1.5)
        in_window = (lift_z >= 0.) & (ee_y.abs() > 1e-6)
        if exterior_height is not None:
            silhouette = torch.cat((.5*(lo[:,0] + hi[:,0]), hi[:,0,:1]-lo[:,0,:1]), dim=1)
            self._reference_box = torch.where(start, silhouette, self._reference_box)
            ratio, in_window = monocular_lift_ratio(
                self._reference_box, silhouette, lift_z, exterior_height,
                self.top_camera.position[:,2:3], self.top_camera.world_from_camera[:,2,:])
        elif lift_travel_m is None:
            lift_travel_m = .025
        if isinstance(lift_travel_m, Tensor):
            lift_travel_m = lift_travel_m.reshape(-1, 1)
        if lift_travel_m is not None:
            in_window = in_window & (lift_z <= lift_travel_m + 1e-6)
        update_ratio = self._lifting & in_window & top_exposed
        self._gamma = torch.where(start | ~lift_active, 0., torch.where(update_ratio, ratio, self._gamma))

        wrist_pos, wrist_frame = wrist_camera
        fk_depth = ((arm_A_fk - wrist_pos)[:, None] @ wrist_frame)[..., 0, 2:3]
        wrist_dist = (vertices.mean(dim=1) - wrist_pos).norm(dim=-1, keepdim=True)
        wrist_valid = in_image[:, 1:] & (wrist_dist > p.wrist_min_range_m) & (fk_depth > 1e-6)
        rel = torch.cat((.5 * (lo[:, 1] + hi[:, 1]) * fk_depth, fk_depth), dim=1)
        profile = (hi[:, 1] - lo[:, 1]).flip(dims=(1,)) * fk_depth
        world_estimate = wrist_pos + (wrist_frame @ rel.unsqueeze(-1)).squeeze(-1)
        sway = ((world_estimate - self._previous_wrist)[:, None] @ wrist_frame)[:, 0, :2]
        sway = torch.where(self._wrist_seen, sway, 0.)
        wrist_exposed = wrist_valid & fresh[:, 1:]
        self._previous_wrist = torch.where(wrist_exposed, world_estimate, self._previous_wrist)
        self._wrist_seen = torch.where(fresh[:, 1:], wrist_valid, self._wrist_seen)

        top_frame = torch.cat((torch.where(top_valid, top_box, 0.), torch.where(top_valid, top_clearance, 0.),
                               self._gamma, update_ratio.to(lo.dtype), top_valid.to(lo.dtype)), dim=1)
        wrist_frame_data = torch.cat((rel, profile, sway, torch.zeros_like(top_clearance), torch.zeros_like(self._gamma), wrist_valid.to(lo.dtype)), dim=1)
        self._ring[self._index] = torch.stack((top_frame, wrist_frame_data), dim=1)
        self._exposed[self._index] = fresh
        index = (self._index - p.delay_steps.squeeze(-1)) % len(self._ring)
        arrived = self._exposed[index, self._batch, self._camera] & (self._age >= p.delay_steps.squeeze(-1))
        delayed = self._ring[index, self._batch, self._camera]
        # A blinded Wrist frame updates validity while retaining the last seen target.
        keep_wrist = delayed[:, 1, -1:] == 0.
        wrist_data = torch.where(keep_wrist, self._last[:, 1, :-1], delayed[:, 1, :-1])
        delayed = torch.stack((delayed[:, 0], torch.cat((wrist_data, delayed[:, 1, -1:]), dim=1)), dim=1)
        self._last = torch.where(arrived.unsqueeze(-1), delayed, self._last)
        self._index = (self._index + 1) % len(self._ring)
        self._age = (self._age + 1).clamp_max(len(self._ring))
        box, clearance, gamma, lift_valid, top_valid = self._last[:, 0].split((6, 4, 1, 1, 1), dim=1)
        rel, profile, sway, _, wrist_valid = self._last[:, 1].split((3, 2, 2, 5, 1), dim=1)
        return RGBObservation(box, clearance, gamma, top_valid.bool(), rel, profile, sway, wrist_valid.bool(), lift_valid.bool())

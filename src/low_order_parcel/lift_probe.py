"""Phase B settings in SI units; height is the observed exterior height."""
from math import sqrt

PROBE_RAMP_S = 2.25
PROBE_SETTLE_S = .42
CAMERA_OFFSET = (0., -.60, .65)
# Camera-to-world columns: right, down, forward.
CAMERA_ROTATION = ((1., 0., 0.), (0., -sqrt(.5), sqrt(.5)), (0., -sqrt(.5), -sqrt(.5)))


def probe_height(height):
    return (.90 * height).clamp(.040, .100)


def monocular_lift_ratio(reference, current, fk_rise, height, camera_z, vertical):
    """Estimate vertical co-motion from calibrated 2-D RGB boxes [cx,cy,width].

    Centers are normalized pinhole coordinates. Apparent width supplies relative
    depth under a constant projected physical width assumption.
    """
    x0, y0, w0 = reference.split(1, dim=-1)
    x, y, w = current.split(1, dim=-1)
    vx, vy, vz = vertical.split(1, dim=-1)
    ray0 = vx*x0 + vy*y0 + vz
    ray = vx*x + vy*y + vz
    valid = (w > 1e-5) & (w0 > 1e-5) & (ray0 < -1e-4) & (fk_rise >= .002) & (height > 0)
    depth0 = (.5*height - camera_z) / ray0.clamp(max=-1e-4)
    rise = depth0 * (w0 / w.clamp_min(1e-5) * ray - ray0)
    return (rise / fk_rise.clamp_min(.002)).clamp(0., 1.5), valid

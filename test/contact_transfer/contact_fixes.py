"""Opt-in, process-local Newton contact corrections; installed packages stay untouched."""
from __future__ import annotations

import warp as wp
from newton._src.solvers.vbd import particle_vbd_kernels as particle
from newton._src.solvers.vbd import rigid_vbd_kernels as rigid

mat32 = particle.mat32


@wp.func
def _friction_scale(length: float, epsilon: float):
    if length > epsilon and length > 0.0:
        return 1.0 / length
    if epsilon > 0.0:
        return (2.0 - length / epsilon) / epsilon
    return 0.0


@wp.func
def projected_friction(mu: float, load: float, normal: wp.vec3, displacement: wp.vec3, epsilon: float):
    tangent = displacement - normal * wp.dot(normal, displacement)
    scale = mu * load * _friction_scale(wp.length(tangent), epsilon)
    return -scale * tangent, scale * (wp.identity(3, float) - wp.outer(normal, normal))


@wp.func
def surface_friction(mu: float, load: float, basis: mat32, displacement: wp.vec2, epsilon: float):
    scale = mu * load * _friction_scale(wp.length(displacement), epsilon)
    return -scale * (basis * displacement), scale * (basis * wp.transpose(basis))


def enable_friction_zero_fix():
    """Apply before the first Warp launch so dependent kernels compile against these functions.

    Zero slip has zero force but a finite tangent stiffness for positive epsilon.
    Both rigid-soft and soft-soft paths need the same continuous limit.
    """
    rigid.compute_projected_isotropic_friction = projected_friction
    particle.compute_friction = surface_friction

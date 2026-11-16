"""Analytic payload geometry and mass properties for the physical parcel."""
import math
import numpy as np


def rotation_matrix(euler):
    x, y, z = euler
    cx, cy, cz = np.cos(euler)
    sx, sy, sz = np.sin(euler)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1.]]) @ np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @ np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])


def mass_properties(shape, size, mass, gradient=(0, 0, 0)):
    """Positive linear density rho=rho0*(1+g dot x/half), integrated exactly.

    The gradient has sum(abs(g)) < 1, so density stays positive everywhere.
    Sphere/cylinder/capsule dimensions are their full axis-aligned bounds;
    cylinders and capsules have their symmetry axis along local x.
    """
    size = np.asarray(size, float)
    gradient = np.asarray(gradient, float)
    if size.shape != (3,) or np.any(size <= 0) or not np.isfinite(size).all():
        raise ValueError('payload dimensions must be three positive finite values')
    if gradient.shape != (3,) or not np.isfinite(gradient).all() or np.abs(gradient).sum() >= 1:
        raise ValueError('density gradient must be finite with L1 norm below one')
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('payload mass must be positive and finite')
    length, width, height = size
    if shape == 'box':
        volume = float(np.prod(size))
        inertia = mass * (np.sum(size**2) - size**2) / 12
    elif shape == 'sphere':
        if not np.allclose(size, size[0]):
            raise ValueError('sphere bounds must have equal dimensions')
        r = length / 2
        volume = 4 * math.pi * r**3 / 3
        inertia = np.full(3, 2 * mass * r*r / 5)
    elif shape in ('cylinder', 'capsule'):
        if not np.isclose(width, height):
            raise ValueError('cylinder/capsule transverse diameters must match')
        r = width / 2
        straight = length if shape == 'cylinder' else length - 2*r
        if straight <= 0:
            raise ValueError('capsule length must exceed its diameter')
        vc = math.pi * r*r * straight
        vs = 0 if shape == 'cylinder' else 4 * math.pi * r**3 / 3
        volume = vc + vs
        mc, ms = mass * vc / volume, mass * vs / volume
        axial = mc * r*r / 2 + ms * 2*r*r / 5
        transverse = mc * (3*r*r + straight**2) / 12 + ms * (2*r*r/5 + straight**2/4 + 3*straight*r/8)
        inertia = np.array([axial, transverse, transverse])
    else:
        raise ValueError('unsupported payload shape')
    second = (inertia.sum()/2 - inertia) / mass
    com = second * gradient / (size/2)
    tensor = np.diag(inertia) - mass*(np.dot(com,com)*np.eye(3)-np.outer(com,com))
    return dict(volume_m3=volume, density_kg_m3=mass/volume, com=com, inertia=tensor)


def inertia_xml(tensor):
    return [tensor[0,0], tensor[1,1], tensor[2,2], tensor[0,1], tensor[0,2], tensor[1,2]]

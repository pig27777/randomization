"""Fixed full-3D StVK/J2 template with physical parameters separated from numerics.

Positive yield thresholds retain the same plastic-state layout across X. A very
large threshold provides an effectively elastic control. The presets are
unmeasured candidates, not calibrated material classes or task guarantees.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from itertools import permutations
from types import SimpleNamespace

import mujoco
import numpy as np


@dataclass(frozen=True)
class Parameters:
    size: tuple = (.24, .16, .05)
    density: float = .29 / (.24 * .16 * .05)
    young: float = 10000.
    poisson: float = .2
    damping: float = .02
    yield_strain: float = .02
    friction: float = .8
    roundness: float = 0.
    bulge: float = 0.
    taper: float = 0.

    def __post_init__(self):
        size = np.asarray(self.size, dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("size must contain three finite positive lengths in metres")
        object.__setattr__(self, "size", tuple(float(x) for x in size))
        for name in ("density", "young", "yield_strain"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("damping", "friction"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.poisson) or not 0 <= self.poisson < .5:
            raise ValueError("poisson must lie in [0, .5)")
        for name, lo, hi in (("roundness", 0, 1), ("bulge", 0, .8), ("taper", -.5, .5)):
            value = getattr(self, name)
            if not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{name} must lie in [{lo}, {hi}]")


@dataclass(frozen=True)
class Template:
    count: tuple = (5, 4, 3)
    timestep: float = .0005
    iterations: int = 100
    ls_iterations: int = 50
    tolerance: float = 1e-8
    contact_time: float = .005
    radius: float = .001

    def __post_init__(self):
        count = tuple(self.count)
        if len(count) != 3 or any(isinstance(x, bool) or int(x) != x or x < 2 for x in count):
            raise ValueError("count must contain three integer vertex counts >= 2")
        object.__setattr__(self, "count", tuple(int(x) for x in count))
        for name in ("timestep", "tolerance", "contact_time", "radius"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("iterations", "ls_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def signature(self):
        payload = dict(law="MuJoCo4/full3D/StVK/J2", **asdict(self))
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def preset(name):
    """Same shape, density, topology and law; vary only physical material X."""
    values = {
        "hard": dict(young=100000., poisson=.25, yield_strain=.05),
        "bendable": dict(young=10000., poisson=.2, yield_strain=.02),
        "compressible": dict(young=3000., poisson=.05, yield_strain=.02),
    }
    if name not in values:
        raise ValueError(f"unknown continuum preset: {name!r}")
    return Parameters(**values[name])


def sample(rng, name="bendable"):
    """Bounded exploration around an anchor; dynamic feasibility is not implied."""
    p = preset(name)
    return replace(p, size=tuple(np.asarray(p.size) * np.exp(rng.uniform(-.15, .15, 3))),
                   density=p.density * np.exp(rng.uniform(-.2, .2)),
                   young=p.young * np.exp(rng.uniform(-.5, .5)),
                   poisson=float(np.clip(p.poisson + rng.uniform(-.04, .04), 0, .45)),
                   damping=p.damping * np.exp(rng.uniform(-.3, .3)),
                   yield_strain=p.yield_strain * np.exp(rng.uniform(-.3, .3)),
                   friction=p.friction * np.exp(rng.uniform(-.2, .2)),
                   roundness=float(rng.uniform(0, 1)), bulge=float(rng.uniform(0, .6)),
                   taper=float(rng.uniform(-.3, .3)))


def shape_parameters(name):
    """Named points in one continuous geometric parameter space."""
    shapes = {"box": (0., 0., 0.), "rounded": (.8, 0., 0.),
              "pillow": (.8, .6, 0.), "tapered": (.3, .2, .35)}
    if name not in shapes:
        raise ValueError(f"unknown continuum shape: {name!r}")
    return dict(zip(("roundness", "bulge", "taper"), shapes[name]))


def geometry(p, b):
    """Deform one fixed connectivity into box/rounded/pillow/tapered volumes.

    Material fills every tetrahedron. Mass follows the actual discretized volume,
    not the enclosing box. No surface-only visual deformation is used.
    """
    unit = np.stack(np.meshgrid(*(np.linspace(-1, 1, n) for n in b.count), indexing="ij"), axis=-1).reshape(-1, 3)
    u, v, w = unit.T
    radius = np.hypot(u, v)
    scale = np.divide(np.maximum(np.abs(u), np.abs(v)), radius,
                      out=np.ones_like(radius), where=radius > 0)
    scale = 1 - p.roundness + p.roundness * scale
    points = np.column_stack((u * scale, v * scale * (1 + p.taper * u),
                              w * (1 - p.bulge + p.bulge * np.sqrt(np.maximum(0, (1-u*u)*(1-v*v))))))
    low, high = points.min(0), points.max(0)
    points = (points - (low + high) / 2) * np.asarray(p.size) / (high - low)
    def index(q):
        return (q[0] * b.count[1] + q[1]) * b.count[2] + q[2]
    elements = []
    for origin in np.ndindex(*(n - 1 for n in b.count)):
        for order in permutations(range(3)):
            corners = [np.array(origin)]
            for axis in order:
                q = corners[-1].copy()
                q[axis] += 1
                corners.append(q)
            ids = [index(q) for q in corners]
            # Orientation depends only on the permutation, not on X.
            inversions = sum(order[i] > order[j] for i in range(3) for j in range(i+1, 3))
            if inversions % 2:
                ids[1], ids[2] = ids[2], ids[1]
            elements.append(ids)
    elements = np.asarray(elements, dtype=np.int32)
    t = points[elements]
    determinants = np.einsum("ij,ij->i", t[:,1]-t[:,0], np.cross(t[:,2]-t[:,0], t[:,3]-t[:,0]))
    if not np.isfinite(determinants).all() or np.any(determinants <= 1e-15):
        raise ValueError("shape parameters produce degenerate or inverted reference tetrahedra")
    volume = float(determinants.sum() / 6)
    top = np.flatnonzero(unit[:,2] == 1).tolist()
    return SimpleNamespace(points=points, elements=elements, volume=volume,
                           mass=p.density * volume, top=top)


def vec(values):
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).ravel())


def require_mujoco4():
    if mujoco.__version__.split(".")[0] != "4":
        raise RuntimeError("continuum requires the isolated MuJoCo 4 environment (.venv-continuum)")


def flex_xml(p, b, position, name="contents"):
    require_mujoco4()
    g = geometry(p, b)
    return f'''<flexcomp name="{name}" type="direct" dim="3" dof="full"
      point="{vec(g.points)}" element="{' '.join(str(int(x)) for x in g.elements.ravel())}" pos="{vec(position)}"
      mass="{g.mass:.17g}" radius="{b.radius:.17g}" rgba=".15 .65 .72 1">
      <elasticity young="{p.young:.17g}" poisson="{p.poisson:.17g}" damping="{p.damping:.17g}"
        yield="{p.yield_strain:.17g}"/>
      <contact selfcollide="none" contype="2" conaffinity="5" condim="3"
        friction="{p.friction:.17g} .005 .0001" solref="{b.contact_time:.17g} 1" solimp=".99 .99 .001"/>
    </flexcomp>'''


def option_xml(b, gravity=(0, 0, -9.81)):
    # Newton is the GPU constraint solver. No CG conversion or CPU dynamics fallback.
    return f'''<option timestep="{b.timestep:.17g}" integrator="discrete" solver="Newton"
      iterations="{b.iterations}" ls_iterations="{b.ls_iterations}" tolerance="{b.tolerance:.17g}"
      gravity="{vec(gravity)}"/>'''


def build(p=Parameters(), b=Template(), *, gravity=(0, 0, -9.81), table=True):
    """Compile geometry on CPU; dynamics are advanced by continuum.gpu only."""
    require_mujoco4()
    floor = (f'<geom name="table" type="plane" size="1 1 .02" contype="1" conaffinity="2" '
             f'friction="{p.friction} .005 .0001" solref="{b.contact_time} 1" solimp=".99 .99 .001"/>') if table else ""
    position = (0, 0, p.size[2] / 2 + .002)
    xml = f'''<mujoco model="continuum"><compiler angle="radian"/>{option_xml(b, gravity)}
      <size memory="128M"/><worldbody>{floor}{flex_xml(p, b, position)}</worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    # Geometry only. Do not invoke CPU dynamics as an initialization fallback.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_flex(model, data)
    return SimpleNamespace(model=model, data=data, xml=xml, parameters=p, template=b,
                           geometry=geometry(p, b))


def topology_signature(model):
    digest = hashlib.sha256()
    for value in (model.flex_dim, model.flex_elem, model.flex_edge, model.flex_vertbodyid):
        digest.update(value.tobytes())
    return digest.hexdigest()

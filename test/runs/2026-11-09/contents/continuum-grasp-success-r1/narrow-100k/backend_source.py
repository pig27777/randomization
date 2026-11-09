"""CUDA continuum contents on the existing full-YAM grasp protocol."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from grasp_platform import MuJoCoSim, TrialConfig, attach_robot, measure_contacts
from .gpu import Simulation
from .model import Template, flex_xml, geometry, option_xml, preset, shape_parameters, topology_signature


@dataclass
class Config(TrialConfig):
    dt: float = .0005
    pinch_coeff: float = 0.
    material: str | None = None
    shape: str = "box"
    size: tuple | None = None
    young: float | None = None
    poisson: float | None = None
    damping: float | None = None
    yield_strain: float | None = None
    density: float | None = None
    friction: float | None = None
    roundness: float | None = None
    bulge: float | None = None
    taper: float | None = None
    simulation_seconds: float = 30.

    def parameters(self):
        if self.pinch_coeff != 0:
            raise ValueError("continuum uses native contact; pass --pinch-coeff 0 to the shared platform")
        p = replace(preset(self.material or self.category), **shape_parameters(self.shape))
        return replace(p, **{name: getattr(self, name) for name in
                            ("size", "young", "poisson", "damping", "yield_strain", "density",
                             "friction", "roundness", "bulge", "taper")
                            if getattr(self, name) is not None})


class GpuSim(MuJoCoSim):
    def __init__(self, built):
        super().__init__(built)
        self.gpu = None
        bodies = self.m.flex_vertbodyid
        self.mass = self.m.body_mass[bodies].copy()

    def prepare(self):
        if self.gpu is None:
            # Yam finishes initialization after the sim factory returns.
            self.gpu = Simulation(self.m, self.d, nconmax=1024, njmax=4096)

    def step(self):
        self.prepare()
        self.gpu.step(ctrl=self.d.ctrl, external=self.d.xfrc_applied)
        self.gpu.sync(self.d)

    def refresh(self):
        self.prepare()
        # Match the shared platform's sampling-stage forward, on GPU only.
        import mujoco_warp as mjw
        mjw.forward(self.gpu.model, self.gpu.data)
        self.gpu.sync(self.d)

    def com(self):
        return np.average(self.d.flexvert_xpos, axis=0, weights=self.mass)


class NoPinch:
    active = None
    load = 0.
    events = ()

    def try_bind(self, tips):
        return False

    def check_load(self, *args):
        return False

    def release(self, *args):
        pass


def scene(cfg):
    p = cfg.parameters()
    b = Template(timestep=cfg.dt, contact_time=cfg.contact_time)
    g = geometry(p, b)
    center = np.array([-.25, .04])
    top = p.size[2] + .002
    xml = f'''<mujoco>{option_xml(b)}<size memory="128M"/>
      <worldbody><light pos="0 0 2"/>
      <geom name="table" type="box" pos="0 0 -.025" size="1 .22 .025" contype="1" conaffinity="6"
        friction=".8 .005 .0001" solref="{cfg.contact_time} 1" solimp=".99 .99 .001"/>
      {flex_xml(p,b,[*center,.002+p.size[2]/2])}
      </worldbody><contact/><equality/></mujoco>'''
    built = SimpleNamespace(xml=xml, desc=SimpleNamespace(), cfg=cfg, parameters=p, template=b)
    return attach_robot(cfg, built, GpuSim, center, top, p.size, g.mass)


def measure(robot):
    return measure_contacts(robot, target_flexes={0})


def attachment(robot, cfg):
    return NoPinch()


def refresh(sim):
    sim.refresh()


STATE_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


def record_state(sim):
    sim.prepare()
    return sim.gpu.state()[0]


def check_state(sim):
    if sim.gpu is None:
        # Preserve the original setup/CUDA exception before a GPU exists.
        return None, {}
    health = sim.gpu.health()[0]
    failures = [(1, "nonfinite state"), (2, "GPU buffer overflow"), (4, "inverted tetrahedron"),
                (8, "Newton iteration cap"), (16, "effective-mass residual")]
    bits = int(health["failure_bits"])
    reason = ("continuum GPU: " + ", ".join(name for bit, name in failures if bits & bit)
              + f" at step {int(health['steps'])}") if bits else None
    return reason, dict(min_determinant=health["minimum_J"], plastic_norm=health["plastic_norm"],
                        smooth_relative_residual=health["peak_relative_residual"],
                        gpu_overflow=health["overflow_or"])


def summarize(trace):
    return dict(min_determinant=min((x["min_determinant"] for x in trace), default=None),
                peak_plastic_norm=max((x["plastic_norm"] for x in trace), default=None),
                peak_relative_residual=max((x["smooth_relative_residual"] for x in trace), default=None))


def dimensions(sim):
    b, p = sim.built.template, sim.built.parameters
    return dict(nv=sim.m.nv, vertices=sim.m.nflexvert, tetrahedra=sim.m.nflexelem,
                nplastic=sim.m.nplastic, representation="full_3d_elastoplastic_continuum",
                parameters_X=asdict(p), template_B=asdict(b), template_sha256=b.signature,
                topology_sha256=topology_signature(sim.m), mass_kg=float(sim.mass.sum()),
                size_mm=(np.asarray(p.size)*1000).tolist(), native_contact_only=True,
                gpu_health=sim.gpu.health() if sim.gpu else None,
                software=sim.gpu.software() if sim.gpu else None)


def source_files():
    here = Path(__file__).parent
    return {"continuum_model": here/"model.py", "continuum_gpu": here/"gpu.py",
            "continuum_residual": here.parents[1]/"bag_sim"/"flex_residual.py"}

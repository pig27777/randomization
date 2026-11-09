"""Frozen v1.3 segmented contents on the common articulated-YAM platform."""
from dataclasses import dataclass, replace
from xml.etree import ElementTree as ET

import numpy as np

from grasp_platform import MuJoCoSim, TrialConfig, attach_robot, measure_contacts
import mj_adapter as A
from model import ANCHORS, build
from pinch import PinchAttachment, fragments


@dataclass
class Config(TrialConfig):
    # The released bushing model requires its validated CPU timestep.
    dt: float = .0004
    size: tuple | None = None
    mass: float | None = None

    def __post_init__(self):
        if self.size is not None:
            value = np.asarray(self.size, dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError('size requires three finite positive lengths in metres')
            self.size = tuple(float(v) for v in value)
        if self.mass is not None and (not np.isfinite(self.mass) or self.mass <= 0):
            raise ValueError('mass must be finite and positive')


class SegmentSim(A.Sim):
    def __init__(self, built):
        # Apply the common robot/pad XML transformation, then initialize all
        # frozen-model caches against that complete articulated scene.
        MuJoCoSim.__init__(self, built)
        super().__init__(built)


class SegmentPinch(PinchAttachment):
    def __init__(self, robot):
        super().__init__(robot.sim, robot.sim.built.rest_top)
        self.frame = self.sim.m.body('pinch_frame').id
        self.point = None

    def set_frame(self, position, quaternion=None):
        # Store a binding point; the actual frame follows the dynamic palm.
        self.point = np.asarray(position, dtype=float).copy()

    def bind_segments(self, segments):
        if not segments or any(i not in self.equations for i in segments):
            raise ValueError('pinch attachment requires top-layer segments')
        m, d = self.sim.m, self.sim.d
        ids = []
        for i in segments:
            body, eq = self.sim.seg_bodies[i], self.equations[i]
            m.eq_data[eq, :3] = d.xmat[self.frame].reshape(3, 3).T @ (self.point - d.xpos[self.frame])
            m.eq_data[eq, 3:6] = d.xmat[body].reshape(3, 3).T @ (self.point - d.xpos[body])
            d.eq_active[eq] = True
            ids.append(eq)
        self.active, self.segment, self.segments = ids, segments[0], list(segments)
        self.load = 0.
        self.events.append(dict(t=float(d.time), event='bind', segment=self.segment, segments=self.segments))

    def try_bind(self, tips):
        return False if self.coefficient == 0 else super().try_bind(tips)

    def check_load(self, dt, grip_force, force_vector=None):
        force = self.force_vector() if force_vector is None else force_vector
        # Same I-37 direction convention as the native-flex platform adapter.
        load = np.array([force[0], force[1], min(float(force[2]), 0.)])
        return super().check_load(dt, grip_force, load)


def scene(cfg):
    anchor = ANCHORS[cfg.category]
    kappa = replace(anchor, pinch_coeff=cfg.pinch_coeff,
                    size=cfg.size if cfg.size is not None else anchor.size,
                    mass=cfg.mass if cfg.mass is not None else anchor.mass)
    desc = build(kappa, omega_max=.8 / cfg.dt,
                 geometry=cfg.category if cfg.size is not None else None)
    center = np.array([-.25, .04])
    top = kappa.size[2] + .002
    _, equality = fragments(desc)
    built = A.scene(desc, [*center, .002 + kappa.size[2] / 2], dt=cfg.dt,
                    equality=equality)
    root = ET.fromstring(built.xml)
    table = root.find(".//geom[@name='table']")
    table.attrib.update(pos='0 0 -.025', size='1 .22 .025', contype='1', conaffinity='6',
                        friction='.8 .005 .0001', solref=f'{cfg.contact_time} 1', solimp='.99 .99 .001')
    built.xml = ET.tostring(root, encoding='unicode')
    built.cfg, built.rest_top = cfg, top
    robot = attach_robot(cfg, built, SegmentSim, center, top, kappa.size, kappa.mass)
    robot.sim.pad_geoms = {side: robot.all_finger_sets[j] for j, side in enumerate(('left', 'right'))}
    return robot


def measure(robot):
    return measure_contacts(robot, target_geoms=robot.sim.obj_geoms)


def attachment(robot, cfg):
    return SegmentPinch(robot)


def check_state(sim):
    return sim.failure(), dict(max_strain=sim.max_strain(),
                               max_relative_angle_deg=sim.max_relative_angle_deg(),
                               confinement_N=float(sim.external_compaction_load))


def summarize(trace):
    return {key: max((row[key] for row in trace), default=None)
            for key in ('max_strain', 'max_relative_angle_deg', 'confinement_N')}


def dimensions(sim):
    return dict(nv=sim.m.nv, segments=len(sim.desc.segments),
                elastic_connections=len(sim.built.bushings),
                size_mm=(np.array(sim.desc.kappa.size) * 1000).tolist(),
                mass_kg=sim.mass)

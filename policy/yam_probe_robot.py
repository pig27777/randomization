"""Frozen-content scene adapter for the shared experience YAM robot.

The side-grasp contact convention and initial content orientation are retained.
Arm assembly, IK, control and diagnostics live only in experience/yam_platform.
"""
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
PLATFORM_ROOT = HERE / "experience" if (HERE / "experience/yam_platform").is_dir() else HERE.parent / "experience"
sys.path.insert(0, str(PLATFORM_ROOT))
from yam_platform.robot import ASSET, ROTATION, Yam as PlatformYam, asset_fingerprint, rotation_error
from yam_platform.provenance import source_files as platform_source_files, snapshot as snapshot_platform


class Yam(PlatformYam):
    def __init__(self, k, seed, *, fixture=None):
        if fixture is None:
            import mj_adapter as A
            from model import build

            carry = np.array([.5 if seed % 2 == 0 else -.5, 0., 0.])
            pick = -carry / 2
            target = pick + [0., 0., k.size[2] / 2 + .0001]
            desc = build(k, omega_max=A.OMEGA_MAX)
            built = A.scene(desc, target, pads=())

            def make_sim(built):
                root = ET.fromstring(built.xml)
                table = root.find("worldbody/geom[@name='table']")
                table.set('pos', '0 0 -0.025')
                table.set('size', '1 .22 .025')
                # Preserve the original side-grasp 40 N/mm contact convention.
                for side in ('left', 'right'):
                    finger = root.find(f".//body[@name='{side}_finger']")
                    for geom in finger.iter('geom'):
                        if geom.get('class') == 'visual':
                            continue
                        for i, segment in enumerate(desc.segments):
                            mass = k.mass if k.rigid else segment.mass
                            kd = min(40000. * (1-A.CONTACT_D)/(A.CONTACT_D*mass), (.5/A.BUSHING_DT)**2/A.CONTACT_D)
                            damping = 2*np.sqrt(kd)
                            bf = min(max(damping,1000.), 1/(2.5*A.CONTACT_D*A.BUSHING_DT))
                            ET.SubElement(root.find('contact'), 'pair', geom1=geom.get('name'), geom2=f's{i}', condim='3',
                                friction=f'{k.surface_friction} {k.surface_friction} .005 .0001 .0001',
                                solref=f'{-kd} {-damping}', solreffriction=f'{-kd} {-bf}', solimp='.99 .99 .001')
                built.xml = ET.tostring(root, encoding='unicode')
                sim = A.Sim(built)
                # Package long axis and its orientation match the historical probe.
                for j in range(sim.m.njnt):
                    if sim.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                        a = sim.m.jnt_qposadr[j]
                        p = sim.d.qpos[a:a+2] - pick[:2]
                        sim.d.qpos[a:a+2] = pick[:2] + [-p[1], p[0]]
                        sim.d.qpos[a+3:a+7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
                return sim

            fixture = dict(built=built, target=target, rotation=ROTATION, sim_factory=make_sim)
        super().__init__(k, seed, fixture=fixture)

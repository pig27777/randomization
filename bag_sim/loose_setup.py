"""Shared construction and placement for loose-bag model variants."""
import hashlib
import math
import os
from pathlib import Path
import time

import mujoco
import numpy as np

from mj_loose import LooseBag, LooseSim, loose_scene
from model import ANCHORS, Kappa, build, load_table
from mj_adapter import OMEGA_MAX

BOX = dict(loose=Kappa(size=(0.12, 0.08, 0.03), mass=0.3, segments=(1, 1, 1), rigid=True, squeeze_stiffness_y=50.,
                       squeeze_stiffness_z=50., lock_strain_y=0.02, lock_strain_z=0.02, droop_ratio=0.0),
           full=Kappa(size=(0.20, 0.15, 0.03), mass=0.5, segments=(1, 1, 1), rigid=True, squeeze_stiffness_y=50.,
                      squeeze_stiffness_z=50., lock_strain_y=0.02, lock_strain_z=0.02, droop_ratio=0.0),
           # round 2 F4: tight fit, 10 mm to the 350 x 250 mm bag body on every side
           fit=Kappa(size=(0.33, 0.23, 0.03), mass=0.5, segments=(1, 1, 1), rigid=True, squeeze_stiffness_y=50.,
                     squeeze_stiffness_z=50., lock_strain_y=0.02, lock_strain_z=0.02, droop_ratio=0.0))


KRAFT_SIZE = (0.35, 0.25)


def content_kappa(name):
    return BOX[name.split("_")[1]] if name.startswith("box") else ANCHORS[name]


def _rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def rotate_package(sim: LooseSim, center, yaw):
    """Rotate every free body (content segments, plate roots) and every flex vertex about the vertical
    axis through center (policy/env.py _rotate_object, extended to slide-joint flex vertices)."""
    m, d = sim.m, sim.d
    q = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    R = _rot(yaw)
    pad_joints = {m.joint(f"{p}_{a}").id for p in sim.pad_names for a in "xyz"}
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        a = m.jnt_qposadr[j]
        p = d.qpos[a:a + 3].copy()
        p[:2] = center[:2] + R @ (p[:2] - center[:2])
        d.qpos[a:a + 3] = p
        qn = np.zeros(4)
        mujoco.mju_mulQuat(qn, q, d.qpos[a + 3:a + 7].copy())
        d.qpos[a + 3:a + 7] = qn
    if sim.bag.kind == "flex":
        for b in sim.bag_bodies:
            j0 = m.body_jntadr[b]
            if j0 < 0 or j0 in pad_joints:
                continue
            adr = m.jnt_qposadr[j0:j0 + 3]
            x = m.body_pos[b] + d.qpos[adr]
            x[:2] = center[:2] + R @ (x[:2] - center[:2])
            d.qpos[adr] = x - m.body_pos[b]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


def make(k: Kappa, bag: LooseBag, pads, table_xmin=-0.6, big_table=False, dt=None, fingertips=(), cache_dir=None):
    """dt: scene step (None: the content model's 0.4 ms). fingertips: v2 fingertips instead of the 40 mm pads."""
    desc = build(k, omega_max=OMEGA_MAX, table=load_table())
    if bag.kind == 'effect':
        from effect_model import effect_scene
        lb = effect_scene(desc, bag, pads, table_xmin=table_xmin, dt=dt, fingertips=fingertips)
    else:
        lb = loose_scene(desc, bag, pads, table_xmin=table_xmin, dt=dt, fingertips=fingertips)
    if big_table:      # policy/env.py: a 4 x 4 m table for the 0.5 m carry in any direction
        import re
        lb.built.xml = re.sub(r'<geom name="table"[^>]*?size="[^"]*"', lambda mt: re.sub(
            r'pos="[^"]*" size="[^"]*"', 'pos="0 0 -0.025" size="2 2 0.025"', mt.group(0)), lb.built.xml, count=1)
    if bag.kind == 'effect':
        from effect_model import EffectSim
        sim = EffectSim(lb, [p[0] for p in (pads or fingertips)])
    else:
        sim = LooseSim(lb, [p[0] for p in (pads or fingertips)])
    t0 = time.perf_counter()
    sim.cache_hit = False
    cache = None
    if cache_dir is not None:
        if not k.rigid:
            raise ValueError('F6 cache currently supports rigid content only')
        from mj_loose import CONTENT_RELEASE_SHA256
        key = hashlib.sha256((lb.built.xml + CONTENT_RELEASE_SHA256 + str(bag)).encode()).hexdigest()
        cache = Path(cache_dir) / (key + '.npz')
    state_type = mujoco.mjtState.mjSTATE_INTEGRATION
    if cache is not None and cache.exists():
        with np.load(cache) as saved:
            mujoco.mj_setState(sim.m, sim.d, saved['state'], state_type)
            warm = sim.d.qacc_warmstart.copy()
            mujoco.mj_forward(sim.m, sim.d)
            sim.d.qacc_warmstart[:] = warm
        sim.cache_hit = True
    else:
        sim.wrap()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            state = np.empty(mujoco.mj_stateSize(sim.m, state_type))
            mujoco.mj_getState(sim.m, sim.d, state, state_type)
            tmp = cache.with_name(cache.stem + f'.{os.getpid()}.npz')
            np.savez(tmp, state=state)
            tmp.replace(cache)
    sim.wrap_wall_s = time.perf_counter() - t0
    return sim


def nearest_bag_body(sim: LooseSim, p):
    if sim.bag.kind == 'effect':
        return int(sim.bag_bodies[0])
    x = sim.bag_points()
    return int(sim.bag_bodies[np.argmin(np.linalg.norm(x - p, axis=1))])

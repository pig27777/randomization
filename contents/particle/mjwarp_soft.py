"""Particle-bond soft model on MuJoCo Warp: the same model as particle_soft.py, batched on the GPU.

The MuJoCo model (node bodies, slide joints, contact skin) is built by particle_soft.Soft.
Bond and column forces run as Warp kernels into qfrc_applied; history is per world.

Usage: mjwarp_soft.py check | speed [--worlds 1 64 256 1024]
"""
from __future__ import annotations

import argparse
import json
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

import particle_soft as ps


@wp.kernel
def bond_kernel(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), start: wp.array(dtype=wp.vec3), q0: int,
                bi: wp.array(dtype=int), bj: wp.array(dtype=int), L0: wp.array(dtype=float), kt: wp.array(dtype=float),
                s1: wp.array(dtype=float), s2: wp.array(dtype=float), e1: wp.array(dtype=float), e2: wp.array(dtype=float),
                klock: wp.array(dtype=float), fric: wp.array(dtype=float), kf: wp.array(dtype=float), c: wp.array(dtype=float),
                v: wp.array2d(dtype=float), s: wp.array2d(dtype=float), delta_out: wp.array2d(dtype=float),
                elastic_out: wp.array2d(dtype=float), slip_out: wp.array2d(dtype=float), qfrc: wp.array2d(dtype=float)):
    w, k = wp.tid()
    i, j = bi[k], bj[k]
    pi = start[i] + wp.vec3(qpos[w, q0 + 3 * i], qpos[w, q0 + 3 * i + 1], qpos[w, q0 + 3 * i + 2])
    pj = start[j] + wp.vec3(qpos[w, q0 + 3 * j], qpos[w, q0 + 3 * j + 1], qpos[w, q0 + 3 * j + 2])
    vi = wp.vec3(qvel[w, q0 + 3 * i], qvel[w, q0 + 3 * i + 1], qvel[w, q0 + 3 * i + 2])
    vj = wp.vec3(qvel[w, q0 + 3 * j], qvel[w, q0 + 3 * j + 1], qvel[w, q0 + 3 * j + 2])
    d = pj - pi
    L = wp.max(wp.length(d), 1e-9)
    n = d / L
    stretch = L - L0[k]
    push = _compression(w, k, wp.max(-stretch, 0.0), s1, s2, e1, e2, klock, fric, kf, v, s, delta_out, elastic_out, slip_out)
    mag = push - kt[k] * wp.max(stretch, 0.0) - c[k] * wp.dot(vj - vi, n)
    _add(qfrc, w, q0, j, mag * n)
    _add(qfrc, w, q0, i, -mag * n)


@wp.func
def _compression(w: int, k: int, delta: float, s1: wp.array(dtype=float), s2: wp.array(dtype=float), e1: wp.array(dtype=float),
                 e2: wp.array(dtype=float), klock: wp.array(dtype=float), fric: wp.array(dtype=float), kf: wp.array(dtype=float),
                 v: wp.array2d(dtype=float), s: wp.array2d(dtype=float), delta_out: wp.array2d(dtype=float),
                 elastic_out: wp.array2d(dtype=float), slip_out: wp.array2d(dtype=float)):
    e = wp.max(delta - v[w, k], 0.0)
    elastic = (s1[k] * wp.min(e, e1[k]) + s2[k] * wp.clamp(e - e1[k], 0.0, wp.max(e2[k] - e1[k], 0.0))
               + klock[k] * wp.max(e - e2[k], 0.0))
    slip = wp.clamp(kf[k] * (delta - s[w, k]), -fric[k], fric[k])
    delta_out[w, k] = delta
    elastic_out[w, k] = elastic
    slip_out[w, k] = slip
    return wp.max(elastic + slip, 0.0)


@wp.func
def _add(qfrc: wp.array2d(dtype=float), w: int, q0: int, node: int, f: wp.vec3):
    wp.atomic_add(qfrc, w, q0 + 3 * node, f[0])
    wp.atomic_add(qfrc, w, q0 + 3 * node + 1, f[1])
    wp.atomic_add(qfrc, w, q0 + 3 * node + 2, f[2])


@wp.kernel
def column_kernel(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), start: wp.array(dtype=wp.vec3), q0: int,
                  bi: wp.array(dtype=int), bj: wp.array(dtype=int), nb: wp.array2d(dtype=int), L0: wp.array(dtype=float),
                  kt: wp.array(dtype=float), s1: wp.array(dtype=float), s2: wp.array(dtype=float), e1: wp.array(dtype=float),
                  e2: wp.array(dtype=float), klock: wp.array(dtype=float), fric: wp.array(dtype=float), kf: wp.array(dtype=float),
                  c: wp.array(dtype=float), k_shear: float, slide_range: float, v: wp.array2d(dtype=float), s: wp.array2d(dtype=float),
                  delta_out: wp.array2d(dtype=float), elastic_out: wp.array2d(dtype=float), slip_out: wp.array2d(dtype=float),
                  qfrc: wp.array2d(dtype=float)):
    """Compression along the local normal; shear as a separate tangential force (see particle_soft.Soft._column_forces)."""
    w, k = wp.tid()
    i, j = bi[k], bj[k]
    p = _pos(qpos, start, q0, w, i)
    pj = _pos(qpos, start, q0, w, j)
    normal = wp.cross(_pos(qpos, start, q0, w, nb[k, 1]) - _pos(qpos, start, q0, w, nb[k, 0]),
                      _pos(qpos, start, q0, w, nb[k, 3]) - _pos(qpos, start, q0, w, nb[k, 2]))
    normal = normal / wp.max(wp.length(normal), 1e-9)
    d = pj - p
    gap = wp.dot(d, normal)
    stretch = gap - L0[k]
    rel = _vel(qvel, q0, w, j) - _vel(qvel, q0, w, i)
    push = _compression(w, k, wp.max(-stretch, 0.0), s1, s2, e1, e2, klock, fric, kf, v, s, delta_out, elastic_out, slip_out)
    mag = push - kt[k] * wp.max(stretch, 0.0) - c[k] * wp.dot(rel, normal)
    tangent = d - gap * normal
    rel_t = rel - wp.dot(rel, normal) * normal
    length = wp.length(tangent)
    hard = 300.0 * wp.max(length - slide_range, 0.0) / wp.max(length, 1e-9)
    total = mag * normal - (k_shear + hard) * tangent - 0.02 * rel_t
    _add(qfrc, w, q0, j, total)
    _add(qfrc, w, q0, i, -total)


@wp.func
def _pos(qpos: wp.array2d(dtype=float), start: wp.array(dtype=wp.vec3), q0: int, w: int, i: int):
    return start[i] + wp.vec3(qpos[w, q0 + 3 * i], qpos[w, q0 + 3 * i + 1], qpos[w, q0 + 3 * i + 2])


@wp.func
def _vel(qvel: wp.array2d(dtype=float), q0: int, w: int, i: int):
    return wp.vec3(qvel[w, q0 + 3 * i], qvel[w, q0 + 3 * i + 1], qvel[w, q0 + 3 * i + 2])


@wp.kernel
def history_kernel(delta: wp.array2d(dtype=float), elastic: wp.array2d(dtype=float), slip: wp.array2d(dtype=float),
                   fric: wp.array(dtype=float), kf: wp.array(dtype=float), k3: wp.array(dtype=float), tau: wp.array(dtype=float),
                   vmax: wp.array(dtype=float), dt: float, v: wp.array2d(dtype=float), s: wp.array2d(dtype=float)):
    w, k = wp.tid()
    if fric[k] > 0.0:
        s[w, k] = delta[w, k] - slip[w, k] / kf[k]
    if k3[k] > 0.0:
        rate = (elastic[w, k] / k3[k] - v[w, k]) / tau[k]
        v[w, k] = wp.clamp(v[w, k] + rate * dt, 0.0, vmax[k])


class GpuBonds:
    def __init__(self, bonds, nworld, nb=None):
        f = lambda a, t=float: wp.array(np.asarray(a), dtype=t)
        self.n = len(bonds.i)
        self.i, self.j = f(bonds.i, int), f(bonds.j, int)
        self.L0, self.kt, self.s1, self.s2 = f(bonds.L0), f(bonds.kt), f(bonds.s1), f(bonds.s2)
        self.e1, self.e2, self.klock, self.f = f(bonds.e1), f(bonds.e2), f(bonds.klock), f(bonds.f)
        self.kf, self.c, self.k3, self.tau, self.vmax = f(bonds.kf), f(bonds.c), f(bonds.k3), f(bonds.tau), f(bonds.vmax)
        self.nb = wp.array(np.asarray(nb, dtype=np.int32), dtype=int) if nb is not None else None
        z = lambda: wp.zeros((nworld, self.n), dtype=float)
        self.v, self.s, self.delta, self.elastic, self.slip = z(), z(), z(), z(), z()


class BatchedSoft:
    """nworld copies of one particle_soft model; per-world history; GPU forces."""

    def __init__(self, soft: ps.Soft, nworld, nconmax=4000, njmax=12000):
        self.soft, self.nworld = soft, nworld
        self.mjm = soft.model
        self.m = mjw.put_model(self.mjm)
        # Only silence the CG iteration-cap notice; buffer overflows must stay visible.
        self.m.opt.warn_overflow &= ~mjw.OverflowType.ITERATIONS
        self.d = mjw.make_data(self.mjm, nworld=nworld, nconmax=nconmax, njmax=njmax)
        self.start = wp.array(soft.start.astype(np.float32), dtype=wp.vec3)
        self.q0 = int(soft.q0)
        self.bonds = GpuBonds(soft.bonds, nworld)
        self.cols = GpuBonds(soft.cols, nworld, soft.cols.nb) if soft.cols is not None else None
        self.extra = wp.zeros((nworld, self.mjm.nv), dtype=float)  # forces from tools and tests
        self.dt = float(self.mjm.opt.timestep)

    def substep(self):
        d, b = self.d, self.bonds
        wp.copy(d.qfrc_applied, self.extra)
        wp.launch(bond_kernel, dim=(self.nworld, b.n), inputs=[d.qpos, d.qvel, self.start, self.q0, b.i, b.j, b.L0, b.kt, b.s1, b.s2,
                                                            b.e1, b.e2, b.klock, b.f, b.kf, b.c, b.v, b.s, b.delta, b.elastic, b.slip, d.qfrc_applied])
        if self.cols is not None:
            c, x = self.cols, self.soft.x
            wp.launch(column_kernel, dim=(self.nworld, c.n), inputs=[d.qpos, d.qvel, self.start, self.q0, c.i, c.j, c.nb, c.L0, c.kt, c.s1,
                                                                  c.s2, c.e1, c.e2, c.klock, c.f, c.kf, c.c, float(x.k_shear), float(x.slide_range),
                                                                  c.v, c.s, c.delta, c.elastic, c.slip, d.qfrc_applied])
        mjw.step(self.m, d)
        for g in (self.bonds, self.cols) if self.cols is not None else (self.bonds,):
            wp.launch(history_kernel, dim=(self.nworld, g.n), inputs=[g.delta, g.elastic, g.slip, g.f, g.kf, g.k3, g.tau, g.vmax, self.dt, g.v, g.s])

    def capture(self, n):
        with wp.ScopedCapture() as cap:
            for _ in range(n):
                self.substep()
        self.graph = cap.graph

    def run_graph(self):
        wp.capture_launch(self.graph)

    def positions(self, world=0):
        qpos = self.d.qpos.numpy()[world]
        return self.soft.start + qpos[self.q0:self.q0 + 3 * self.soft.n].reshape(-1, 3)


BOX = ('<body name="box" pos="0 0 0.075"><freejoint/>'
       '<geom name="boxg" type="box" size=".12 .1 .01" mass="2" contype="1" conaffinity="2" condim="3" friction=".8 .005 .0001" '
       'solref="-2e4 -60" solimp="0.95 0.99 0.0005" rgba=".3 .5 .85 1"/></body>')
FLOOR = f'<geom name="floor" type="plane" size="2 2 .01" {ps.TOOL_CONTACT.format(mu=0.6)}/>'


def under_box_drop(start_pos, pos):
    top = np.where(start_pos[:, 2] > start_pos[:, 2].max() - 1e-3)[0]
    under = top[(np.abs(start_pos[top, 0]) < 0.1) & (np.abs(start_pos[top, 1]) < 0.08)]
    return float((start_pos[under, 2] - pos[under, 2]).mean() * 1e3)


def check():
    """Same tests on CPU MuJoCo and MuJoCo Warp: direct force on the top layer, and a 2 kg box resting on top."""
    x = ps.SoftX()
    out = {}
    # CPU reference: direct 20 N on the top layer
    soft = ps.Soft(ps.build_flat(x), x, '', FLOOR)
    top = np.where(soft.start[:, 2] > soft.start[:, 2].max() - 1e-3)[0]
    for _ in range(int(3 / soft.dt)):
        soft.apply()
        soft.data.qfrc_applied[soft.d0 + 3 * top + 2] += -20.0 / len(top)
        mujoco.mj_step(soft.model, soft.data)
        soft.advance_history()
    p = soft.positions()
    out['cpu_direct_20N_top_drop_mm'] = float((soft.start[top, 2] - p[top, 2]).mean() * 1e3)
    # CPU: box
    soft = ps.Soft(ps.build_flat(x), x, BOX, FLOOR)
    for _ in range(int(3 / soft.dt)):
        soft.apply()
        mujoco.mj_step(soft.model, soft.data)
        soft.advance_history()
    out['cpu_box_drop_mm'] = under_box_drop(soft.start, soft.positions())
    out['cpu_box_z'] = float(soft.data.xpos[soft.model.body('box').id][2])
    # GPU: box, 4 worlds
    soft = ps.Soft(ps.build_flat(x), x, BOX, FLOOR)
    gpu = BatchedSoft(soft, 4)
    for _ in range(int(3 / gpu.dt)):
        gpu.substep()
    out['gpu_box_drop_mm_per_world'] = [under_box_drop(soft.start, gpu.positions(w)) for w in range(4)]
    out['gpu_box_z'] = float(gpu.d.xpos.numpy()[0][soft.model.body('box').id][2])
    out['nefc_max'] = int(gpu.d.nefc.numpy().max()) if hasattr(gpu.d, 'nefc') else None
    return out


def speed(worlds, nconmax=4000, njmax=2000):
    x = ps.SoftX()
    res = {}
    for n in worlds:
        soft = ps.Soft(ps.build_flat(x), x, BOX, FLOOR)
        gpu = BatchedSoft(soft, n, nconmax=nconmax, njmax=njmax)
        frame = int(round(0.01 / gpu.dt))
        for _ in range(frame):
            gpu.substep()
        gpu.capture(frame)
        gpu.run_graph()
        wp.synchronize()
        t0 = time.perf_counter()
        frames = 20
        for _ in range(frames):
            gpu.run_graph()
        wp.synchronize()
        wall = time.perf_counter() - t0
        res[n] = dict(env_frames_per_s=n * frames / wall, frames_per_s_per_env=frames / wall,
                      finite=bool(np.isfinite(gpu.d.qpos.numpy()).all()),
                      box_drop_mm_world0=under_box_drop(soft.start, gpu.positions(0)), max_nefc=int(gpu.d.nefc.numpy().max()))
        print(n, json.dumps(res[n]), flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['check', 'speed'])
    ap.add_argument('--worlds', type=int, nargs='+', default=[1, 64, 256])
    a = ap.parse_args()
    print(json.dumps(check() if a.mode == 'check' else speed(a.worlds), indent=1))

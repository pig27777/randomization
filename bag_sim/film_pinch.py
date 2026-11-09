"""F7: v1.3 point-pinch rule on upper flex vertices; content remains purely physical."""
import math

import mujoco
import numpy as np

from frozen_content import activate

activate()

from pinch import PinchAttachment, DEPTH_MIN


class FilmPinch(PinchAttachment):
    def __init__(self, sim, coefficient):
        self.sim = sim
        self.coefficient = float(coefficient)
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError('nonnegative finite bag_pinch_coeff required')
        self.tip_width = .020
        self.upper = np.asarray(sim.meta['upper'])
        self.top = self.upper.tolist()
        self.equations = {int(i): sim.m.equality(f'film_pinch_{i}').id for i in self.upper}
        self.mocap = sim.m.body_mocapid[sim.m.body('pinch_frame').id]
        self.active = self.segment = None
        self.segments, self.events = [], []
        self.broken = False
        self.load = self.peak_force = 0.
        self.contact_height = None
        self.upper_faces = np.all(np.isin(sim.meta['faces'], self.upper), axis=1)

    def top_contacts(self):
        found = {p: set() for p in self.sim.pad_names}
        for c in self.sim.d.contact:
            if c.efc_address < 0 or self.sim.d.efc_force[c.efc_address] <= .05:
                continue
            for p in found:
                side = 1 if c.geom1 in self.sim.pad_geoms[p] else 0 if c.geom2 in self.sim.pad_geoms[p] else -1
                if side < 0 or c.flex[side] != 0:
                    continue
                face, vertex = int(c.elem[side]), int(c.vert[side])
                if face >= 0 and self.upper_faces[face]:
                    found[p].update(int(i) for i in self.sim.meta['faces'][face])
                elif vertex in self.equations:
                    found[p].add(vertex)
        return tuple(found[p] for p in self.sim.pad_names)

    def covered(self, tips):
        midpoint = tips.mean(0)
        along = tips[1, :2] - tips[0, :2]
        length = np.linalg.norm(along)
        if length < 1e-12:
            return []
        along /= length
        across = np.array([-along[1], along[0]])
        delta = self.sim.bag_points()[self.upper, :2] - midpoint[:2]
        selected = self.upper[(np.abs(delta @ along) <= length / 2 + 1e-9) &
                              (np.abs(delta @ across) <= self.tip_width / 2 + 1e-9)]
        return sorted(selected.tolist(), key=lambda i: (np.linalg.norm(self.sim.bag_points()[i] - midpoint), i))

    def candidate(self, tips):
        if self.active is not None or self.broken or self.contact_height is None:
            return None
        if np.any(self.contact_height - tips[:, 2] < DEPTH_MIN - 1e-12):
            return None
        if not all(self.top_contacts()):
            return None
        return self.covered(tips) or None

    def bind_segments(self, segments):
        point = self.sim.d.mocap_pos[self.mocap].copy()
        for i in segments:
            eq = self.equations[i]
            body = self.sim.bag_bodies[i]
            self.sim.m.eq_data[eq, :3] = 0.
            self.sim.m.eq_data[eq, 3:6] = point - self.sim.d.xpos[body]
            self.sim.d.eq_active[eq] = True
        self.active = [self.equations[i] for i in segments]
        self.segment, self.segments = segments[0], segments
        self.load = 0.
        self.events.append(dict(t=float(self.sim.d.time), event='bind', vertices=segments,
                                point=point.tolist(), lower_vertices=0))


def contact_forces(sim):
    """Signed real fingertip reactions from bag contacts, including tangential force."""
    out = {p: np.zeros(3) for p in sim.pad_names}
    if sim.bag.kind == 'effect':
        from top_grasp import reaction_index, contact_reactions
        index = reaction_index(sim)
        index[1][sim.m.geom('bag_envelope').id] = True
        return contact_reactions(sim, index)[0]
    c, d = sim.d.contact, sim.d
    if not d.ncon:
        return out
    address = c.efc_address
    j = np.arange(3)
    valid = (address[:, None] >= 0) & (j < c.dim[:, None])
    local = np.where(valid, d.efc_force[np.where(valid, address[:, None] + j, 0)], 0.)
    on2 = np.einsum('nki,nk->ni', c.frame.reshape(-1, 3, 3), local)
    for p, gs in sim.pad_geoms.items():
        out[p] -= on2[np.isin(c.geom[:, 0], list(gs)) & (c.flex[:, 1] == 0)].sum(0)
        out[p] += on2[np.isin(c.geom[:, 1], list(gs)) & (c.flex[:, 0] == 0)].sum(0)
    return out

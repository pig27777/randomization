"""Legacy probe contact accounting, preserved from frozen v1.3 top_grasp."""
import numpy as np


def reaction_index(sim):
    """Geom lookups for contact_reactions: fingertip index per geom (-1 elsewhere) and content geoms."""
    pad = np.full(sim.m.ngeom, -1)
    for k, p in enumerate(sim.pad_names):
        pad[sorted(sim.pad_geoms[p])] = k
    content = np.zeros(sim.m.ngeom, bool)
    content[sorted(sim.obj_geoms)] = True
    return pad, content


def contact_reactions(sim, index):
    """sim.contact_forces()[:2] (force on each fingertip by the object, table force on the object),
    vectorized for the per-step loop; contact-frame forces are the elliptic-cone efc rows."""
    pad, content = index
    d = sim.d
    pads = {p: np.zeros(3) for p in sim.pad_names}
    table = np.zeros(3)
    if not d.ncon:
        return pads, table
    c = d.contact
    g1, g2, address = c.geom[:, 0], c.geom[:, 1], c.efc_address
    j = np.arange(3)
    valid = (address[:, None] >= 0) & (j < c.dim[:, None])
    local = np.where(valid, d.efc_force[np.where(valid, address[:, None] + j, 0)], 0.)
    on2 = np.einsum('nki,nk->ni', c.frame.reshape(-1, 3, 3), local)   # force on geom2 from geom1
    for k, p in enumerate(sim.pad_names):
        pads[p] -= on2[(pad[g1] == k) & content[g2]].sum(0)
        pads[p] += on2[(pad[g2] == k) & content[g1]].sum(0)
    table += on2[(g1 == sim.table_geom) & content[g2]].sum(0)
    table -= on2[(g2 == sim.table_geom) & content[g1]].sum(0)
    return pads, table

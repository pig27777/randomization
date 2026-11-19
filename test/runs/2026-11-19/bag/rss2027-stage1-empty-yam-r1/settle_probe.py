"""Isolate native contact peaks during the empty bag's initial settle."""

import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parents[4]))
from bag_sim.physical_parcel import Config, Parcel, parcel_profile  # noqa: E402


def settle(dt):
    p = Parcel(Config(mode="empty", geometry="geometry/kraft_edge_sealed_35.npz",
                      payload_mass=0., timestep=dt, **parcel_profile("pe")["shell"]))
    peaks = {}
    support = p.m.geom("table").id
    finite = True
    for _ in range(round(.30 / dt)):
        p.step()
        distances = np.where(p.d.contact.efc_address >= 0, p.d.contact.dist, 0.)
        ground = (p.d.contact.geom == support).any(axis=1)
        for name, selected in (("support", ground), ("self", ~ground)):
            depth = max(0., -float(distances[selected].min(initial=0.)))
            if depth > peaks.get(name, {}).get("depth_m", 0.):
                idx = np.flatnonzero(selected)[np.argmin(distances[selected])]
                c = p.d.contact[idx]
                peaks[name] = dict(depth_m=depth, time_s=p.d.time - dt, geom=c.geom.tolist(),
                                   element=c.elem.tolist(), xyz_m=c.pos.tolist())
        finite = bool(np.isfinite(p.d.qpos).all() and np.isfinite(p.d.qvel).all())
        if p.d.warning.number.sum() or not finite:
            break
    return dict(dt_s=dt, simulated_s=p.d.time, peaks=peaks, warnings=p.d.warning.number.tolist(),
                passed=bool(finite and p.d.time > .299 and all(x["depth_m"] <= .001 for x in peaks.values())
                            and not p.d.warning.number.sum()))


if __name__ == "__main__":
    results = []
    for step_us in (100., 50., 25.):
        result = settle(step_us * 1e-6)
        print(json.dumps(result), flush=True)
        results.append(result)
    (RUN / "settle-probe.json").write_text(json.dumps(results, indent=2) + "\n")

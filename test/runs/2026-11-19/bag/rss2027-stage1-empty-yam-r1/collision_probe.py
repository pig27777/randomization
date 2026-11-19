"""Compare the same empty bag with the two built-in convex collision paths."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parents[4]))
from bag_sim.physical_parcel import Config, Parcel, parcel_profile  # noqa: E402


def simulate(path, duration, dt, tolerance_nm=None):
    parcel = Parcel(Config(mode="empty", geometry="geometry/kraft_edge_sealed_35.npz",
                           payload_mass=0., timestep=dt, **parcel_profile("pe")["shell"]))
    model, data = parcel.m, parcel.d
    if path == "mpr":
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    if tolerance_nm is not None:
        model.opt.ccd_tolerance = tolerance_nm * 1e-9
    faces = parcel.envelope["faces"]
    peak = 0.
    self_peak = None
    finite = True
    for _ in range(round(duration / dt)):
        parcel.step()
        active = data.contact.efc_address >= 0
        depth = np.where(active, -data.contact.dist, 0.)
        peak = max(peak, float(depth.max(initial=0.)))
        candidates = np.where((data.contact.flex >= 0).all(axis=1), depth, 0.)
        if candidates.max(initial=0.) > (self_peak or {}).get("depth_m", 0.):
            contact = data.contact[int(candidates.argmax())]
            z = data.flexvert_xpos[faces[contact.elem], 2]
            z_upper_bound = min(z[0].max() - z[1].min(), z[1].max() - z[0].min()) + 2 * model.flex_radius[0]
            self_peak = dict(time_s=data.time - dt, depth_m=-contact.dist,
                             elems=contact.elem.tolist(), triangle_z_m=z.tolist(),
                             vertical_separation_upper_bound_m=max(0., float(z_upper_bound)))
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        if data.warning.number.sum() or not finite:
            break
    return dict(path=path, ccd_tolerance_m=model.opt.ccd_tolerance, dt_s=dt, simulated_s=data.time, peak_penetration_m=peak,
                self_peak=self_peak, warnings=data.warning.number.tolist(),
                passed=bool(finite and data.time >= duration - dt and peak <= .001 and not data.warning.number.sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=.060)
    parser.add_argument("--dt", type=float, default=.0001)
    parser.add_argument("--paths", nargs="+", choices=("native", "mpr"), default=["native", "mpr"])
    parser.add_argument("--tolerances-nm", type=float, nargs="+")
    args = parser.parse_args()
    results = []
    cases = [(path, tol) for path in args.paths for tol in args.tolerances_nm or [None]]
    for path, tol in cases:
        row = simulate(path, args.duration, args.dt, tol)
        print(json.dumps(row), flush=True)
        results.append(row)
    with args.output.open("x") as stream:
        stream.write(json.dumps(results, indent=2) + "\n")

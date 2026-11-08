"""L4 construction check for mixed segment graphs and explicit homogeneous buckets."""
import argparse
import json
from pathlib import Path
import warnings

from newton.solvers import SolverMuJoCo
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
import warp as wp

from segment_adapter import add_content


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--descriptions", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error("output already exists")
wp.config.enable_backward = False
cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(integrator="implicitfast"))
builder = NewtonMJWarpManager.create_builder(physics_cfg=cfg, gravity=(0, 0, -9.81))
inputs = []
for path in args.descriptions:
    desc = json.loads(path.read_text())["description"]
    builder.begin_world()
    mapping = add_content(builder, desc, "proxy")
    builder.end_world()
    inputs.append({"category": desc["kappa"]["category"], "segments": len(desc["segments"]),
                   "edges": len(desc["joints"]), "proxy_bodies": mapping["proxy_bodies"]})
error = None
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model = builder.finalize(device="cuda:0")
    try:
        solver = SolverMuJoCo(model, integrator="implicitfast", iterations=100, ls_iterations=20,
                              tolerance=1e-8, njmax=4096, nconmax=4096)
    except (ValueError, RuntimeError, AssertionError) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
result = {"worlds": inputs, "construction_succeeded": error is None, "error": error,
          "warnings": [str(w.message) for w in caught], "dynamics_run": False}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result), flush=True)
if error is not None:
    raise SystemExit(2)

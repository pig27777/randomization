import argparse
import json
import runpy

import torch
from isaaclab.app import add_launcher_args, launch_simulation

from src.isaac_yam_parcel.yam_parcel_base_env import YAMPickEnv, yam_parcel_cfg
from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer

run = runpy.run_path("test/isaaclab/check_route_b_closed_loop.py")
rows = [("C4", "top", z, 0., 0.) for z in (-.003, -.0045, -.006, -.008)]
rows += [("C5", "top", z, 0., 4.) for z in (-.024, -.028, -.032, -.036, -.040)]
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", required=True)
parser.add_argument("--result", required=True)
parser.add_argument("--c4-offsets-mm", nargs="+", type=float)
add_launcher_args(parser)
parser.set_defaults(visualizer=[], livestream=0)
args = parser.parse_args()
if args.c4_offsets_mm:
    rows = [("C4", "top", -.001 * z, 0., 0.) for z in args.c4_offsets_mm]
cfg = yam_parcel_cfg(len(rows))
with launch_simulation(cfg.sim, args):
    dr = DomainRandomizer("cuda:0", 17)
    fixture = CornerSuite(dr).select(dr.tensor([int(row[0][1:]) - 1 for row in rows], dtype=torch.long))
    env = YAMPickEnv(cfg, episode=fixture)
    try:
        result, _ = run["acquisition"](env, rows, 12. if args.c4_offsets_mm else 6.5)
        with open(args.result, "w") as file:
            json.dump(result, file, indent=2)
        print("ROUTE_B_TOP_PRESSURE " + json.dumps(result), flush=True)
    finally:
        env.close()

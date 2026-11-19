"""Repeat the refined-bag trial with reduced downward force late in closure."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-top-local-mesh-r1")))
from refine_parcel import backend, main as record  # noqa: E402
from yam_platform import protocol  # noqa: E402

GAP_M = .011
RELIEF_N = .05


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    scene = backend.scene
    robot = None

    def record_scene(cfg):
        nonlocal robot
        robot = scene(cfg)
        return robot

    class RelievedPress(protocol.NormalPress):
        def __init__(self, target_N, surface_z):
            super().__init__(target_N, surface_z)
            self.relieved = False

        def move(self, z, tip_z, force_N, support_N, dt, approach):
            if not approach and not self.relieved and robot.gap() <= GAP_M:
                self.relieved = True
                transition = dict(time_s=float(robot.sim.d.time), gap_m=float(robot.gap()),
                                  actual_downforce_N=force_N, target_before_N=self.target_N, target_after_N=RELIEF_N)
                with (args.output / "closure-relief.json").open("x") as stream:
                    stream.write(json.dumps(transition, indent=2) + "\n")
                print(json.dumps(transition), flush=True)
            if not self.relieved:
                return super().move(z, tip_z, force_N, support_N, dt, approach)
            requested = self.target_N
            self.target_N = RELIEF_N
            try:
                return super().move(z, tip_z, force_N, support_N, dt, approach)
            finally:
                self.target_N = requested

    with patch.object(backend, "scene", record_scene), patch.object(protocol, "NormalPress", RelievedPress):
        record()
    (args.output / "closure_controller_source.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()

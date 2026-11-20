"""Probe one rounded-pad radius of additional insertion during actual closure."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

from record_faster_approach import protocol, record
from record_ridge import backend
from record_seated_close import SeatedClose


class DeeperYam(protocol.Yam):
    feed_m = 0.

    def command(self, position, gap_command):
        if gap_command < .012:
            feedback = backend.measure(self)
            down = sum(force[2] for force in feedback["force_on_fingers_N"])
            support = feedback["direct_support_N"]
            dt = self.sim.m.opt.timestep
            if support > .05 or down > 2.5:
                self.feed_m = max(0., self.feed_m - .006 * dt)
            elif gap_command > 0.:
                self.feed_m = min(.002, self.feed_m + .006 * dt)
        if self.feed_m:
            position = position.copy()
            position[2] -= self.feed_m
        super().command(position, gap_command)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument("--output", type=Path, required=True)
    args, _ = cli.parse_known_args()
    sys.argv.append("--record-capture-forces")
    with patch.object(protocol, "Yam", DeeperYam), patch.object(protocol, "NormalPress", SeatedClose):
        record()
    (args.output / "depth_controller_source.py").write_bytes(Path(__file__).read_bytes())

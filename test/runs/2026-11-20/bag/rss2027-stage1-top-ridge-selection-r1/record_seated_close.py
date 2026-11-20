"""Hold the attained top-grasp height through symmetric YAM jaw closure."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

from record_faster_approach import FasterApproach, protocol, record


class SeatedClose(FasterApproach):
    def move(self, z, tip_z, force_N, support_N, dt, approach):
        position = super().move(z, tip_z, force_N, support_N, dt, approach)
        if not approach and support_N <= self.support_limit_N and force_N <= 2.5:
            return z
        return position


if __name__ == "__main__":
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument("--output", type=Path, required=True)
    args, _ = cli.parse_known_args()
    sys.argv.append("--record-capture-forces")
    with patch.object(protocol, "NormalPress", SeatedClose):
        record()
    (args.output / "seat_controller_source.py").write_bytes(Path(__file__).read_bytes())

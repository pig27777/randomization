"""Increase only the post-contact approach gain in the shared YAM controller."""

import argparse
from pathlib import Path
from unittest.mock import patch

from record_ridge import main as record  # noqa: E402
from yam_platform import protocol  # noqa: E402


class FasterApproach(protocol.NormalPress):
    def move(self, z, tip_z, force_N, support_N, dt, approach):
        position = super().move(z, tip_z, force_N, support_N, dt, approach)
        if approach and .05 < force_N < .8 * self.target_N and position < z:
            return z + max(4. * (position - z), -.005 * dt)
        return position


if __name__ == "__main__":
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument("--output", type=Path, required=True)
    args, _ = cli.parse_known_args()
    with patch.object(protocol, "NormalPress", FasterApproach):
        record()
    (args.output / "approach_controller_source.py").write_bytes(Path(__file__).read_bytes())

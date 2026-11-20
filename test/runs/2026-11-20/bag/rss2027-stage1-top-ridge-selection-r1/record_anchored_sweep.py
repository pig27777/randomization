"""Close the existing YAM jaws with a lateral shift that anchors the right pad."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

from record_faster_approach import FasterApproach, protocol, record


class AnchoredYam(protocol.Yam):
    def command(self, position, gap_command):
        initial = self.sim.built.cfg.grasp_opening_m
        shift = min(.010, .5 * max(0., initial - gap_command))
        if shift > 0.:
            # Grasp-site +y is opposite to the direction from the left to the right pad.
            position = position - shift * self.rotation[:, 1]
        super().command(position, gap_command)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument("--output", type=Path, required=True)
    args, _ = cli.parse_known_args()
    sys.argv.append("--record-capture-forces")
    with patch.object(protocol, "Yam", AnchoredYam), patch.object(protocol, "NormalPress", FasterApproach):
        record()
    (args.output / "sweep_controller_source.py").write_bytes(Path(__file__).read_bytes())

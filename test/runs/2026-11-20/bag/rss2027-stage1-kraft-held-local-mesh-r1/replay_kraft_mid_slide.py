"""Measure signed withdrawal in the archived Kraft sliding window."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

from replay_settled import RUN, replay, settled_scene

sys.path.insert(0, str(RUN.with_name("rss2027-stage1-pe-held-load-step-r1")))
from replay_local_tangent import replay_window  # noqa: E402

SOURCE = RUN / "settled-recording/dt-50us"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    native = json.loads(SOURCE.with_suffix(".json").read_bytes())
    onset = native["sliding_onset"]
    ratio = onset["tension_N"] / (onset["left_inner_N"] + onset["right_inner_N"])
    with patch.object(replay, "held_scene", settled_scene):
        replay_window(args.output, native, SOURCE.with_suffix(".npz"), .750, .768, ratio)

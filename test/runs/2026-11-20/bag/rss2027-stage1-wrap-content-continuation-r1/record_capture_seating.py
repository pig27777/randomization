"""Record native YAM capture as a function of measured side-entry travel."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np

from select_side_flap import record, shifted_target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    args, _ = parser.parse_known_args()
    selected = json.loads((args.config.parent / "selection.json").read_bytes())["selected"]
    advance = selected["close_advance_m"]
    placement = shifted_target(np.zeros(3), advance) if advance is not None else nullcontext()
    with placement:
        record()

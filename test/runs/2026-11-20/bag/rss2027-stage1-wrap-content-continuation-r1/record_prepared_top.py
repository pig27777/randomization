"""Drive the standard top approach from a geometrically prepared package."""

import argparse
import json
from pathlib import Path

from record_local_top import record_with_braking_height


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args, _ = parser.parse_known_args()
    companion = args.config.with_name("press-braking-height.json") if args.config else None
    if companion and companion.is_file():
        record_with_braking_height(json.loads(companion.read_bytes()))
        (args.output / companion.name).write_bytes(companion.read_bytes())
    else:
        record_with_braking_height()

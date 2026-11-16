"""Render the 2D parcel with the shared YAM cameras and a selected FFmpeg.

The shared generic caption names a 3D flex. This 2D-only entry omits that caption
and also works with the installed FFmpeg build lacking the drawtext filter.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experience"))
from yam_platform import render_video  # noqa: E402


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--view", choices=("both", "detail"), default="both")
    args = parser.parse_args()
    encoder = args.ffmpeg.resolve(strict=True)
    original = render_video.subprocess.Popen

    def encode(command, *positional, **keywords):
        command = list(command)
        if command[0] == "ffmpeg":
            command[0] = str(encoder)
            index = command.index("-vf")
            del command[index:index + 2]
        return original(command, *positional, **keywords)

    with patch.object(render_video.subprocess, "Popen", encode):
        render_video.render(args.run, args.output, fps=20, view=args.view)
    metadata = args.output.with_name(args.output.stem + "_render.json")
    result = json.loads(metadata.read_text())
    result.update(encoder=str(encoder), representation="existing_two_sheet_2d_flex",
                  captions="omitted", native_contact_only=True)
    metadata.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

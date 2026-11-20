"""Recover the real bubble-bag state before testing a fixed actuator hold."""

from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-kraft-held-local-mesh-r1")))
from replay_refined import replay  # noqa: E402
from settle_refined import main as relax  # noqa: E402


if __name__ == "__main__":
    replay.SOURCE = RUN.with_name("rss2027-stage1-material-load-ramps-r1") / "bubble-early-withdrawal/recording"
    replay.CHECKPOINT = RUN / "held-state.npz"
    if sys.argv[1:] == ["--relax"]:
        relax(RUN, (.0002, .0001), replay.held_scene)
    else:
        replay.main()

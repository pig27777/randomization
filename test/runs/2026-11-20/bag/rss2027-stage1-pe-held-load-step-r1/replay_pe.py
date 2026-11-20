"""Replay the accepted PE suspension with the shared held-load measurement."""

from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-kraft-held-load-step-r1")))
import replay_held_load as replay  # noqa: E402


if __name__ == "__main__":
    replay.SOURCE = RUN.with_name("rss2027-stage1-material-load-ramps-r1") / "pe/recording"
    replay.CHECKPOINT = RUN / "held-state.npz"
    replay.main()

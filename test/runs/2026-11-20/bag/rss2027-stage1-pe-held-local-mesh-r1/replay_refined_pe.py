"""Use the verified jaw-local remeshing on the accepted PE checkpoint."""

from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-kraft-held-local-mesh-r1")))
import replay_refined as experiment  # noqa: E402


def configure():
    experiment.RUN = RUN
    experiment.ASSET = RUN / "prepared_mesh.npz"
    experiment.GEOMETRY = "../" + experiment.ASSET.relative_to(experiment.ROOT).as_posix()
    experiment.COMPILED_DT = .000025
    experiment.replay.SOURCE = RUN.with_name("rss2027-stage1-material-load-ramps-r1") / "pe/recording"
    experiment.replay.CHECKPOINT = RUN.with_name("rss2027-stage1-pe-held-load-step-r1") / "held-state.npz"


if __name__ == "__main__":
    configure()
    if sys.argv[1:] == ["--relax"]:
        from settle_refined import main as relax
        relax(RUN, (.000025, .0000125))
    else:
        experiment.main()

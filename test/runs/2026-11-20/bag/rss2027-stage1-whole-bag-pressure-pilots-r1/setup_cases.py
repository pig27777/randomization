"""Use the measured materials and prepared whole bags in the top-pressure protocol."""

import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
BAGS = (
    ("kraft", "rss2027-stage1-kraft-release-r2/recording"),
    ("bubble", "rss2027-stage1-bubble-free-release-floor-r1/recording"),
    ("paperboard", "rss2027-stage1-material-load-ramps-r1/paperboard-rounded-pad/recording"),
)


def main():
    for name, original in BAGS:
        source = RUN.parent / original / "trial/result.json"
        recorded = json.loads(source.read_bytes())
        cfg = recorded["config"]
        if cfg["bag_profile"] != name or not (RUN.parents[4] / cfg["initial_state"]).is_file():
            raise ValueError(f"Missing matching prepared whole bag for {name}")
        params = cfg | dict(grasp_face="top", yaw=0., tilt=0., press_target_N=.5, press_start_offset_m=.013,
                            press_timeout_s=12., press_only=False, simulation_seconds=25., wall_limit=1500.,
                            pickup_only=True, pickup_release_seconds=0., close_gap_hold_s=.04, smoke_seconds=0.)
        case = RUN / name
        case.mkdir(exist_ok=False)
        for filename, data in (("config.json", params), ("smoke.json", params | {"smoke_seconds": .02})):
            (case / filename).write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()

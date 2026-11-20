"""Choose separate measured bag and contents dimensions for the 0.5 N runs."""

import csv
import json
import math
import re
import sys

from screen_slack_geometries import DESIGN, RUN
from setup_cases import write_inputs

sys.path.insert(0, str(RUN.parents[3] / "bag_model"))
import physical_parcel_yam as backend  # noqa: E402


def fit_timestep(params):
    try:
        return backend.parcel_world(backend.Config(**params))[0]
    except ValueError as error:
        limit = re.fullmatch(r"timestep does not resolve explicit film bending; use <= (\S+) s", str(error))
        if limit is None:
            raise
    upper = float(limit[1])
    params["dt"] /= 2 ** max(1, math.ceil(math.log2(params["dt"] / upper)))
    return backend.parcel_world(backend.Config(**params))[0]


def main():
    inputs = json.loads((RUN / "slack-screen.json").read_bytes())["geometries"]
    with DESIGN.open(newline="") as stream:
        design = list(csv.DictReader(stream))
    output = RUN / "slack-series"
    output.mkdir(exist_ok=False)
    prepared = []
    for material in ("bubble", "paperboard", "kraft"):
        rated = sorted((row for row in inputs[material] if row["status"] == "enclosed"),
                       key=lambda row: row["slack_area_upper_m2"])
        candidates = rated[:1] + (rated[-1:] if len(rated) >= 2 else [])
        original = json.loads((RUN / material / "config.json").read_bytes())
        for row in candidates:
            name = f"{material}-{row['sample']:03d}"
            params = original | dict(initial_state=None, bag_size=row["cavity_m"], payload_size=row["payload_m"],
                                     payload_offset_xy=row["payload_offset_m"], support_height=0.,
                                     press_timeout_s=18., simulation_seconds=27.)
            parcel = fit_timestep(params)
            geometry = parcel.envelope["metadata"]["preparation"]
            case = output / name
            write_inputs(case, params)
            (case / "settle.json").write_text(json.dumps(params | {"smoke_seconds": .30}, indent=2) + "\n")
            record = dict(source_geometry=design[row["sample"] - 1], screen=row, preparation=geometry, dt_s=params["dt"])
            (case / "inputs.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            prepared.append(name)
    (output / "cases.json").write_text(json.dumps(prepared, indent=2) + "\n")
    print(" ".join(prepared), flush=True)


if __name__ == "__main__":
    main()

"""Use the shared YAM recorder at a pose selected from the existing shell."""

import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-top-local-mesh-r1")))
import refine_parcel  # noqa: E402
from refine_parcel import backend  # noqa: E402


def main():
    selected = json.loads((RUN / "ridge-selection.json").read_bytes())["strongest"][0]
    center_at = np.asarray(selected["center_xy_m"])
    attach = backend.attach_robot

    def at_ridge(cfg, built, sim_factory, center, _top, size, mass, approach=None):
        if approach is not None or cfg.grasp_face != "top" or cfg.yaw != selected["yaw_deg"]:
            raise ValueError("The recorder requires the selected top pose")
        np.testing.assert_allclose(center_at - center, selected["xy_offset_m"], atol=1e-12, rtol=0.)
        tip_height = selected["first_tip_height_m"] + built.parcel.cfg.radius
        return attach(cfg, built, sim_factory, center_at, tip_height, size, mass)

    with patch.object(backend, "attach_robot", at_ridge):
        refine_parcel.main()


if __name__ == "__main__":
    main()

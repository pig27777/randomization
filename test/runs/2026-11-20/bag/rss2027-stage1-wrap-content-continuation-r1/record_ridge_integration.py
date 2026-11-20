"""Reproduce the selected ridge action while keeping its native integration states."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

from record_local_top import FasterApproach, INTEGRATION_SPEC, RUN, backend, integration_state, protocol
from record_ridge import main as record


def verify(case):
    base = RUN.with_name("rss2027-stage1-top-ridge-selection-r1")
    original = base / "faster-approach"
    trial = case / "recording/trial"
    source = json.loads((original / "recording/trial/trace.json").read_bytes())
    rows = json.loads((trial / "trace.json").read_bytes())
    if rows != source[:len(rows)]:
        raise ValueError("The repeated ridge states differ before the requested cutoff")
    result = json.loads((trial / "result.json").read_bytes())
    if (result["warnings"] or result["peaks"]["penetration_mm"] > 1.
            or any(row["payload_geometry"]["intersecting_triangles"] or row["payload_geometry"]["outside_corners"]
                   for row in rows)):
        raise ValueError("The repeated ridge window violates the original geometry or numerical limit")
    comparison = json.loads((original / "physical-comparison.json").read_bytes())["faster-approach"]
    fit = comparison["fold_window"]
    with (np.load(trial / "trajectory.npz") as archive,
          np.load(original / "recording/trial/trajectory.npz") as geometry):
        time = archive["time"]
        states = archive["integration_state"]
        np.testing.assert_array_equal(time, [row["t"] for row in rows])
        np.testing.assert_array_equal(archive["qpos"], geometry["qpos"][:len(rows)])
        indices = np.flatnonzero((time >= fit["t_start_s"]) & (time <= fit["t_last_s"]))
        if (len(states) != len(time) or len(indices) != fit["samples"]
                or int(archive["integration_state_spec"]) != INTEGRATION_SPEC):
            raise ValueError("The native replay did not record every original gathering state")
    selected = json.loads((base / "ridge-selection.json").read_bytes())["strongest"][0]
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    height = dict(target_center_xy_m=selected["center_xy_m"],
                  top_reference_m=float(selected["first_tip_height_m"] + model.flex_radius[0]))
    comparison = dict(fold_window=fit, identical_samples=len(rows), saved_native_states=len(states),
                      source_reference=original.relative_to(RUN.parents[4]).as_posix())
    for name, value in (("height.json", height), ("physical-comparison.json", comparison)):
        with (case / name).open("x") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--verify"]:
        parser = argparse.ArgumentParser(__doc__)
        parser.add_argument("--verify", type=Path, required=True)
        verify(parser.parse_args().verify.resolve())
    else:
        with patch.object(protocol, "NormalPress", FasterApproach), \
                patch.multiple(backend, create=True, STATE_SPEC=INTEGRATION_SPEC, record_state=integration_state):
            record()

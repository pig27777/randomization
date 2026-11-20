"""Exercise a prepared bend in the original closed whole-bag film."""

import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[4]
CASE = RUN / "aligned"
sys.path[:0] = [str(ROOT / "test/bag_model"), str(RUN.with_name("rss2027-stage1-top-ridge-selection-r1"))]
import physical_parcel_yam as backend  # noqa: E402
import record_physical_parcel as recorder  # noqa: E402
from record_faster_approach import FasterApproach, protocol  # noqa: E402

ANGLE_DEG = 70.
CREASE_HALF_LENGTH_M = .035


def prepare():
    source = ROOT / "bag_sim/geometry/kraft_edge_sealed_35.npz"
    with np.load(source) as archive:
        asset = dict(archive.items())
    flat = asset["vertices"]
    assert np.all(flat[:, 2] == 0.)
    folded = flat.copy()
    length = -flat[:, 1]
    distance = abs(length)
    angle = np.deg2rad(ANGLE_DEG)
    inside = np.minimum(distance, CREASE_HALF_LENGTH_M)
    outside = distance - inside
    folded[:, 1] = -np.sign(length) * (outside + inside * np.cos(angle))
    folded[:, 2] = (CREASE_HALF_LENGTH_M - inside) * np.sin(angle)
    # Rotate the full bag so the original reachable top wrist can cross the fold.
    folded[:, :2] = folded[:, [1, 0]] * [-1., 1.]
    a, b = np.unique(np.sort(np.stack((asset["faces"], np.roll(asset["faces"], -1, axis=1)), axis=-1)
                             .reshape(-1, 2), axis=1), axis=0).T
    before = np.linalg.norm(flat[a] - flat[b], axis=1)
    after = np.linalg.norm(folded[a] - folded[b], axis=1)
    np.testing.assert_allclose(after, before, atol=1e-12, rtol=0.)
    CASE.mkdir(exist_ok=False)
    mesh = CASE / "creased_asset.npz"
    with mesh.open("xb") as stream:
        np.savez_compressed(stream, **(asset | {"vertices": folded}))
    initial = ROOT / "test/runs/2026-11-19/bag/rss2027-stage1-empty-yam-r4/recording/trial/config.json"
    cfg = json.loads(initial.read_bytes()) | dict(geometry=f"../{mesh.relative_to(ROOT).as_posix()}", yaw=0.,
                                                   dt=.0001, close_gap_hold_s=.04, pickup_only=True,
                                                   press_timeout_s=16., simulation_seconds=23., wall_limit=1400.)
    for name, seconds in (("config", 0.), ("smoke", .02), ("settle", .30)):
        (CASE / f"{name}.json").write_text(json.dumps(cfg | {"smoke_seconds": seconds}, indent=2) + "\n")
    report = dict(source=source.relative_to(ROOT).as_posix(), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  vertices=len(flat), faces=len(asset["faces"]), angle_deg=ANGLE_DEG, crease_height_m=float(folded[:, 2].max()),
                  projected_fold_width_m=2. * CREASE_HALF_LENGTH_M * np.cos(angle), bag_rotation_deg=90.,
                  maximum_edge_length_error_m=float(np.max(abs(after - before))))
    (CASE / "preparation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def record():
    attach = backend.attach_robot

    def at_flanks(cfg, built, factory, center, _top, size, mass, approach=None):
        if cfg.mode != "empty" or approach is not None:
            raise ValueError("The creased validation uses an empty bag and a top approach")
        points = built.parcel.envelope["current"]
        flank = points[abs(points[:, 1] - center[1]) >= CREASE_HALF_LENGTH_M]
        top = float(flank[:, 2].max() + built.parcel.cfg.radius)
        return attach(cfg, built, factory, center, top, size, mass)

    with patch.object(backend, "attach_robot", at_flanks), patch.object(protocol, "NormalPress", FasterApproach):
        recorder.main()


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare"]:
        prepare()
    else:
        record()

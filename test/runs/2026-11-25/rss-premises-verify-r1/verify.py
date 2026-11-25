"""Check the archived Stage 1 evidence for consistency."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def invalid(token):
    raise ValueError(f"Nonfinite JSON token {token}")


def read_json(path):
    return json.loads(path.read_text(), parse_constant=invalid)


def main():
    for path in sorted(ROOT.glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
    print("Python AST passed")

    audit = read_json(ROOT / "stage1_audit.json")["trials"]
    assert len(audit) == 20, len(audit)
    for summary in audit:
        rows = read_json(ROOT / summary["data"])["rows"]
        with np.load(ROOT / summary["frames"]) as frames:
            assert frames["film"].shape[1:] == (952, 3)
            assert frames["faces"].shape == (1900, 3)
            assert np.isfinite(frames["film"]).all() and np.isfinite(frames["bodies"]).all()
            assert frames["steps"][-1] == len(rows) - 1
        assert abs(summary["film_mass_kg"] + summary["payload_mass_kg"] - summary["whole_bag_mass_kg"]) < 1.e-12
        assert abs(rows[-1]["time_s"] - summary["observed_end_s"]) < 1.e-12
        assert summary["valid_steps"] == len(rows) - 1
    print("20 complete NPZ/JSON records matched, including rejected startup runs")

    paths = ("joint-load/1C-empty", "joint-load-retry/1C-500", "joint-load-retry/1C-1000")
    targets = []
    for path in paths:
        with np.load(ROOT / f"{path}.npz") as data:
            targets.append(data["q_targets"].copy())
    assert all(q.shape == (116, 8) and np.array_equal(q, targets[0]) for q in targets)
    print("Identical Gate 1C targets SHA-256", hashlib.sha256(targets[0].tobytes()).hexdigest())

    by_path = {r["data"]: r for r in audit}
    film = by_path["grasp-reset/1A.json"]
    assert abs(film["payload_mass_kg"] - .300) < 1.e-12 and abs(film["film_mass_kg"] - .0398138181) < 1.e-9
    assert film["max_core_clearance_m"] < .001 and film["max_edge_ratio"] < 1.05
    assert film["max_intersecting_triangles"] == 0 and film["max_outside_corners"] == 0
    assert by_path["grasp-reset/1B.json"]["max_loaded_arm_core"] == 2
    continuous = by_path["through-approach/1B.json"]
    assert continuous["max_intersecting_triangles"] == 2 and continuous["max_loaded_arm_core"] == 0
    assert continuous["hold_end_s"] is None
    assert by_path[f"{paths[0]}.json"]["max_film_clearance_m"] < .001
    assert all(by_path[f"{path}.json"]["hold_end_s"] is None for path in paths[1:])
    print("Original-mass negative gates and reported masses checked against raw traces")

    comparisons = {
        "film-10g/1A": "grasp-reset/1A",
        "through-10g/1B": "through-approach/1B",
        "joint-10g/1C-empty": "joint-load/1C-empty",
        "joint-10g/1C-500": "joint-load-retry/1C-500",
        "joint-10g/1C-1000": "joint-load-retry/1C-1000",
    }
    for current, original in comparisons.items():
        row = by_path[f"{current}.json"]
        assert row["specified_film_mass_g"] == 10.
        assert abs(row["film_mass_kg"] - .010) < 1.e-7
        with np.load(ROOT / f"{current}.npz") as new, np.load(ROOT / f"{original}.npz") as old:
            assert np.array_equal(new["q_targets"], old["q_targets"])
            assert np.array_equal(new["film"][0], old["film"][0])
            assert np.array_equal(new["faces"], old["faces"])
        print(current, "matched prior geometry and targets, and measured particle mass 10 g")
    corrected = [p for p in comparisons if p.startswith("joint-10g/")]
    for p in corrected:
        with np.load(ROOT / f"{p}.npz") as data:
            assert np.array_equal(data["q_targets"], targets[0])
    print("Corrected Gate 1C retains the identical 116 × 8 target array")

    corrected_film = by_path["film-10g/1A.json"]
    assert corrected_film["max_core_clearance_m"] < .001 and corrected_film["max_edge_ratio"] <= 1.05
    assert corrected_film["max_intersecting_triangles"] == corrected_film["max_outside_corners"] == 0
    corrected_clamp = by_path["through-10g/1B.json"]
    assert corrected_clamp["max_intersecting_triangles"] == 2 and corrected_clamp["hold_end_s"] is None
    assert corrected_clamp["max_loaded_arm_core"] == corrected_clamp["max_loaded_arm_table"] == 0
    assert by_path["joint-10g/1C-empty.json"]["max_film_clearance_m"] < .001
    assert all(by_path[f"{p}.json"]["hold_end_s"] is None for p in corrected[1:])
    print("10 g negative gates corroborated from measured data")

    images = sorted(ROOT.glob("fig_stage1*.png"))
    assert len(images) == 2
    for path in images:
        with Image.open(path) as im:
            assert im.width >= 2400 and im.height >= 1000
            size = im.size
            im.verify()
        print(path.name, size, "PNG passed")


if __name__ == "__main__":
    main()

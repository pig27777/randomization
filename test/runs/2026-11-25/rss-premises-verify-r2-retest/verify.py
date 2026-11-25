"""Validate route integrity, replay identity, and exported artifacts."""
import ast
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
R1 = ROOT.parent / "rss-premises-verify-r1"


def reject_nonfinite(token):
    raise ValueError(token)


def read(path):
    return json.loads(path.read_text(), parse_constant=reject_nonfinite)


def main():
    for p in ROOT.glob("*.py"):
        ast.parse(p.read_text(), filename=str(p))
    print("Python AST passed")
    d = read(ROOT / "retest_summary.json")
    assert d["verdict"] == "KILL"
    cases = {
        "1B-600": ("through-10g/1B.npz", 600.),
        "1B-200": ("through-10g/1B.npz", 200.),
        "1C-empty": ("joint-10g/1C-empty.npz", 0.),
        "1C-150": ("joint-10g/1C-500.npz", 150.),
        "1C-250": ("joint-10g/1C-1000.npz", 250.),
    }
    shared = None
    for case, (original, mass) in cases.items():
        metrics = d["newton"][case]
        assert metrics["payload_mass_g"] == mass
        assert abs(metrics["film_mass_g"] - 10.) < 1.e-6
        path = ROOT / metrics["source"]
        trial = read(path)
        rows = trial["rows"]
        assert abs(rows[-1]["t_s"] - metrics["end_s"]) < 1.e-9
        assert metrics["audit_stride"] == 10 and trial["dt_s"] == .0005
        if trial["failure"] == "film–payload 2 mm geometry":
            reason = rows[-1]["geometry"]
            assert reason["inset_triangles"] > 0 or reason["outside_max_um"] > 2000.
        if trial["failure"] == "direct arm–payload load":
            assert rows[-1]["direct_core_loaded"] > 0
        raw_path = path.with_suffix(".npz")
        with np.load(raw_path) as raw, np.load(R1 / original) as prior:
            targets = raw["original_q_targets"]
            assert np.array_equal(targets, prior["q_targets"])
            assert sha256(targets.tobytes()).hexdigest() == metrics["original_target_sha256"]
            assert np.array_equal(targets, raw["q_targets"][:len(targets)])
            assert np.array_equal(raw["film"][0], prior["film"][0])
            assert np.array_equal(raw["faces"], prior["faces"])
            assert raw["steps"][-1] == trial["steps"]
            if not case.startswith("1B"):
                assert targets.shape == (116, 8)
                if shared is None:
                    shared = targets.copy()
                assert np.array_equal(shared, targets)
        print(case, "passed archive identity and raw trace validation")

    for case, result in d["flex_cpu_50us"].items():
        config = read(ROOT / "flex-cpu" / case / "config.json")
        assert config["dt_s"] == .00005 and config["engine"] == "4.0.1"
        assert result["status"] in {"PASS", "FAIL"}
        with np.load(ROOT / "flex-cpu" / case / "trace.npz") as data:
            assert len(data["time"]) > 1 and np.isfinite(data["time"]).all()
        print("Flex", case, result["status"], "50 μs trace readback passed")

    for name in ("fig_jaw_diagnosis.png", "fig_retest_keyframes.png", "fig_flex_reference.png"):
        with Image.open(ROOT / name) as img:
            assert img.width >= 2400
            img.verify()
        print(name, "verified")


if __name__ == "__main__":
    main()

"""Replay the archived stage-one captures with the YAM finger-body frame."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, "test/bag_model")
from rss2027_yam_capture_audit import audit  # noqa: E402


RUNS = Path("test/runs")
BASELINE = "5b2c9480986716b3b483395116ff6b7a7ee0fbe2"
RELEASE_ANALYZER = RUNS / "2026-11-20/bag/rss2027-stage1-whole-bag-release-r1/analyze_release.py"
APERTURE_FIELDS = {"aperture_faces", "aperture_area_m2", "aperture_reference_area_m2",
                   "aperture_material_length_m"}


def same_observations(old_path, new_path):
    with old_path.open() as old, new_path.open() as new:
        baseline = csv.DictReader(old)
        fields = set(baseline.fieldnames) - APERTURE_FIELDS
        previous = [{key: row[key] for key in fields} for row in baseline]
        corrected = [{key: row[key] for key in fields} for row in csv.DictReader(new)]
        return previous == corrected


def replay_release(trial, destination):
    recording = destination / "recording"
    recording.symlink_to(trial.parent.resolve(), target_is_directory=True)
    try:
        subprocess.run([sys.executable, "-B", RELEASE_ANALYZER, destination], check=True)
    finally:
        recording.unlink()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    run = parser.parse_args().output
    if (run / "index.json").exists():
        raise FileExistsError(f"Replay output already exists in {run}")
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASELINE, "--",
                                     RUNS / "2026-11-19", RUNS / "2026-11-20"], text=True).splitlines()
    records = {}
    for path in map(Path, paths):
        if path.name != "capture-summary.json" or "rss2027-stage1-" not in str(path):
            continue
        previous = json.loads(path.read_text())
        trial = Path(previous["trial"])
        if str(trial) not in records:
            parts = trial.relative_to(RUNS).parts[2:-1]
            case = parts[0].removeprefix("rss2027-stage1-")
            subcase = [part for part in parts[1:] if part != "recording"]
            destination = run.joinpath(case, *subcase)
            corrected = audit(trial, destination / "capture")
            if corrected["max_closing_axis_error"] > 1e-6:
                raise ValueError(f"Closing direction disagrees with the finger axes in {trial}")
            if json.loads((trial / "result.json").read_text())["config"].get("pickup_release_seconds", 0.):
                replay_release(trial, destination)
            records[str(trial)] = dict(corrected_directory=str(destination.relative_to(run)),
                                       sample_count=corrected["sample_count"],
                                       max_closing_axis_error=corrected["max_closing_axis_error"],
                                       compared_originals=[])
        record = records[str(trial)]
        destination = run / record["corrected_directory"]
        old_curve = path.with_name("capture-audit.csv")
        new_curve = destination / "capture/capture-audit.csv"
        if not same_observations(old_curve, new_curve):
            raise ValueError(f"Non-aperture data changed for {path}")
        record["compared_originals"].append(str(path.parent))
    (run / "index.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Replayed {len(records)} trajectories and checked "
          f"{sum(len(entry['compared_originals']) for entry in records.values())} archived curves")


if __name__ == "__main__":
    main()

"""Complete the remaining tests after the first guarded probe, under the caller's lock."""
import json
from pathlib import Path
import subprocess
import sys

from arm_retest import OFFSETS, PROFILES

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def run(script, *args, log):
    with (ROOT/log).open("w") as stream:
        subprocess.run([sys.executable, "-B", str(ROOT/script), *map(str, args)],
                       check=True, stdout=stream, stderr=subprocess.STDOUT)


def probes(profile, folder, modes):
    run("arm_retest.py", ROOT/folder, "--profile", profile, "--modes", *modes, log=f"{folder}.log")
    return [read(ROOT/folder/case/"summary.json") for case in modes]


def main():
    left = read(ROOT/"first-left/left/summary.json")
    first = [left]
    if left["gate_pass"]:
        first += probes("first", "first-others", ("center", "right"))
    qualified = all(case["gate_pass"] for case in first) and len(first) == len(OFFSETS)
    if not qualified:
        failing = [p for p in first if not p["gate_pass"]]
        second = "softer" if max(p["native_peak_N"] for p in failing) > 3. else "harder"
        print("Second profile", second, flush=True)
        trial = probes(second, f"{second}-probes", tuple(OFFSETS))
        if sum(p["gate_pass"] for p in trial) >= sum(p["gate_pass"] for p in first):
            selected = second
        else:
            selected = "first"
            if len(first) < len(OFFSETS):
                probes("first", "first-others", ("center", "right"))
    else:
        selected = "first"
    samples = [read(path) for case in ("a2", "a3", "a2-stiffer-film", "a3-stiffer-film")
               if (path := ROOT/case/"summary.json").is_file()]
    physical_winners = [row for row in samples if row["pass_hold"] and row["max_edge_ratio"] <= 1.05]
    if physical_winners:
        chosen = physical_winners[0]
    else:
        # A2 was also used by the archived Yam_B attempts, so it isolates the pad-contact change.
        chosen = dict(config="a2", young_scale=1.)
    tuning, stiffness = chosen["config"], chosen["young_scale"]
    selection = dict(profile=selected, tuning=tuning, young_scale=stiffness,
                     qualified_fixture_winner=bool(physical_winners))
    (ROOT/"selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(selection, flush=True)
    run("fixture_retest.py", "--film", tuning, "--young-scale", stiffness, "--contact", *PROFILES[selected],
        log="film-fixture.log")
    run("arm_retest.py", ROOT/"arms", "--profile", selected, "--tuning", tuning, "--young-scale", stiffness,
        "--modes", "through", "film", log="arms.log")


if __name__ == "__main__":
    main()

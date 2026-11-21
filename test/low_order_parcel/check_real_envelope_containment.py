"""Check hand-recorded UMI intervals against an independently generated DR envelope."""
import argparse
import csv
import json
import math
from pathlib import Path

FAILURE_COVERAGE = {
    "empty_grip": "CONTAINED",
    "one_finger_slip": "CONTAINED",
    "push_away": "CONTAINED",
    "payload_rollover": "MARGINAL",
    "electrostatic_adhesion": "OUT_OF_ENVELOPE",
    "corner_jam": "OUT_OF_ENVELOPE",
    "tearing": "OUT_OF_ENVELOPE",
}
PASS = "CONTAINED"
EDGE = "MARGINAL"
FAIL = "OUT_OF_ENVELOPE"
BOUND_FIELDS = ("real_min", "real_max", "uncertainty", "sim_dr_min", "sim_dr_max")
CASE_FIELDS = ("section", "id", "unit", *BOUND_FIELDS, "failure_mode", "reference_condition")


def measurement(row):
    numbers = []
    for field in BOUND_FIELDS:
        try:
            number = float(row[field] or ("0" if field == "uncertainty" else "nan"))
        except ValueError as exc:
            raise ValueError(f"{row['id']}: {field} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{row['id']}: enter finite real and matched simulated bounds")
        numbers.append(number)
    lo, hi, uncertainty, sim_lo, sim_hi = numbers
    if lo > hi or sim_lo > sim_hi or uncertainty < 0.:
        raise ValueError(f"{row['id']}: bounds must be ordered and uncertainty nonnegative")
    if row["unit"] == "0_or_1" and (any(x not in (0., 1.) for x in (lo, hi, sim_lo, sim_hi)) or uncertainty != 0.):
        raise ValueError(f"{row['id']}: binary observations use bounds 0 or 1 and zero uncertainty")
    if lo - uncertainty >= sim_lo and hi + uncertainty <= sim_hi:
        return PASS
    if lo + uncertainty < sim_lo or hi - uncertainty > sim_hi:
        return FAIL
    return EDGE


def evaluate(path):
    checks = []
    modes = dict.fromkeys(FAILURE_COVERAGE, 0)
    observed_trials, successes, pending = 0, 0, 0
    ids = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(CASE_FIELDS).issubset(reader.fieldnames):
            raise ValueError("use the column headings in configs/umi_containment_template.csv")
        for raw in reader:
            if any(raw[name] is None for name in CASE_FIELDS):
                raise ValueError("incomplete CSV row")
            row = {name: raw[name].strip() for name in CASE_FIELDS}
            key = row["id"]
            if not key or key in ids:
                raise ValueError(f"empty or duplicate check id: {key!r}")
            ids.add(key)
            if row["section"] == "failure":
                mode = row["failure_mode"]
                if not mode:
                    pending += 1
                    continue
                observed_trials += 1
                if mode == "success":
                    successes += 1
                    continue
                if mode not in FAILURE_COVERAGE:
                    raise ValueError(f"{key}: unknown failure mode {mode!r}")
                modes[mode] += 1
                status = FAILURE_COVERAGE[mode]
            elif row["section"] in ("pullout", "gather", "payload"):
                if not row["real_min"] and not row["real_max"]:
                    pending += 1
                    continue
                status = measurement(row)
            else:
                raise ValueError(f"{key}: unknown section {row['section']!r}")
            checks.append(dict(id=key, status=status))

    failures = sum(modes.values())
    covered = sum(n for mode, n in modes.items() if FAILURE_COVERAGE[mode] == PASS)
    missing = [mode for mode, n in modes.items() if n and FAILURE_COVERAGE[mode] == FAIL]
    partial = [mode for mode, n in modes.items() if n and FAILURE_COVERAGE[mode] == EDGE]
    return dict(checks=checks, observed_trials=observed_trials, pending_rows=pending, successes=successes,
                failures=failures, covered_failure_modes=covered, covered_fraction=covered / failures if failures else None,
                first_order_model_gap=bool(missing) if failures else None, missing_mechanisms=missing,
                partial_mechanisms=partial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args.csv), indent=2, ensure_ascii=False))
    except (OSError, ValueError) as error:
        parser.error(str(error))

"""Compare the whole-bag load and separation slopes using the existing native criteria."""

import json
from pathlib import Path

import numpy as np

from analyze_pe_parameters import fit_windows, intervals

RUN = Path(__file__).resolve().parent
DATA = RUN.with_name("rss2027-stage1-reference-curves-r1") / "stage1_whole_bag_data.json"


def measure(case, columns):
    evidence = case["measurement"]
    if not evidence["numerical_ok"] or evidence["load_start_s"] is None:
        reason = "numerical gate failed" if not evidence["numerical_ok"] else "no qualified load ramp"
        return dict(case=case["id"], material=case["material"], reason=reason, windows=[]), None
    values = np.asarray([row[:-1] for row in case["samples"]], dtype=float)
    data = dict(zip(columns[:-1], values.T, strict=True))
    time, added = data["t_s"], data["applied_load_N"]
    normal = np.column_stack((data["left_inner_N"], data["right_inner_N"]))
    slide = np.column_stack((data["left_sliding_N"], data["right_sliding_N"]))
    tension, support = data["tension_N"], data["support_N"]
    np.testing.assert_allclose(np.diff(time), .002, atol=1e-9, rtol=0.)
    qualified = (added > 0.) & (support <= .05) & (tension > .001) & (normal.min(axis=1) > .001)
    qualified &= (slide < .5 * normal).all(axis=1)
    if onset := evidence["first_sustained_bilateral_sliding"]:
        qualified &= time < onset["t_s"]
    separation = data["tip_z_m"] - data["payload_com_z_m"]
    table = np.column_stack((time, added, tension, support, separation))
    fitted = fit_windows(table, qualified, .00025)
    return dict(case=case["id"], material=case["material"], rate_N_s=evidence["ramp_rate_N_s"], windows=fitted), (
        table, qualified)


def shared_force_ramp(fast, slow):
    limits = []
    for table, eligible in (fast, slow):
        a, b = intervals(eligible)[0]
        limits.append((table[a, 1], table[b - 1, 1]))
    lo, hi = max(r[0] for r in limits), min(r[1] for r in limits)
    fits = [fit_windows(t, q & (t[:, 1] >= lo - 1e-10) & (t[:, 1] <= hi + 1e-10), .00025)
            for t, q in (fast, slow)]
    values = [rows[0] if rows else None for rows in fits]
    change = None
    if all(values):
        change = values[1]["stiffness_N_m"] / values[0]["stiffness_N_m"] - 1.
    return dict(added_load_overlap_N=[float(lo), float(hi)], fast=values[0], slow=values[1],
                relative_slow_minus_fast=change)


def main():
    source = json.loads(DATA.read_bytes())["load_ramps"]
    results, data = [], {}
    for case in source["cases"]:
        result, raw = measure(case, source["sample_columns"])
        results.append(result)
        data[case["id"]] = raw
    matched = shared_force_ramp(data["kraft-clear-fast"], data["kraft-clear-slow"])
    report = dict(measurement="world-vertical tip-to-payload separation versus native finger tension",
                  sample_period_s=.002, windows_by_case=results, kraft_common_loading_force=matched)
    with (RUN / "stiffness-comparison.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

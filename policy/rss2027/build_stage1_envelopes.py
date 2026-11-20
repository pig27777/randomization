"""Build the Stage 2 input from immutable Stage 1 numerical archives."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "test/runs"
NOV19 = RUNS / "2026-11-19/bag"
NOV20 = RUNS / "2026-11-20/bag"
LOADS = NOV20 / "rss2027-stage1-material-load-ramps-r1"
CURVES = NOV20 / "rss2027-stage1-reference-curves-r1"
PE_SLIP = NOV20 / "rss2027-stage1-pe-held-load-step-r1"
KRAFT_SLIP = NOV20 / "rss2027-stage1-kraft-held-local-mesh-r1"
WRAP = NOV20 / "rss2027-stage1-wrap-content-continuation-r1"
MATERIALS = ("pe", "kraft", "bubble", "paperboard")
UNITS = {"S_bag": "m", "alpha": "m/m", "k": "N/m", "mu_L": "1", "mu_R": "1", "beta": "m/(N*s)",
         "s_crit": "m", "F_min": "N", "h_min": "m"}
EXPANSION = 1.75


class FrozenEvidence:
    def __init__(self):
        self.sources = {}

    def read(self, path):
        raw = path.read_bytes()
        key = path.relative_to(ROOT).as_posix()
        self.sources[key] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    def json(self, path):
        return json.loads(self.read(path))


def limits(values):
    values = list(values)
    if not values or any(not math.isfinite(value) or value < 0. for value in values):
        raise ValueError(f"Expected finite nonnegative source values, got {values}")
    return [min(values), max(values)]


def expanded(anchor, bounds):
    """Expand positive observations multiplicatively within the design limits."""
    if anchor is None:
        return list(bounds)
    low, high = anchor
    result = [max(bounds[0], low / EXPANSION), min(bounds[1], high * EXPANSION)]
    if not result[0] <= low <= high <= result[1]:
        raise ValueError(f"Source interval {anchor} exceeds the design bounds {bounds}")
    return result


def priors(ranges):
    # Two independent latent effects represent material stiffness and shared pad friction.
    # These are engineering priors. No paired data identify the empirical covariance.
    stiffness = {"alpha": -.65, "k": .70, "s_crit": .35, "F_min": .45, "h_min": .35}
    friction = {"mu_L": .9, "mu_R": .9, "beta": -.35}
    order = list(UNITS)
    corr = [[1. if i == j else stiffness.get(i, 0.) * stiffness.get(j, 0.)
             + friction.get(i, 0.) * friction.get(j, 0.) for j in order] for i in order]
    scales = [(ranges[key][1] - ranges[key][0]) / math.sqrt(12.) for key in order]
    cov = [[scales[i] * corr[i][j] * scales[j] for j in range(len(order))] for i in range(len(order))]
    return dict(order=order, correlation=corr, covariance_si=cov, empirical_covariance=None,
                basis="engineering prior using uniform marginal variances; geometry conditions S_bag separately")


def geometry(evidence):
    folder = NOV20 / "rss2027-stage1-geometric-slack-budget-r1"
    summary = evidence.json(folder / "slack-limits.json")
    groups = {group: [] for group in summary["groups"]}
    for row in csv.DictReader(evidence.read(folder / "slack-limits.csv").splitlines()):
        opposing = [sum(float(row[f"slack_{axis}_{side}_m"]) for side in ("minus", "plus")) for axis in ("x", "y")]
        area_cap = float(row["reserve_length_at_pad_width_upper_m"])
        groups[row["group"]].append(min(max(opposing), area_cap))
    for name, values in groups.items():
        if len(values) != summary["groups"][name]["samples"]:
            raise ValueError(f"Missing frozen geometry samples for {name}")
    return dict(sample_count=sum(map(len, groups.values())), design_sha256=summary["design_sha256"],
                pad_width_m=summary["pad_width_m"], by_fill_group=summary["groups"],
                opposing_axis_budget_by_group_m={group: limits(values) for group, values in groups.items()},
                maximum_observed_budget_m=max(max(values) for values in groups.values()),
                domain_fill_fraction_prior=[.25, .95],
                use="recompute from each bag, payload, seal and offset; group extrema are geometric upper estimates")


def release_csv(evidence, file):
    first = None
    previous = None
    for row in csv.DictReader(evidence.read(file).splitlines()):
        added = float(row["applied_load_N"])
        time = float(row["t_s"])
        if added > 0. and float(row["max_finger_contact_N"]) <= .05 and float(row["support_N"]) <= .05:
            if first is None or time - previous > .0025:
                first = dict(added_N=added, opening_m=float(row["actual_gap_m"]), time_s=time)
            if time - first["time_s"] >= .048 - 1e-9:  # 25 samples at 2 ms
                return first
        else:
            first = None
        previous = time
    raise ValueError(f"No sustained unsupported detachment in {file}")


def release_json(evidence, file):
    report = evidence.json(file)
    loss = report.get("native_load", report)["loss"]
    if loss["support_N"] > .05:
        raise ValueError(f"Detachment has support in {file}")
    return dict(added_N=loss["added_N"], opening_m=loss["actual_gap_m"])


def build():
    evidence = FrozenEvidence()
    curve_set = evidence.json(CURVES / "stage1_whole_bag_data.json")
    cases, ramps = curve_set["cases"], curve_set["load_ramps"]["cases"]
    stiffness = evidence.json(LOADS / "stiffness-comparison.json")
    windows = {row["case"]: row["windows"] for row in stiffness["windows_by_case"]}
    k = {"pe": limits([windows["pe-matched-load"][0]["stiffness_N_m"]]),
         "kraft": limits(row["stiffness_N_m"] for key, row in stiffness["kraft_common_loading_force"].items()
                         if key in ("fast", "slow")),
         "bubble": limits(windows[key][0]["stiffness_N_m"] for key in ("bubble-fast-lift", "bubble-fast-lift-20g")),
         "paperboard": limits([windows["paperboard-rounded-pad"][0]["stiffness_N_m"]])}
    local_alpha = evidence.json(NOV19 / "rss2027-stage1-pe-gather-fine-grid-r1/common-window.json")
    conditions = [evidence.json(NOV20 / "rss2027-stage1-pe-gather-conditions-r1" / case / "fit.json")
                  for case in ("slack-4mm", "pressure-0p8N")]
    reference_alpha = limits(row["slope"] for row in local_alpha["case_fit"].values())
    ridge = WRAP / "fold-aligned-60-side-energy/sample-058/ridge-height"
    ridge_geo = evidence.json(ridge / "gathering-material-balance.json")
    ridge_step = evidence.json(ridge / "local-step-comparison.json")
    beta = {key: limits(row["fitted"]["beta_m_per_N_s"] for row in evidence.json(path)["trials"])
            for key, path in (("pe", PE_SLIP / "local-tangent-exact/comparison.json"),
                              ("kraft", KRAFT_SLIP / "local-tangent/comparison.json"))}
    feed = evidence.json(WRAP / "pe-capture-travel/comparison-11p25mm.json")["trials"]
    retained = evidence.json(CURVES / "retained-material-windows.json")
    slack = geometry(evidence)
    releases = {
        "pe": {"vertical": [release_csv(evidence, LOADS / "pe/recording/load-curve.csv")]},
        "kraft": {"vertical": [release_csv(evidence, NOV20 / name / "recording/load-curve.csv") for name in
                  ("rss2027-stage1-kraft-load-ramp-clear-r1", "rss2027-stage1-kraft-load-ramp-r1")],
                  "tangential": [release_json(evidence, NOV20 / "rss2027-stage1-kraft-held-load-step-r1/tangent" /
                                             f"dt-{step}us.json") for step in (50, 100)]},
        "bubble": {"tangential": [release_json(evidence, LOADS / "bubble-tangent" / run / "recording/summary.json")
                                  for run in ("half-step", "fast-ramp")]},
        "paperboard": {"vertical": [release_csv(evidence, LOADS / "paperboard-rounded-pad/recording/load-curve.csv")],
                       "tangential": [release_json(evidence, LOADS / run / "recording/summary.json")
                                      for run in ("paperboard-tangent/slow-ramp", "paperboard-tangent")]},
    }
    records = {}
    for material in MATERIALS:
        alpha = reference_alpha if material == "pe" else None
        bounds = {"S_bag": [0., slack["maximum_observed_budget_m"] * EXPANSION],
                  "alpha": {"pe": [.55, .95], "kraft": [.25, .70], "bubble": [.45, .95],
                            "paperboard": [.25, .55]}[material],
                  "k": [5., 2000.], "mu_L": [.15, 1.10], "mu_R": [.15, 1.10],
                  "beta": [.0002, .03], "s_crit": [.001, .020], "F_min": [.05, .80], "h_min": [.001, .025]}
        # Keep alpha broad because a local PE fit alone does not identify its whole-bag range.
        bounds["k"] = expanded(k[material], bounds["k"])
        bounds["beta"] = expanded(beta.get(material), bounds["beta"])
        records[material] = dict(
            anchors=dict(alpha_local=alpha, k_N_per_m=k[material], beta_local_m_per_N_s=beta.get(material),
                         release_applied_load_by_axis=releases[material],
                         retained_reference_excess_m=limits(value for row in retained if row["material"] == material
                                                           for value in row["ranges"]["reference_excess_m"]),
                         s_crit=None, mu_eff=None, F_min=None, h_min=None),
            tier1_ranges=bounds, dependency=priors(bounds))
    return dict(schema_version=1, stage1_status="frozen_without_full_acceptance", units=UNITS,
                proxy_contract=dict(dt_s=[.001, .002], state_shape=["N_envs", 2, 6], finger_order=["left", "right"],
                                    state_fields={"s": "m", "capture": "1", "slip_cum": "m", "hold_time": "s",
                                                  "feed_depth": "m", "peak_T": "N"}),
                sample_counts=dict(actions=len(cases), action_samples=sum(len(row["samples"]) for row in cases),
                                   load_curves=len(ramps), load_samples=sum(len(row["samples"]) for row in ramps)),
                exploratory_action_ids=sorted(curve_set["exploratory_cases"]),
                tier1_expansion=EXPANSION, global_design_ranges=dict(alpha=[.25, .95], k_N_per_m=[5., 2000.]),
                notes=dict(alpha="PE controlled local fold slope; not a cross-material estimate",
                           beta="local same-state time-step evidence only; mesh and full loading not accepted",
                           release="applied external load at separation, kept by axis; not a friction coefficient",
                           mass="payload and parcel mass are simulation inputs, not measured specimen weights"),
                pe_additional_local_alpha=[row["qualifying_intervals"][0]["slope_fold_m_per_total_inward_m"]
                                           for row in conditions], pe_feed_depth_diagnostics=feed,
                pe_whole_bag_ridge_diagnostic=dict(
                    spatial_fold_slope_m_per_m=ridge_step["original_slope_m_per_m"],
                    reference_excess_slope_m_per_m=ridge_geo["reference_excess_slope_m_per_m"],
                    reference_excess_growth_m=ridge_geo["reference_excess_growth_m"], calibrated=False),
                geometry=slack, materials=records, source_sha256=evidence.sources)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("proxy_material_envelopes.yaml"))
    args = parser.parse_args()
    result = build()
    # The JSON representation is valid YAML 1.2 and allows generation without PyYAML.
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("material | Stage 1 alpha | Stage 1 k (N/m) | Stage 1 beta (m/(N*s)) | Tier 1 SI ranges")
    for material, entry in result["materials"].items():
        raw = entry["anchors"]
        print(f"{material} | {raw['alpha_local']} | {raw['k_N_per_m']} | {raw['beta_local_m_per_N_s']} | "
              f"{json.dumps(entry['tier1_ranges'], separators=(',', ':'))}")


if __name__ == "__main__":
    main()

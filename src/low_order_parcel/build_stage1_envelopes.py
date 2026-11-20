"""Generate the Stage 2 material configuration from the frozen Milestone 1 sources."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from policy.rss2027.build_stage1_envelopes import build as build_frozen_evidence

OUTPUT = ROOT / "configs/proxy_material_envelopes.yaml"
FROZEN = ROOT / "policy/rss2027/proxy_material_envelopes.yaml"

# Engineering choices for reproducible, single-body probes. The measured anchors in
# each entry remain unchanged and are independent of these model inputs.
PROBE_PRIORS = {
    # mass (kg), alpha and beta if unmeasured, pad friction, slip reserve (m),
    # clamp load (N) and opening span (m)
    "pe": (.20, None, None, (.63, .55), .018, 6., .0087),
    "kraft": (.20, .47, None, (.66, .72), .0037, 2.45, .0176),
    "bubble": (.04, .74, .022, (.39, .45), .0025, .62, .0030),
    "paperboard": (.20, .35, .010, (.66, .70), .018, 5.4, .0205),
}


def build():
    evidence = build_frozen_evidence()
    frozen_bytes = FROZEN.read_bytes()
    if json.loads(frozen_bytes) != evidence:
        raise ValueError(f"Live evidence differs from the frozen artifact {FROZEN}")
    evidence["frozen_manifest_sha256"] = hashlib.sha256(frozen_bytes).hexdigest()
    evidence["nominal_basis"] = {
        "k": "each rope contributes half of the whole-bag elastic slope; use slow kraft/20 g bubble anchors",
        "c_d": "0.8 critical damping of the nominal vertical rigid mass on the two parallel strands",
        "alpha_beta": "midpoint of the local anchor when available, otherwise an engineering prior",
        "others": "engineering priors for the Stage 2 probes; forces, slack and friction are not specimen fits",
    }
    for material, entry in evidence["materials"].items():
        mass, alpha, beta, mu, reserve, grip, opening_span = PROBE_PRIORS[material]
        anchors = entry["anchors"]
        k = anchors["k_N_per_m"][0] / 2.
        params = dict(S_bag=.034, s_max=reserve, alpha=alpha if alpha is not None else sum(anchors["alpha_local"]) / 2.,
                      k=k, c_d=.8 * math.sqrt(2. * mass * k), mu_eff=list(mu),
                      beta=beta if beta is not None else sum(anchors["beta_local_m_per_N_s"]) / 2.,
                      s_crit=.0015, F_min=.25, F_lock=.48, F_release=.10, h_min=.0035, cos_theta_max=.65,
                      tau_gate=.012, v_gather_max=.075, L0=[.065, .065])
        entry["nominal"] = dict(mass_kg=mass, proxy=params, probe_grip_N=grip, probe_opening_span_m=opening_span)
    return evidence


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    envelope = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(envelope, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("material | Stage 1 alpha | Stage 1 total k (N/m) | nominal alpha | nominal rope k (N/m) | Tier 1 ranges (SI)")
    for material, entry in envelope["materials"].items():
        anchors, nominal = entry["anchors"], entry["nominal"]["proxy"]
        print(f"{material} | {anchors['alpha_local']} | {anchors['k_N_per_m']} | {nominal['alpha']:.4f} | "
              f"{nominal['k']:.4f} | {json.dumps(entry['tier1_ranges'], separators=(',', ':'))}")


if __name__ == "__main__":
    main()

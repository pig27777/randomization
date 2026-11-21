"""A1C. Fixed-equation, simulation-only sensitivity around archived Stage 2 probes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim
from src.low_order_parcel.verify_stage2_alignment import CURVES, detachment_metrics, feed_metrics

DT = .002


def save_wave(path, trace):
    names = ("time_s","gap_m","external_load_N","normal_force_N","tension_N","slip_rate_mps",
             "contact_state","support_N","table_clearance_m","action_change_s")
    np.savez_compressed(path, **{key:trace[key] for key in names})


@torch.inference_mode()
def main(output):
    output.mkdir(parents=True, exist_ok=False)
    entries=json.loads((ROOT/"configs/proxy_material_envelopes.yaml").read_bytes())["materials"]
    archive=json.loads(CURVES.read_bytes())
    bubble=entries["bubble"]["nominal"]
    opening = next(float(r["release"]["first_final_contact_loss_sample"]["actual_gap_m"])
                   for r in archive["cases"] if r["id"] == "bubble-unsupported-release")
    load=entries["bubble"]["anchors"]["release_applied_load_by_axis"]["tangential"][0]["added_N"]
    rows=[]
    cases = [("nominal",1.,.034),("friction",.85,.034),("budget",1.,.011),
                                 ("combined",.85,.011),("lower",.75,.008),("low_budget",.85,.006),
                                 ("near_gate",.85,.0038),("near_gate_lower_mu",.65,.0038),
                                 ("small_reserve",.90,.0034),("low_friction",.50,.0042)]
    # Resolve the transition between a prepared grip and loss during preloading.
    cases.extend((f"near_gate_{pct}", pct/100, .0038) for pct in (52, 54, 56, 58, 60, 62, 64))
    for name, mu_scale, budget in cases:
        mu=[v*mu_scale for v in bubble["proxy"]["mu_eff"]]
        sim=MacroParcelSim.from_nominal(bubble|{"proxy":bubble["proxy"]|{"mu_eff":mu,"S_bag":budget}},DT)
        for mode,trace,observable,ref in (
                ("release",sim.run_controlled_release(grip_N=bubble["probe_grip_N"],
                    opening_span_m=bubble["probe_opening_span_m"]).channels,"gap_m",opening),
                ("ramp",sim.run_suspended_load_ramp(grip_N=bubble["probe_grip_N"],axis="tangential").channels,
                 "external_load_N",load)):
            save_wave(output/f"bubble-{name}-{mode}.npz",trace)
            row=detachment_metrics(trace,DT,observable,[float(ref)])
            rows.append(dict(case=name,mode=mode,mu_L=mu[0],mu_R=mu[1],budget_m=budget,
                             reference=ref,**row))
    pe=entries["pe"]["nominal"]
    feeds=[]
    for label,h_min,s_crit in (("nominal",.0035,.0015),("threshold",.0045,.0015),("reserve_gate",.0035,.0018),
                               ("right_blocked",.011,.0015)):
        sim=MacroParcelSim.from_nominal(pe|{"proxy":pe["proxy"]|{"h_min":h_min,"s_crit":s_crit}},DT)
        for depth in (.0075,.01125,.015531134179):
            trace=sim.run_side_grasp_and_hover(feed_depth_m=depth).channels
            save_wave(output/f"pe-{label}-{1000*depth:.2f}mm.npz",trace)
            feeds.append(dict(case=label,h_min_m=h_min,s_crit_m=s_crit,depth_m=depth,**feed_metrics(trace,DT)))
    (output/"sensitivity.json").write_text(json.dumps(dict(bubble=rows,pe=feeds),indent=2,allow_nan=False)+"\n")
    for row in rows:
        print(row["case"],row["mode"],row.get("value"),row["pass_gate"])
    for row in feeds:
        print(row["case"],row["depth_m"],row["bilateral_unsupported_s"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    main(parser.parse_args().output)

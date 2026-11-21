"""Fixed physical corners, open-loop Sobol actions, strict Stage 2 hover and release checks."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch

from policy.rss2027.domain import DomainRandomizer
from policy.rss2027.grasp_env import SoftParcelGraspEnv
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim
from src.low_order_parcel.lumped_contact_proxy import LumpedContactProxy
from src.low_order_parcel.unified_payload import UnifiedPayloadTemplate
from src.low_order_parcel.verify_stage2_alignment import detachment_metrics


def measure_release():
    """One fixed C1 episode on CPU float32, compare the frozen unsupported tangential ramp."""
    e = DomainRandomizer("cpu",0).sample(1,corners=torch.tensor([0]))
    sim = MacroParcelSim(LumpedContactProxy(e.proxy,.002),payload=UnifiedPayloadTemplate(e.x))
    trace = sim.run_suspended_load_ramp(grip_N=.62,axis="tangential").channels
    return detachment_metrics(trace,.002,"external_load_N",[.55])


@torch.no_grad()
def search(candidates, device):
    """Return all 8×N measured outcomes and the best observed open-loop action per corner."""
    torch.manual_seed(347)
    ids = torch.arange(8,device=device).repeat_interleave(candidates)
    e = DomainRandomizer(device,347).sample(len(ids),corners=ids)
    env = SoftParcelGraspEnv(e,open_loop=True)
    points = torch.quasirandom.SobolEngine(5,scramble=True,seed=347).draw(candidates).to(device).repeat(8,1)
    mode = (points[:,0]>.5).long()
    env.touch()
    env.grasp(mode,points[:,1:])
    env.finish(torch.zeros_like(mode),points[:,:2])
    holds = env.best.reshape(8,candidates).cpu()
    finite = env.finite.reshape(8,candidates).cpu()
    controls = points.reshape(8,candidates,5).cpu()
    results = {}
    for i,row in enumerate(holds):
        j = row.argmax()
        results[f"C{i+1}"] = dict(best_hold_s=row[j].item()*.002, successes=int(((row>=1500)&finite[i]).sum()),
                                 finite=int(finite[i].sum()), best_mode=int(controls[i,j,0]>.5),
                                 best_action=controls[i,j,1:].tolist(),ticks=row.int().tolist())
    return results


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--candidates",type=int,default=128)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path")
    start=perf_counter()
    release=measure_release()
    results=search(args.candidates,args.device)
    record=dict(torch=torch.__version__,wall_s=perf_counter()-start,candidates=args.candidates,release=release,
                corners=results,passed=bool(release["pass_gate"] and .45<=release["value"]<=.67
                                            and all(r["successes"] for r in results.values())))
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False,default=lambda x:x.item())+"\n")
    print({k:(v["best_hold_s"],v["successes"]) for k,v in results.items()},release,flush=True)

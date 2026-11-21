"""Train and blind-evaluate hybrid-action, asymmetric PPO on the segmented parcel."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import socket
from time import perf_counter

import torch
from torch import nn
from torch.distributions import Categorical, Normal
from .domain import DomainRandomizer
from .grasp_env import SoftParcelGraspEnv, VARIANTS


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(18,128,batch_first=True)
        self.mode = nn.ModuleList([nn.Linear(128,2),nn.Linear(128,3)])
        self.mean = nn.ModuleList([nn.Linear(128,8),nn.Linear(128,2)])
        self.log_std = nn.ParameterList([nn.Parameter(torch.zeros(4)),nn.Parameter(torch.zeros(2))])
        self.critic = nn.Sequential(nn.Linear(198,128),nn.Tanh(),nn.Linear(128,1))
        with torch.no_grad():
            self.mode[1].bias.copy_(torch.tensor([.5,0.,-.6]))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Previously saved actors had no Phase B current/vision channels.
        key = "encoder.weight_ih_l0"
        if key in state_dict and state_dict[key].shape[1] == 15:
            state_dict = state_dict.copy()
            state_dict[key] = nn.functional.pad(state_dict[key], (0, 3))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def distributions(self, obs, stage):
        """Float32 [B,T,18] normalized hardware history -> [B,128] latent and action heads."""
        if obs.shape[-1] == 15:
            obs = nn.functional.pad(obs, (0, 3))
        h = self.encoder(obs)[1][0]
        return h,Categorical(logits=self.mode[stage](h)),self.mean[stage](h)

    def decide(self, obs, priv, stage, *, saved=None, forced=None, greedy=False):
        """Normalized float32 history [B,T,18], privileged [B,70] -> hybrid choice/logp/value."""
        h, dist, means = self.distributions(obs,stage)
        mode = saved[0] if saved is not None else forced if forced is not None else (
            dist.probs.argmax(1) if greedy else dist.sample())
        if stage==0:
            means = means.reshape(-1,2,4)[torch.arange(len(obs),device=obs.device),mode]
        continuous = Normal(means,self.log_std[stage].clamp(-2.,1.).exp())
        raw = saved[1] if saved is not None else means if greedy else continuous.sample()
        # The stored action is a latent normal variate. The deterministic sigmoid
        # Jacobian cancels in the PPO likelihood ratio for this same action.
        mask = 1. if stage==0 else (mode==1).float()
        logp = continuous.log_prob(raw).sum(1)*mask
        entropy = continuous.entropy().sum(1)*mask
        if forced is None:
            logp = logp+dist.log_prob(mode)
            entropy = entropy+dist.entropy()
        value = self.critic(torch.cat((h,priv),1))[:,0]
        return mode,raw,logp,value,entropy


@dataclass
class Decision:
    stage: int
    obs: torch.Tensor
    priv: torch.Tensor
    mode: torch.Tensor
    raw: torch.Tensor
    logp: torch.Tensor
    value: torch.Tensor
    active: torch.Tensor
    reward: torch.Tensor
    forced: bool
    target: torch.Tensor | None = None


def heuristic(obs,stage,previous):
    """Dimensionless [B,T,18] hardware features -> int64 [B], float32 [B,4 or 2] controls."""
    if stage==0:
        mode = ((obs[:,:,0].amax(1)<.85)&(obs[:,-1,10]>=.2)).long()
        if previous is not None:
            mode = torch.where(obs[:,-1,10]>=.2,1-previous,mode)
        action = obs.new_tensor([.55,.82,.6,.72]).expand(len(obs),-1)
    else:
        ratio = obs[:,-3:,2].mean(1)
        mode = torch.where(ratio<.08,2,torch.where(ratio<.65,1,0))
        action = obs.new_tensor([.65,.65]).expand(len(obs),-1)
    return mode,torch.logit(action)


@torch.no_grad()
def rollout(policy,episode,variant,*,training=False,balanced=False):
    """Run at most 3 attempts; return PPO decisions and per-world metric tensors."""
    env = SoftParcelGraspEnv(episode,variant,training)
    n = env.count
    live = torch.ones_like(episode.material,dtype=torch.bool)
    success = ~live
    recovered = ~live
    failed_micro = ~live
    retries = torch.zeros_like(episode.material)
    first = None
    previous = None
    preceding_micro = None
    decisions = []
    for attempt in range(3):
        touch = env.touch()
        route_before = previous
        def choose(obs,stage,forced=None):
            priv = env.privileged()
            if variant=="b0_heuristic":
                return heuristic(obs,stage,route_before)
            mode,raw,lp,value,_ = policy.decide(obs,priv,stage,greedy=not training,forced=forced)
            decisions.append(Decision(stage,obs,priv,mode,raw,lp,value,live.clone(),
                                      torch.zeros_like(lp),forced is not None))
            return mode,raw

        forced = None
        if balanced:
            # Exactly half of the rows practise each primitive on a shuffled domain.
            forced = torch.randperm(n,device=touch.device)%2
        decision_obs = touch
        if preceding_micro is not None and variant not in ("ablation_no_history","ablation_no_microlift"):
            decision_obs = torch.cat((preceding_micro,touch),1)
        mode,raw = choose(decision_obs,0,forced)
        if first is None:
            first = mode
        previous = mode
        micro = env.grasp(mode,raw.sigmoid())
        preceding_micro = micro
        valid = env.valid_micro[:,0].clone()
        overpress = (env.peak_press[:,0]>3.) & (episode.x.max_indent_ratio[:,0]==0) & (mode==0)
        if training and variant!="b0_heuristic":
            decisions[-1].reward += -.15*(mode==1)-.35*overpress
            if attempt==0:
                decisions[-1].reward += .25*(env.net[:,0]/.02).clamp(0.,1.)
        post = torch.cat((touch,micro),1) if variant!="ablation_no_history" else micro
        route,comp = choose(post,1)
        ok = env.finish(route,comp.sigmoid()) & live
        recovered |= ok & (failed_micro | (~valid & (route==1)))
        success |= ok
        wants_retry = live & (route==2)
        if training and variant!="b0_heuristic":
            decisions[-1].reward += 10.*ok-4.*(env.dropped[:,0] & live)-.35*wants_retry
            decisions[-1].reward += 1.75*(ok & (failed_micro | (~valid & (route==1))))
            decisions[-1].reward -= 2.*(live & ~ok & (attempt==2))
        failed_micro |= wants_retry & ~valid
        retries += wants_retry & (attempt<2)
        live = wants_retry & (attempt<2)
        if not live.any():
            break
        env.rewind()
    run = torch.zeros_like(success,dtype=torch.float32)
    for row in reversed(decisions):
        run = torch.where(row.active,row.reward+.99*run,0.)
        row.target = run
    metrics = dict(success=success,first=first,retries=retries,recovered=recovered,
                   finite=env.finite[:,0],bucket=episode.material*5+episode.content,
                   content=episode.content,physics_steps=env.physics_steps)
    return decisions,metrics


def update(policy,opt,rows):
    """Clipped PPO using active samples and complete discounted macro returns."""
    for _ in range(4):
        for row in rows:
            ids = torch.where(row.active)[0]
            adv = row.target[ids]-row.value[ids]
            adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-5)
            for selection in torch.randperm(len(ids),device=ids.device).split(256):
                idx = ids[selection]
                _,_,lp,value,entropy = policy.decide(row.obs[idx],row.priv[idx],row.stage,
                                    saved=(row.mode[idx],row.raw[idx]),forced=row.mode[idx] if row.forced else None)
                ratio = (lp-row.logp[idx]).exp()
                gain = torch.minimum(ratio*adv[selection],ratio.clamp(.8,1.2)*adv[selection])
                loss = -gain.mean()+.25*(value-row.target[idx]).square().mean()-.006*entropy.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(),.5,error_if_nonfinite=True)
                opt.step()


def summarize(m):
    """Convert per-world outcome tensors to aggregate counts with explicit denominators."""
    ok,content = m["success"],m["content"]
    def count(mask):
        return int(mask.sum())
    return dict(worlds=len(ok),successes=count(ok),rate=float(ok.float().mean()),finite=count(m["finite"]),
                retrying=count(m["retries"]>0),recovered=count(m["recovered"]),
                book_worlds=count(content==0),book_side=count((content==0)&(m["first"]==1)),
                textile_worlds=count((content>=1)&(content<=3)),
                textile_top=count((content>=1)&(content<=3)&(m["first"]==0)),
                physics_steps=m["physics_steps"])


def evaluate(policy,variant,seed,n,device):
    """Independent fixed full-domain and C1–C8 evaluation with genuine proxy and hardware."""
    torch.manual_seed(seed)
    dr = DomainRandomizer(device,seed)
    _,full = rollout(policy,dr.sample(n),variant)
    ids = torch.arange(n,device=device)%8
    _,corners = rollout(policy,dr.sample(n,corners=ids),variant)
    return dict(full=summarize(full),corners={f"C{i+1}":dict(successes=int(corners["success"][ids==i].sum()),
                worlds=int((ids==i).sum())) for i in range(8)})


def train(args):
    device,n=args.device,args.worlds
    torch.manual_seed(args.seed)
    policy=Policy().to(device)
    opt=torch.optim.Adam(policy.parameters(),lr=.001)
    dr=DomainRandomizer(device,args.seed)
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,obj):
        (args.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    save("config.json",vars(args)|dict(output=str(args.output),torch=torch.__version__,host=socket.gethostname()))
    wide=False
    seconds=0.
    total_steps=0
    stable=0
    converged=None
    weights=None
    last=None
    for i in range(args.updates if args.variant!="b0_heuristic" else 0):
        warm=i<max(1,round(args.updates*.20))
        if device=="cuda":
            torch.cuda.synchronize()
        start=perf_counter()
        e=dr.sample(n,wide=wide,rigid=args.variant=="ablation_rigid_payload_only",bucket_weights=weights if wide else None)
        decisions,metrics=rollout(policy,e,args.variant,training=True,balanced=warm)
        update(policy,opt,decisions)
        if device=="cuda":
            torch.cuda.synchronize()
        seconds+=perf_counter()-start
        total_steps+=metrics["physics_steps"]
        result=summarize(metrics)
        counts=torch.bincount(metrics["bucket"],minlength=20)
        wins=torch.bincount(metrics["bucket"],weights=metrics["success"].float(),minlength=20)
        rates=wins/counts.clamp_min(1)
        weights=torch.where((counts>0)&(rates<rates.mean()),2.,1.)
        record=dict(update=i+1,warm=warm,wide=wide,train=result,cumulative_s=seconds,
                    env_steps_s=total_steps/seconds,gpu_hours=seconds/3600)
        if not warm and not wide:
            # Route with deterministic actions in a separately seeded narrow-domain holdout.
            if (i+1)%args.eval_every==0:
                _,narrow=rollout(policy,DomainRandomizer(device,66001).sample(n,wide=False,
                                rigid=args.variant=="ablation_rigid_payload_only"),args.variant)
                record["narrow_validation"]=summarize(narrow)
                wide = record["narrow_validation"]["rate"]>=.85
        elif wide and (i+1)%args.eval_every==0:
            last=evaluate(policy,args.variant,77001,n,device)
            record["validation"]=last
            passed=last["full"]["rate"]>=.85 and all(v["successes"] for v in last["corners"].values())
            stable=stable+1 if passed else 0
            if stable==3:
                converged=seconds/3600
        with (args.output/"updates.jsonl").open("a") as log:
            print(json.dumps(record),file=log)
        print(i+1,round(result["rate"],3),record["wide"],flush=True)
        if converged is not None:
            break
    if args.variant!="b0_heuristic":
        torch.save(policy.state_dict(),args.output/"policy.pt")
    final=evaluate(policy,args.variant,88001,n,device)
    save("result.json",dict(variant=args.variant,seed=args.seed,training_gpu_hours=seconds/3600,
         env_steps_s=total_steps/max(seconds,1e-8),wide_training=any(
         json.loads(line)["wide"] for line in (args.output/"updates.jsonl").read_text().splitlines()
         ) if total_steps else False,converged_gpu_hours=converged,evaluation=final))


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--variant",choices=VARIANTS,default="ours")
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--device",default="cuda",choices=("cuda","cpu"))
    p.add_argument("--worlds",type=int,default=256)
    p.add_argument("--updates",type=int,default=30)
    p.add_argument("--eval-every",type=int,default=3)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.worlds<8 or a.updates<1 or a.eval_every<1:
        p.error("require at least 8 worlds and positive update/evaluation intervals")
    train(a)

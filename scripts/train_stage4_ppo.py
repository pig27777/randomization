"""Train and gate Stage 4 Policy 1 on the frozen Fix 4 macro environment."""

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import socket
import subprocess
from time import perf_counter

import torch
from torch.utils._pytree import tree_map

from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.low_order_parcel.envs.pick_to_air_env import PickToAirEnv
from src.low_order_parcel.rl.asymmetric_ppo import AsymmetricPPO


BUCKET_NAMES = ("hard_book", "soft_clothes_pillow", "light_at_most_65g", "loose_fill", "side_obstruction")
FROZEN_PATHS = (
    "src/low_order_parcel/lumped_contact_proxy.py",
    "src/low_order_parcel/unified_payload.py",
    "src/low_order_parcel/macro_parcel_sim.py",
    "src/low_order_parcel/dual_arm_parcel_sim.py",
    "src/low_order_parcel/pad_torsion.py",
    "src/low_order_parcel/hardware_filters.py",
    "src/low_order_parcel/domain_randomizer.py",
    "configs",
)
BASE = "046384ab"
VALIDATION_SEED = 77001
BLIND_SEED = 88001


def curriculum(update, total):
    warm_end = round(.20 * total)
    wide_start = round(.50 * total)
    if update < warm_end:
        return "A", 0.
    if update < wide_start:
        return "B", 0.
    return "C", (update - wide_start) / max(total - wide_start - 1, 1)


def take(episode, rows):
    if isinstance(episode, torch.Tensor):
        return episode[rows]
    if is_dataclass(episode):
        return type(episode)(*(take(getattr(episode, f.name), rows) for f in fields(episode)))
    if isinstance(episode, tuple):
        return type(episode)(*(take(value, rows) for value in episode))
    return episode


def buckets(episode):
    content = episode.content
    return ((content == 0), ((content == 1) | (content == 2)),
            episode.payload.total_mass[:, 0] <= .065, content == 4,
            episode.clearance_m.amin(dim=-1) < .020)


def sample_training(dr, count, stage, alpha, hard_bucket):
    if stage == "A":
        pool = dr.sample(4 * count, curriculum_alpha=alpha)
        rigid, soft, *_ = buckets(pool)
        top = torch.nonzero(soft).flatten()[:count // 2]
        side = torch.nonzero(rigid).flatten()[:count // 2]
        if len(top) != count // 2 or len(side) != count // 2:
            raise RuntimeError("Warm-up pool cannot supply the requested balanced partition")
        rows = torch.cat((top, side))
        modes = (torch.arange(count, device=dr.device) >= count // 2).long()
        return take(pool, rows), modes
    if stage == "C" and hard_bucket is not None:
        pool = dr.sample(2 * count, curriculum_alpha=alpha)
        weights = 1. + buckets(pool)[hard_bucket].float()
        rows = torch.multinomial(weights, count, replacement=True, generator=dr.rng)
        return take(pool, rows), None
    return dr.sample(count, curriculum_alpha=alpha), None


def hardest_bucket(records):
    rates = [(r["successes"] / r["worlds"], i) for i, r in enumerate(records) if r["worlds"]]
    return min(rates)[1] if rates else None


@torch.no_grad()
def collect(model, episode, *, seed, forced_mode=None, training=False, graph=True):
    env = PickToAirEnv(episode, seed=seed, cuda_graph=graph)
    obs = env.reset(forced_mode=forced_mode)
    first = torch.full_like(obs.phase, -1)
    clearance = env.zeros(3)
    measured = torch.zeros_like(clearance, dtype=torch.bool)
    false_grasp = torch.zeros_like(clearance, dtype=torch.bool)
    corrective = torch.zeros_like(clearance, dtype=torch.bool)
    observations, samples, rewards, terminals = [], [], [], []
    decisions = torch.zeros_like(obs.phase)
    rows = torch.arange(env.batch, device=env.device)
    for _ in range(6):
        if env.done.all():
            break
        action = model.act(obs, deterministic=not training)
        active = obs.phase >= 0
        routed = obs.phase == 0
        checked = obs.phase == 1
        previous_slot = (env.attempts - 1).clamp(0, 2)
        corrective[rows, previous_slot] |= checked & (action.action.choice != 0)
        first = torch.where(routed & (first < 0), action.action.choice, first)
        transition = env.step(action.action)
        decisions += active
        slot = (env.attempts - 1).clamp(0, 2)
        clearance[rows, slot] = torch.where(routed, env.phase_b_clearance_m, clearance[rows, slot])
        false_grasp[rows, slot] |= routed & env.phase_b_false_grasp
        measured[rows, slot] |= routed
        if training:
            observations.append(obs)
            samples.append(action)
            rewards.append(transition.reward)
            terminals.append(transition.done)
        obs = transition.obs
    if not env.done.all():
        raise RuntimeError("Policy 1 exceeded its three-attempt macro horizon")
    if not all(torch.isfinite(channel).all() for channel in
               (obs.static, obs.seq, obs.priv, env.state.payload.pos, env.state.payload.vel)):
        raise FloatingPointError("Nonfinite Policy 1 rollout")
    data = dict(success=env.success, mode=first, torn=env.torn, dropped=env.dropped,
                attempts=env.attempts, clearance=clearance, measured=measured, false_grasp=false_grasp,
                corrective=corrective, decisions=decisions, physics_ticks=env.physics_ticks)
    return (observations, samples, rewards, terminals), data


def update_ppo(model, optimizer, trajectory, *, epochs, minibatch):
    obs, samples, rewards, terminals = trajectory
    last_value = torch.zeros_like(samples[0].value)
    advantage = torch.zeros_like(last_value)
    targets, advantages = [], []
    for sample, reward, terminal in reversed(list(zip(samples, rewards, terminals, strict=True))):
        cont = (~terminal).float()
        residual = reward + .99 * cont * last_value - sample.value
        advantage = residual + .99 * .95 * cont * advantage
        targets.append(advantage + sample.value)
        advantages.append(advantage)
        last_value = sample.value
    obs = tree_map(lambda *parts: torch.cat(parts), *obs)
    samples = tree_map(lambda *parts: torch.cat(parts), *samples)
    targets = torch.cat(targets[::-1])
    advantages = torch.cat(advantages[::-1])
    eligible = torch.nonzero(obs.phase >= 0).flatten()
    active = advantages[eligible]
    advantages = (advantages - active.mean()) / active.std(unbiased=False).clamp_min(1e-6)
    losses = []
    for _ in range(epochs):
        for batch in torch.randperm(len(eligible), device=eligible.device).split(minibatch):
            rows = eligible[batch]
            observation, saved = (tree_map(lambda tensor: tensor[rows], part) for part in (obs, samples))
            loss = model.loss(observation, saved, targets[rows], advantages[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
    return torch.stack(losses).mean().item()


def rate(outcome, relevant):
    successes = int((outcome & relevant).sum())
    worlds = int(relevant.sum())
    return dict(successes=successes, worlds=worlds, rate=successes / worlds if worlds else None)


def metrics(episode, data, relevant=None):
    success, mode = data["success"], data["mode"]
    if relevant is None:
        relevant = torch.ones_like(success)
    hard, soft, *_, obstructed = buckets(episode)
    all_clearance = data["clearance"][relevant][data["measured"][relevant]]
    false = data["false_grasp"]
    rescue = (false & data["corrective"]).any(dim=1)
    if len(all_clearance):
        q = torch.quantile(all_clearance, all_clearance.new_tensor([0., .05, .25, .50, .75, .95, 1.]))
        clearance = dict(frames=len(all_clearance), zero=int((all_clearance <= 0.).sum()),
                         p0_p5_p25_p50_p75_p95_p100_mm=(q * 1000.).tolist())
    else:
        clearance = dict(frames=0, zero=0, p0_p5_p25_p50_p75_p95_p100_mm=[])
    return dict(
        success=rate(success, relevant),
        modes=dict(hard_book_side=rate(mode == 1, relevant & hard),
                   unobstructed_soft_top=rate(mode == 0, relevant & soft & ~obstructed),
                   obstructed_soft_top=rate(mode == 0, relevant & soft & obstructed)),
        clearance=clearance,
        false_grasp_episodes=int((false.any(dim=1) & relevant).sum()),
        false_grasp_corrective_decision=rate(rescue, relevant & false.any(dim=1)),
        microlift_rescue=rate(rescue & success, relevant & false.any(dim=1)),
        torn=rate(data["torn"], relevant),
        dropped=rate(data["dropped"], relevant),
        mean_retries=(data["attempts"][relevant] - 1).clamp_min(0).float().mean().item(),
        decisions=int(data["decisions"][relevant].sum()),
        physics_ticks=int(data["physics_ticks"][relevant].sum()),
    )


def evaluate(model, device, count, seed, graph, *, corners=False):
    dr = DomainRandomizer(device, seed)
    if corners:
        ids = torch.arange(count, device=device) % 8
        episode = CornerSuite(dr).select(ids)
        _, data = collect(model, episode, seed=seed + 1, graph=graph)
        return {f"C{i+1}": metrics(episode, data, ids == i) for i in range(8)}
    episode = dr.sample(count)
    _, data = collect(model, episode, seed=seed + 1, graph=graph)
    return metrics(episode, data)


def preflight(device):
    subprocess.run(("git", "diff", "--exit-code", BASE, "--", *FROZEN_PATHS), check=True)
    rgb = "src/low_order_parcel/virtual_rgb.py"
    # main retains an explicit calibrated experiment API. Both revisions use
    # the identical Fix 4 projection whenever exterior_height is omitted.
    if all(subprocess.run(("git", "diff", "--quiet", revision, "--", rgb)).returncode != 0
           for revision in (BASE, "ed0181d0")):
        raise RuntimeError("Training requires a verified Fix 4 RGB projection")
    if device == "cuda" and (socket.gethostname() != "ubun" or "5090" not in torch.cuda.get_device_name()):
        raise RuntimeError("Formal training requires ubun and RTX 5090")


def train(args):
    preflight(args.device)
    torch.manual_seed(args.seed)
    model = AsymmetricPPO(PickToAirEnv.static_size, PickToAirEnv.seq_size,
                          PickToAirEnv.priv_size, PickToAirEnv.decisions).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    dr = DomainRandomizer(args.device, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        path = args.output / name
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temp.replace(path)

    save("config.json", vars(args) | dict(output=str(args.output), torch=torch.__version__,
         host=socket.gethostname(), git_head=subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
         frozen_base=BASE, validation_seed=VALIDATION_SEED, blind_seed=BLIND_SEED,
         gpu=torch.cuda.get_device_name() if args.device == "cuda" else None))
    seconds, decision_steps, physics_steps, hard = 0., 0, 0, None
    bucket_counts = torch.zeros((len(BUCKET_NAMES), 2), dtype=torch.long, device=args.device)
    best = None
    best_score = (-1., -1.)
    checkpoint = args.output / "best.pt"
    for i in range(args.updates):
        stage, alpha = curriculum(i, args.updates)
        start = perf_counter()
        episode, forced = sample_training(dr, args.worlds, stage, alpha, hard)
        trajectory, data = collect(model, episode, seed=args.seed + 2*i, forced_mode=forced,
                                   training=True, graph=args.graph)
        loss = update_ppo(model, opt, trajectory, epochs=args.epochs, minibatch=args.minibatch)
        if args.device == "cuda":
            torch.cuda.synchronize()
        seconds += perf_counter() - start
        latest = metrics(episode, data)
        decision_steps += latest["decisions"]
        physics_steps += latest["physics_ticks"]
        if stage == "C":
            for j, member in enumerate(buckets(episode)):
                bucket_counts[j, 0] += member.sum()
                bucket_counts[j, 1] += (member & data["success"]).sum()
        counts = [dict(name=name, worlds=n, successes=w)
                  for name, (n, w) in zip(BUCKET_NAMES, bucket_counts.tolist(), strict=True)]
        record = dict(update=i+1, stage=stage, alpha=alpha, loss=loss, training=latest,
                      train_s=seconds, decision_steps_s=decision_steps / seconds,
                      physical_env_steps_s=physics_steps / seconds, stage_c_bucket_counts=counts,
                      hard_bucket=BUCKET_NAMES[hard] if hard is not None else None)
        if stage == "C" and (i+1) % 20 == 0:
            hard = hardest_bucket(counts)
            bucket_counts.zero_()
        if stage == "C" and ((i+1) % args.eval_every == 0 or i+1 == args.updates) or i+1 == args.updates:
            validation = evaluate(model, args.device, args.validation_worlds, VALIDATION_SEED, args.graph)
            record["validation"] = validation
            score = (validation["success"]["rate"], validation["modes"]["hard_book_side"]["rate"] or 0.)
            if score > best_score:
                best_score = score
                best = i+1
                tmp = checkpoint.with_suffix(".tmp")
                torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), update=best,
                                validation=validation, seed=args.seed), tmp)
                tmp.replace(checkpoint)
        with (args.output / "updates.jsonl").open("a") as log:
            print(json.dumps(record, allow_nan=False), file=log)
        print(f"{i+1}/{args.updates} {stage} alpha={alpha:.3f} "
              f"train={latest['success']['rate']:.3f} train_minutes={seconds / 60:.2f}", flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True)["model"])
    full = evaluate(model, args.device, args.eval_worlds, BLIND_SEED, args.graph)
    corners = evaluate(model, args.device, args.eval_worlds, BLIND_SEED + 2, args.graph, corners=True)
    gates = dict(full=full["success"]["rate"] >= .80,
                 hard_route=(full["modes"]["hard_book_side"]["rate"] or 0.) >= .90,
                 corners=all(row["success"]["rate"] >= .80 for row in corners.values()))
    summary = dict(best_update=best, seed=args.seed, updates=args.updates, train_s=seconds,
                   gpu_hours=seconds/3600 if args.device == "cuda" else 0.,
                   peak_torch_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0.,
                   peak_torch_reserved_mib=torch.cuda.max_memory_reserved() / 2**20 if args.device == "cuda" else 0.,
                   decision_steps_s=decision_steps/seconds, physical_env_steps_s=physics_steps/seconds,
                   full=full, corners=corners, gates=gates, passed=all(gates.values()),
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    save("result.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--variant", choices=("ours",), default="ours")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--eval-worlds", type=int, default=4096)
    parser.add_argument("--validation-worlds", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.worlds < 8 or args.worlds % 2 or args.eval_worlds < 8:
        parser.error("require an even training batch and at least eight training/evaluation worlds")
    if min(args.updates, args.epochs, args.eval_every, args.minibatch, args.validation_worlds) < 1:
        parser.error("require positive updates, epochs, minibatches and validation sizes")
    train(args)

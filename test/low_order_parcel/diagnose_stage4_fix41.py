"""Read the archived Fix 4 Policy 1 run, with optional frozen checkpoint replay."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np


DAY = Path("test/runs/2026-11-22/policy")
RUN = DAY / "stage4-policy1-ours-s17" / "training"
REPLAY = DAY / "stage4-policy1-replay-r2"


def archived():
    result = json.loads((RUN / "result.json").read_text())
    updates = [json.loads(line) for line in (RUN / "updates.jsonl").read_text().splitlines()]
    quiet = {}
    for name, record in (("wide", result["full"]), *result["corners"].items()):
        n = record["success"]["worlds"]
        quiet[name] = n - sum(record[key]["successes"] for key in ("success", "dropped", "torn"))
    bounds = []
    for record in updates[:7]:
        update, stage, train = (record[key] for key in ("update", "stage", "training"))
        row = {"update": update, "success": train["success"], "routes": train["modes"]}
        if stage == "A":
            total = train["success"]["successes"]
            row["success_count_bounds_per_forced_mode"] = [max(0, total - 2048), min(2048, total)]
        bounds.append(row)

    config = json.loads(Path("configs/payload_template_X.yaml").read_text())
    weights = np.array((config["defaults"] | config["presets"]["rigid_box"])["mass_weights"])
    metadata = json.loads((REPLAY / "summary.json").read_text())
    with np.load(REPLAY / "trajectory.npz") as recording:
        last = 3 * metadata["fps"] + 1
        assert (recording["stage"][-last:] == 3).all()
        trace = recording["trace"][-last:, metadata["rows"].index(5)]
    assert np.isfinite(trace).all()
    jaw = trace[:, 20:26].reshape(last, 2, 3).mean(axis=1)
    com = (trace[:, :12].reshape(last, 4, 3) * (weights / weights.sum())[:, None]).sum(axis=1)
    speed = np.linalg.norm(np.diff(com, axis=0), axis=1) * metadata["fps"] * 1000.
    frequencies = np.fft.rfftfreq(last, 1 / metadata["fps"])
    spectra = [np.abs(np.fft.rfft((series - series.mean()) * np.hanning(last)))
               for series in (jaw[:, 0], com[:, 0])]
    sway = dict(jaw_x_span_mm=float(np.ptp(jaw[:, 0]) * 1000),
                com_x_span_mm=float(np.ptp(com[:, 0]) * 1000),
                dominant_hz=[float(frequencies[1 + x[1:].argmax()]) for x in spectra],
                com_speed_p50_p95_mm_s=np.quantile(speed, [.50, .95]).tolist(),
                intervals_at_least_20=int((speed >= 20).sum()),
                bilateral_frames=int((trace[:, 31:33] > .5).all(axis=1).sum()),
                wrist_zeta=[12 / (2 * math.sqrt(85 * mass)) for mass in (.25, .315)])

    r = np.array([0., 0., -.10])  # wrist pad to parcel anchor
    relative = np.array([.05, 0., 0.])
    force = -3. * relative
    pad_omega = np.array([0., -4., 0.])
    moment = np.cross(r, force)
    work_mJ = 2. * force.dot(relative)
    compensated_work_mJ = work_mJ - 2. * moment.dot(pad_omega)
    corotating = relative - np.cross(pad_omega, r)
    np.testing.assert_allclose(compensated_work_mJ, 2. * force.dot(corotating), atol=1e-12)
    return dict(silent_failure_counts=quiet, curriculum=bounds, c6=sway,
                dissipation_counterexample=dict(unbalanced_moment_Nm=moment.tolist(),
                    force_pair_work_mJ=work_mJ, with_pad_couple_work_mJ=compensated_work_mJ),
                archived_env_steps_s=result["physical_env_steps_s"],
                p6_env_steps_s=8192 * 2385 / 5.213)


def replay(device, training_updates):
    import torch
    from src.low_order_parcel.lumped_contact_proxy import ContactState

    # Run the recorded training procedure even if the entry point changes later.
    subprocess.run(("git", "diff", "--quiet", "5d42a43b", "--", "src/low_order_parcel", "configs"),
                   check=True)
    baseline = {"__name__": "archived_stage4_training"}
    source = subprocess.check_output(("git", "show", "5d42a43b:scripts/train_stage4_ppo.py"))
    exec(compile(source, "5d42a43b:scripts/train_stage4_ppo.py", "exec"), baseline)
    base = baseline["PickToAirEnv"]
    use_cuda = device == "cuda"

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    class Probe(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seconds = Counter()
            self.segment, self.started = "other", perf_counter()
            self.quiet = torch.zeros(self.batch, dtype=torch.long, device=self.device)
            self.longest = self.quiet.clone()

        def mark(self, segment):
            sync()
            now = perf_counter()
            self.seconds[self.segment] += now - self.started
            self.segment, self.started = segment, now

        def _touchdown(self, active):
            previous = self.segment
            self.mark("phase_0")
            super()._touchdown(active)
            self.mark(previous)

        def _grasp(self, action, active):
            self.mark("phase_A")
            reward = super()._grasp(action, active)
            self.mark("other")
            return reward

        def _verify(self, action, active):
            self.mark("phase_C")
            reward = super()._verify(action, active)
            self.mark("other")
            return reward

        def _tick(self, command, active, *, lift=False):
            if lift and self.segment == "phase_A":
                self.mark("phase_B")
            super()._tick(command, active, lift=lift)

        def _stable(self, arm):
            stable = super()._stable(arm)
            if not use_cuda:
                holding = stable & ~self.success & ~self.dropped
                self.quiet = torch.where(holding, self.quiet + 1, 0)
                self.longest = torch.maximum(self.longest, self.quiet)
            return stable

    def capture_env(*args, **kwargs):
        nonlocal env
        env = Probe(*args, **kwargs)
        return env

    def collect(label, episode, seed, *, ids=None, forced=None, optimizer=None):
        trajectory, data = baseline["collect"](model, episode, seed=seed, forced_mode=forced,
                                               graph=use_cuda, training=optimizer is not None)
        if optimizer is not None:
            env.mark("ppo")
            baseline["update_ppo"](model, optimizer, trajectory, epochs=4, minibatch=1024)
        env.mark("done")
        end_capture = (env.state.contacts_A[..., ContactState.CAPTURE] > .5).sum(dim=1)
        clear = env.result.table_clearance[:, 0]
        speed = env.sim.payload.com_velocity(env.state.payload).norm(dim=-1)
        support = env.result.support_N[:, 0]
        quiet = ~data["success"] & ~data["dropped"] & ~data["torn"]
        gates = dict(zero_capture=end_capture == 0, single_capture=end_capture == 1,
                     bilateral_on_table=(end_capture == 2) & (clear <= .001),
                     airborne_fast=(end_capture == 2) & (clear > .001) & (speed >= .020),
                     airborne_slow_supported=(end_capture == 2) & (clear > .001) & (speed < .020) & (support != 0.),
                     airborne_slow_no_support=(end_capture == 2) & (clear > .001) & (speed < .020) & (support == 0.))
        groups = {"all": torch.ones_like(quiet)}
        if ids is not None:
            groups = {f"C{i + 1}": ids == i for i in ids.unique().tolist()}
        elif forced is not None:
            groups.update({f"mode_{i}": forced == i for i in range(2)})
        rows = {}
        for name, mask in groups.items():
            failures = {k: int((mask & quiet & predicate).sum()) for k, predicate in gates.items()}
            assert sum(failures.values()) == int((mask & quiet).sum())
            rows[name] = dict(worlds=int(mask.sum()), success=int((mask & data["success"]).sum()),
                              dropped=int((mask & data["dropped"]).sum()),
                              torn=int((mask & data["torn"]).sum()),
                              modes=[int((mask & (data["mode"] == i)).sum()) for i in range(2)],
                              silent_counts=failures)
            if not use_cuda:
                rows[name].update(clearance_mm=(1000 * clear[mask]).tolist(),
                                  speed_mm_s=(1000 * speed[mask]).tolist(),
                                  longest_quiet_s=(.002 * env.longest[mask]).tolist())
        print(json.dumps(dict(run=label, graph=env._graph is not None, seconds=dict(env.seconds),
                              physical_ticks=int(data["physics_ticks"].sum()), rows=rows)), flush=True)

    env = None
    baseline["PickToAirEnv"] = capture_env
    dr, corners, ppo = (baseline[key] for key in ("DomainRandomizer", "CornerSuite", "AsymmetricPPO"))
    model = ppo(base.static_size, base.seq_size, base.priv_size, base.decisions).to(device).eval()
    model.load_state_dict(torch.load(RUN / "best.pt", map_location=device, weights_only=True)["model"])
    if use_cuda:
        collect("blind_full", dr(device, 88001).sample(4096), 88002)
    ids = torch.arange(4096, device=device) % 8 if use_cuda else torch.tensor([0, 6, 7])
    collect("blind_corners", corners(dr(device, 88003)).select(ids), 88004, ids=ids)
    if training_updates:
        torch.manual_seed(17)
        model = ppo(base.static_size, base.seq_size, base.priv_size, base.decisions).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
        training_dr = dr(device, 17)
        for i in range(training_updates):
            episode, forced = baseline["sample_training"](training_dr, 4096, "A", 0., None)
            collect(f"stage_A_{i + 1}", episode, 17 + 2 * i, forced=forced, optimizer=optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", choices=("cpu", "cuda"))
    parser.add_argument("--training-updates", type=int, choices=range(7), default=0)
    args = parser.parse_args()
    if args.training_updates and args.replay != "cuda":
        parser.error("--training-updates requires a locked CUDA replay")
    print(json.dumps(archived(), indent=2), flush=True)
    if args.replay:
        replay(args.replay, args.training_updates)

"""Ordered v1.2 acceleration checks; run formal checks under the shared flock.

The risk check deliberately uses frozen NumPy forces as a host bridge. It isolates
MJWarp's native integration before implementing or claiming a new batch backend.
"""
from __future__ import annotations

import argparse
import cProfile
from collections import defaultdict
import functools
import hashlib
import importlib.metadata
import json
import os
import pstats
from pathlib import Path
import sys
import tarfile
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results/cpu_equivalence_v1_2_1"
RELEASE = ROOT / "results/pinch_effect_v1_2/v22_review/release_v1_2"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment():
    return dict(time=datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
                loadavg=list(os.getloadavg()), affinity=sorted(os.sched_getaffinity(0)),
                proc_stat=Path("/proc/stat").read_text().splitlines()[0],
                versions={p: importlib.metadata.version(p) for p in
                          ("mujoco", "mujoco-warp", "warp-lang", "numpy")})


def reference():
    tag = json.loads((RELEASE / "v1.2.json").read_text())
    archive = RELEASE / "v1.2.tar.gz"
    if digest(archive) != tag["archive_sha256"]:
        raise RuntimeError("v1.2 archive hash mismatch")
    target = BASE / "reference_v1_2"
    if not target.exists():
        target.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(target, filter="data")
    for name, expected in tag["source_sha256"].items():
        if digest(target / name) != expected:
            raise RuntimeError(f"frozen source mismatch: {name}")
    runtime = target / "contents/segment"
    sys.path.insert(0, str(runtime))
    return tag, runtime


def state(sim):
    d = sim.d
    # Invalid numbers are retained as strings, so JSON remains standards compliant.
    def values(a):
        return np.where(np.isfinite(a), a.astype(str), "nonfinite").tolist()
    return dict(time=float(d.time), qpos=values(d.qpos), qvel=values(d.qvel),
                eq_active=d.eq_active.tolist(), warning_counts=d.warning.number.tolist(),
                max_abs_qvel=float(np.max(np.abs(d.qvel))) if np.isfinite(d.qvel).all() else None)


def risk_case(name, out, seconds):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    import model
    import protocols as P

    out.mkdir()
    k = model.ANCHORS[name]
    cpu, host = P.make(k, []), P.make(k, [])
    if cpu.built.xml != host.built.xml:
        raise RuntimeError("CPU/GPU XML mismatch")
    (out / "model.xml").write_text(cpu.built.xml)
    m, d = host.m, host.d
    md = mjw.put_model(m)
    dd = mjw.put_data(m, d, nworld=1, nconmax=1024, njmax=8192)
    external = np.zeros_like(d.xfrc_applied)
    fbuf = wp.empty((1, m.nbody), dtype=wp.spatial_vector)
    eqbuf = md.eq_data.numpy()
    dt = m.opt.timestep
    every = round(.010 / dt)
    count = round(seconds / dt)
    samples, failure = [], None
    init = state(cpu)
    save(out / "initial.json", init)

    # Compile/warm from the initial state, then restore all device state. This is
    # diagnostic only; no timing here is reported as throughput.
    mjw.step1(md, dd)
    mjw.step2(md, dd)
    wp.synchronize()
    dd = mjw.put_data(m, d, nworld=1, nconmax=1024, njmax=8192)
    start = time.perf_counter()
    for step in range(1, count + 1):
        cpu.step()
        mjw.step1(md, dd)
        if host.bushings is not None:
            for field in ("xpos", "xquat", "xmat", "xipos"):
                target = getattr(d, field)
                target[:] = getattr(dd, field).numpy()[0].reshape(target.shape)
            d.xfrc_applied[:] = external
            host.bushings.apply(m, d)
            fbuf.assign(d.xfrc_applied.astype(np.float32)[None])
            wp.copy(dd.xfrc_applied, fbuf)
        mjw.step2(md, dd)
        mjw.get_data_into(d, m, dd)
        d.xfrc_applied[:] = external
        dd.xfrc_applied.zero_()
        if host.bushings is not None:
            host.switch_welds(external)
            dd.eq_active.assign(d.eq_active[None])
            eqbuf[0, :, :m.eq_data.shape[1]] = m.eq_data
            md.eq_data.assign(eqbuf)
        if step % every == 0:
            cf, gf = cpu.failure(), host.failure()
            row = dict(step=step, t=step * dt, cpu_failure=cf, gpu_failure=gf,
                       cpu_max_abs_qvel=float(np.max(np.abs(cpu.d.qvel))),
                       gpu_max_abs_qvel=float(np.max(np.abs(d.qvel))) if np.isfinite(d.qvel).all() else None,
                       qpos_max_abs_diff=float(np.max(np.abs(d.qpos-cpu.d.qpos))) if np.isfinite(d.qpos).all() else None,
                       cpu_active_welds=int(cpu.d.eq_active.sum()),
                       gpu_active_welds=int(d.eq_active.sum()))
            samples.append(row)
            print(name, json.dumps(row), flush=True)
            if cf or gf:
                failure = row
                break
    result = dict(anchor=name, passed=failure is None, first_failure=failure,
                  steps=step, timestep=dt, solver=int(m.opt.solver), integrator=int(m.opt.integrator),
                  iterations=int(m.opt.iterations), ls_iterations=int(m.opt.ls_iterations), tolerance=float(m.opt.tolerance),
                  ntendon=m.ntendon, damped_tendons=int(np.count_nonzero(m.tendon_damping)),
                  nbody=m.nbody, neq=m.neq, xml_sha256=hashlib.sha256(cpu.built.xml.encode()).hexdigest(),
                  diagnostic_wall_s=time.perf_counter()-start, samples=samples,
                  bridge="MJWarp step1; frozen state-dependent bushing; MJWarp step2; frozen compaction switch",
                  cpu_final=state(cpu), gpu_final=state(host))
    save(out / "result.json", result)
    return result


class Timings:
    """Exclusive category timers; nested methods are never counted twice."""
    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.stack = []
        self.patches = []

    def wrap(self, owner, method, category):
        original = getattr(owner, method)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            frame = [time.perf_counter(), 0.]
            self.stack.append(frame)
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - frame[0]
                self.stack.pop()
                self.seconds[category] += elapsed - frame[1]
                self.calls[category] += 1
                if self.stack:
                    self.stack[-1][1] += elapsed
        setattr(owner, method, measured)
        self.patches.append((owner, method, original))

    def restore(self):
        for owner, method, original in reversed(self.patches):
            setattr(owner, method, original)


def profile_case(name, kind, out):
    import mujoco
    import bushing
    import mj_adapter as A
    import model
    import pinch
    import top_grasp
    import policy_protocol

    out.mkdir()
    os.chdir(out)
    # The frozen archive contains the adapter and the protocol specification;
    # its read-only executor dependency lives in the shared policy directory.
    policy_protocol.SOURCE = ROOT.parents[1] / "policy/env.py"
    side, side_sha = policy_protocol.load_protocol()
    timers = Timings()
    for method in ("mj_step", "mj_step1", "mj_step2", "mj_forward"):
        timers.wrap(mujoco, method, "mujoco")
    timers.wrap(A.Sim, "__init__", "setup")
    timers.wrap(bushing.Bushings, "apply", "bushing")
    timers.wrap(A.Sim, "switch_welds", "compaction")
    for cls in (pinch.PinchAttachment, pinch.PinchSpring):
        for method, func in vars(cls).items():
            if callable(func):
                timers.wrap(cls, method, "pinch")
    k = model.ANCHORS[name]
    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    try:
        result = (top_grasp.evaluate(k, dict(w=.04, d=.015, F=10.), 0) if kind == "top" else
                  side(dict(kappa=k.to_dict(), cls=name, friction=k.surface_friction),
                       (0., 0., np.pi/2, 10., .1, 2.), 0))
    finally:
        profile.disable()
        wall = time.perf_counter() - start
        timers.restore()
    profile.dump_stats(str(out / "calls.pstats"))
    with (out / "calls.txt").open("w") as stream:
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(35)
    times = dict(timers.seconds)
    times["python_other"] = wall - sum(times.values())
    row = dict(anchor=name, kind=kind, wall_s=wall, seconds=times,
               shares={key: value/wall for key, value in times.items()},
               calls=dict(timers.calls), result=result, side_source_sha256=side_sha,
               note="Instrumented single run; cProfile/timer overhead included; not a speed benchmark")
    save(out / "result.json", row)
    print(name, kind, json.dumps(dict(wall_s=wall, shares=row["shares"])), flush=True)
    return row


def force_pilot(out):
    """Identical-state force comparison before testing coupled trajectories."""
    import mujoco
    import model
    import protocols as P
    import warp as wp
    from warp_mechanics import WarpBushings

    rows = []
    wp.init()
    for name in ("bendable", "compressible"):
        sim = P.make(model.ANCHORS[name], [])
        candidate = WarpBushings(sim.desc, sim.seg_bodies, sim.built.bushings)
        for step in range(751):
            if step % 25 == 0:
                sim.d.xfrc_applied[:] = 0.
                sim.bushings.apply(sim.m, sim.d)
                ref = sim.d.xfrc_applied.copy()
                rho = sim.bushings.orthogonal_compaction.copy()
                sim.d.xfrc_applied[:] = 0.
                candidate.apply(sim.m, sim.d)
                fd = float(np.max(np.abs(sim.d.xfrc_applied - ref)))
                rd = float(np.max(np.abs(candidate.orthogonal_compaction-rho)))
                rows.append(dict(anchor=name, step=step, force_max_abs_diff=fd, rho_max_abs_diff=rd))
                sim.d.xfrc_applied[:] = 0.
            if step < 750:
                sim.step()
    save(out / "summary.json", dict(passed=all(max(r["force_max_abs_diff"], r["rho_max_abs_diff"])<=1.e-9 for r in rows),
                                   rows=rows, scope="identical frozen resting states; not full regression"))
    print("force pilot max diff", max(r["force_max_abs_diff"] for r in rows), flush=True)
    return all(max(r["force_max_abs_diff"], r["rho_max_abs_diff"])<=1.e-9 for r in rows)


def regression_manifest():
    jobs = [dict(category=name, test=test) for name in ("hard", "bendable", "compressible")
            for test in ("r1y", "r1z", "r2", "r3", "r4")]
    e4_path = ROOT / "results/pinch_effect_v1_2/v22/e4_formal/summary.json"
    e4 = json.loads(e4_path.read_text())["rows"]
    for name in ("hard", "bendable", "compressible"):
        group = sorted((r for r in e4 if r["category"]==name), key=lambda r:r["index"])
        for index in np.linspace(0, len(group)-1, 10, dtype=int):
            row = group[index]
            jobs.append(dict(category=name, test="e4", kappa=row["kappa"],
                             grasp=row["grasp"], seed=row["seed"], original_index=row["index"]))
    e5_path = ROOT / "results/pinch_effect_v1_2/v22_review/e5_formal/summary.json"
    e5 = sorted(json.loads(e5_path.read_text())["rows"], key=lambda r:r["index"])
    for row in e5[:20]:
        jobs.append(dict(category=row["category"], test="e5", kappa=row["kappa"],
                         grasp=row["grasp"], seed=row["seed"], original_index=row["index"]))
    return jobs


def physical_differences(a, b, path="result"):
    differences = []
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [dict(path=path, reason="different keys")]
        for key in a:
            if key not in ("wall_s", "rtf", "total_wall_s"):
                differences.extend(physical_differences(a[key], b[key], f"{path}.{key}"))
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [dict(path=path, reason="different lengths", reference=len(a), candidate=len(b))]
        for i, (aa, bb) in enumerate(zip(a, b)):
            differences.extend(physical_differences(aa, bb, f"{path}[{i}]"))
    elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
        delta = abs(a-b)
        if not np.isfinite(delta) or delta > 1.e-9:
            differences.append(dict(path=path, reference=a, candidate=b, abs_diff=delta))
    elif a != b:
        differences.append(dict(path=path, reference=a, candidate=b))
    return differences


def regression_run(job, out, candidate):
    import mj_adapter as A
    import model
    import protocols as P
    import top_grasp
    from warp_mechanics import WarpBushings

    out.mkdir()
    os.chdir(out)
    original_class, original_step = A.Bushings, A.Sim.step
    if candidate:
        A.Bushings = WarpBushings
    frames = []
    def observed(sim):
        original_step(sim)
        frames.append(np.concatenate((sim.d.qpos, sim.d.qvel, sim.d.eq_active)))
    A.Sim.step = observed
    k = model.Kappa.from_dict(job["kappa"]) if "kappa" in job else model.ANCHORS[job["category"]]
    try:
        test = job["test"]
        if test.startswith("r1"):
            result = P.r1(k, test[-1])
        elif test == "r2":
            result = P.r2(k)
        elif test == "r3":
            result = P.r3(k)
        elif test == "r4":
            result = top_grasp.measure_r4(k)
        elif test == "e4":
            result = top_grasp.evaluate(k, job["grasp"], job["seed"])
        else:
            result = dict(rest=P.rest(k), grasp=top_grasp.evaluate(k, job["grasp"], job["seed"]))
    finally:
        A.Sim.step, A.Bushings = original_step, original_class
    trace = np.stack(frames)
    save(out / "result.json", result)
    np.savez_compressed(out / "trace.npz", state=trace)
    return result, trace


def force_regression(out):
    jobs = regression_manifest()
    save(out / "manifest.json", dict(tolerance=1.e-9, compare="all-step qpos, qvel, eq_active and physical result fields",
         stage="bushing kernel subgate; compaction/pinch migration is gated by this comparison", jobs=jobs))
    rows = []
    for index, job in enumerate(jobs):
        case = out / f"case_{index:04d}"
        case.mkdir()
        ref, a = regression_run(job, case / "reference", False)
        cand, b = regression_run(job, case / "candidate", True)
        differences = physical_differences(ref, cand)
        shape_match = a.shape == b.shape
        delta = float(np.max(np.abs(a-b))) if shape_match else None
        bitwise = shape_match and a.tobytes() == b.tobytes()
        first_difference = None
        if shape_match:
            exceeding = np.flatnonzero(np.max(np.abs(a-b), axis=1) > 1.e-9)
            if len(exceeding):
                t = int(exceeding[0])
                first_difference = dict(step=t+1, max_abs_diff=float(np.max(np.abs(a[t]-b[t]))))
        row = dict(index=index, job=job, passed=shape_match and delta<=1.e-9 and not differences,
                   bitwise=bitwise, state_max_abs_diff=delta, reference_shape=list(a.shape), candidate_shape=list(b.shape),
                   first_state_difference=first_difference, physical_differences=differences,
                   reference_trace_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                   candidate_trace_sha256=hashlib.sha256(b.tobytes()).hexdigest())
        save(case / "comparison.json", row)
        rows.append(row)
        print(index, job["category"], job["test"], "passed", row["passed"], "state diff", delta, flush=True)
        if not row["passed"]:
            break
        # Successful bitwise cases retain hashes/results; only failing traces
        # need their full arrays for reproduction and diagnosis.
        for folder in ("reference", "candidate"):
            (case / folder / "trace.npz").unlink()
    passed = len(rows)==len(jobs) and all(r["passed"] for r in rows)
    save(out / "summary.json", dict(passed=passed, stage="S2 bushing kernel subgate", rows=rows,
         planned=len(jobs), completed=len(rows), pending=len(jobs)-len(rows), stop_on_first_failure=True))
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("risk", "profile", "force-pilot", "force-regression"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seconds", type=float, default=.3)
    ap.add_argument("--anchors", nargs="+", default=["hard", "bendable", "compressible"])
    args = ap.parse_args()
    BASE.mkdir(exist_ok=True)
    tag, runtime = reference()
    out = BASE / args.run_id
    out.mkdir()
    metadata = dict(before=environment(), arguments=vars(args), frozen_archive_sha256=tag["archive_sha256"],
                    validator_sha256=digest(Path(__file__)), runtime=str(runtime))
    save(out / "metadata.json", metadata)
    os.chdir(out)
    rows = []
    if args.action == "risk":
        import warp as wp
        wp.init()
        with wp.ScopedDevice("cuda:0"):
            for name in args.anchors:
                row = risk_case(name, out / name, args.seconds)
                rows.append(row)
                if not row["passed"]:
                    break
        passed = all(r["passed"] for r in rows) and len(rows) == len(args.anchors)
    elif args.action == "profile":
        for name in args.anchors:
            for kind in ("side", "top"):
                rows.append(profile_case(name, kind, out / f"{name}_{kind}"))
        passed = len(rows) == 2*len(args.anchors)
    elif args.action == "force-pilot":
        passed = force_pilot(out)
    else:
        passed = force_regression(out)
    metadata["after"] = environment()
    metadata["frozen_archive_unchanged"] = digest(RELEASE / "v1.2.tar.gz") == tag["archive_sha256"]
    save(out / "metadata.json", metadata)
    if args.action in ("risk", "profile"):
        save(out / "summary.json", dict(passed=passed, rows=rows,
             throughput_measured=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

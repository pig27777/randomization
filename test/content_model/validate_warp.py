"""Reviewed S2 gates, followed by MJWarp gates only after S2 succeeds."""
from contextlib import nullcontext
import argparse
import json
from multiprocessing import get_context
import os
from pathlib import Path
import tarfile
import traceback

import numpy as np

import sys

CONTENT = Path(__file__).resolve().parents[2] / "contents/segment"
sys.path.insert(0, str(CONTENT))

from validate_acceleration import ROOT, RELEASE, digest, environment, reference, regression_manifest, save

BASE = ROOT / "results/warp_backend_v1_2"


def numeric(a, b):
    return float(np.max(np.abs(np.asarray(a, float)-np.asarray(b, float)))) if np.size(a) else 0.


def operator_checks(out, smoke=False):
    import mujoco
    import model
    import mj_adapter as A
    import protocols as P
    import pinch
    import top_grasp as T
    import warp_cpu as C
    from warp_mechanics import WarpBushings

    rows = []
    def check(name, delta=0., discrete=True, **extra):
        row = dict(name=name, max_abs_diff=float(delta), discrete_equal=bool(discrete),
                   passed=bool(discrete and np.isfinite(delta) and delta<=1.e-12), **extra)
        rows.append(row)
        if not row["passed"]:
            save(out / "summary.json", dict(passed=False, rows=rows))
            raise AssertionError(f"operator gate failed: {row}")

    for name in ("bendable", "compressible"):
        sim = P.make(model.ANCHORS[name], [])
        candidate = WarpBushings(sim.desc, sim.seg_bodies, sim.built.bushings)
        count = 1 if smoke else 751
        for step in range(count):
            if step % 25 == 0:
                sim.d.xfrc_applied[:] = 0.
                sim.bushings.apply(sim.m, sim.d)
                force, rho = sim.d.xfrc_applied.copy(), sim.bushings.orthogonal_compaction.copy()
                sim.d.xfrc_applied[:] = 0.
                candidate.apply(sim.m, sim.d)
                check("bushing", max(numeric(force, sim.d.xfrc_applied), numeric(rho, candidate.orthogonal_compaction)),
                      anchor=name, step=step)
                sim.d.xfrc_applied[:] = 0.
                eq, active, filtered = sim.m.eq_data.copy(), sim.d.eq_active.copy(), sim.external_compaction_load
                external = np.zeros_like(sim.d.xfrc_applied)
                # Include native contact forces and a known opposing external load.
                external[sim.seg_bodies[0], 0] = 2.
                external[sim.seg_bodies[-1], 0] = -2.
                A.Sim.switch_welds(sim, external)
                ref = sim.m.eq_data.copy(), sim.d.eq_active.copy(), sim.external_compaction_load, sim.external_compaction_load_raw
                sim.m.eq_data[:], sim.d.eq_active[:], sim.external_compaction_load = eq, active, filtered
                C.switch_welds(sim, external)
                check("compaction_native_state", max(numeric(ref[0], sim.m.eq_data),
                      abs(ref[2]-sim.external_compaction_load), abs(ref[3]-sim.external_compaction_load_raw)),
                      np.array_equal(ref[1], sim.d.eq_active), anchor=name, step=step)
                sim.m.eq_data[:], sim.d.eq_active[:], sim.external_compaction_load = eq, active, filtered
            if step+1 < count:
                sim.step()
        # Canonical threshold cases, with nontrivial relative orientations.
        sim = P.make(model.ANCHORS[name], [], table=False, gravity=False)
        rng = np.random.default_rng(20261107)
        for address in sim._seg_qadr:
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            sim.d.qpos[address+3:address+7] = q
        mujoco.mj_forward(sim.m, sim.d)
        thresholds = np.array([A.WELD_ON-2.e-12, A.WELD_ON, A.WELD_ON+2.e-12,
                               A.WELD_RELEASE-2.e-12, A.WELD_RELEASE, A.WELD_RELEASE+2.e-12])
        sim.bushings.orthogonal_compaction = np.resize(thresholds, len(sim.welds))
        for load in (A.COMPACTION_LOAD_MIN-2.e-12, A.COMPACTION_LOAD_MIN, A.COMPACTION_LOAD_MIN+2.e-12, 2.):
            for previous_on in (False, True):
                sim.d.eq_active[sim.welds] = previous_on
                eq, active = sim.m.eq_data.copy(), sim.d.eq_active.copy()
                external = np.zeros_like(sim.d.xfrc_applied)
                external[sim.seg_bodies[0], 0], external[sim.seg_bodies[-1], 0] = load, -load
                sim.external_compaction_load = load
                A.Sim.switch_welds(sim, external)
                ref = sim.m.eq_data.copy(), sim.d.eq_active.copy(), sim.external_compaction_load, sim.external_compaction_load_raw
                sim.m.eq_data[:], sim.d.eq_active[:], sim.external_compaction_load = eq, active, load
                C.switch_welds(sim, external)
                check("compaction_threshold_pose", max(numeric(ref[0], sim.m.eq_data),
                      abs(ref[2]-sim.external_compaction_load), abs(ref[3]-sim.external_compaction_load_raw)),
                      np.array_equal(ref[1], sim.d.eq_active), anchor=name, load=load, previous_on=previous_on)

    for name in ("hard", "bendable", "compressible"):
        sim, starts, original_top = T.task_scene(model.ANCHORS[name], T.TopGrasp(), 0.)
        ref, cand = pinch.PinchAttachment(sim, original_top), C.WarpPinchAttachment(sim, original_top)
        for descent in (.002999999998, .003, .003000000002, .008):
            for p, start in starts.items():
                sim.set_pad(p, start-[0, 0, descent], np.zeros(3))
            mujoco.mj_forward(sim.m, sim.d)
            tips = np.array([sim.pad_pos(p)-[0, 0, T.HEIGHT/2] for p in starts])
            check("pinch_contacts", discrete=ref.top_contacts()==cand.top_contacts(), anchor=name, descent=descent)
            check("pinch_candidate", discrete=ref.candidate(tips)==cand.candidate(tips), anchor=name, descent=descent)
        # Keep actual top contacts fixed while probing the depth qualification
        # itself: fingertip starting clearance is 5mm, so descent != depth.
        for offset in (-2.e-12, 0., 2.e-12):
            probe = tips.copy()
            probe[:, 2] = original_top-pinch.DEPTH_MIN-offset
            check("pinch_depth_threshold", discrete=ref.candidate(probe)==cand.candidate(probe),
                  anchor=name, depth_offset=offset)
        saved_force = sim.d.efc_force.copy()
        addresses = sim.d.contact.efc_address
        valid_addresses = addresses[addresses>=0]
        for force_value in (.05-2.e-12, .05, .05+2.e-12):
            sim.d.efc_force[valid_addresses] = force_value
            check("pinch_contact_threshold", discrete=ref.top_contacts()==cand.top_contacts(),
                  anchor=name, contact_force=force_value)
        sim.d.efc_force[:] = saved_force
        if not ref.top:
            continue
        # Rotated mocap frame exercises both position and quaternion outputs.
        q = np.array([.8, .2, -.3, .4]); q /= np.linalg.norm(q)
        ref.set_frame(np.array([.01, .02, .03]), q)
        i = ref.top[len(ref.top)//2]
        ref.bind_segment(i)
        eq, active = sim.m.eq_data.copy(), sim.d.eq_active.copy()
        sim.d.eq_active[:] = False
        cand.bind_segment(i)
        check("pinch_bind_pose", numeric(eq, sim.m.eq_data), np.array_equal(active, sim.d.eq_active), anchor=name)
        mujoco.mj_forward(sim.m, sim.d)
        check("pinch_reaction", numeric(ref.force_vector(), cand.force_vector()), anchor=name)
        # Force exactly at, below, and above the current capacity; safe readings
        # reset the continuity timer, and a final uninterrupted 20ms releases.
        dt = sim.m.opt.timestep
        grip = 10.
        force = np.zeros(3)
        ref.force_vector = lambda: force.copy()
        cand.force_vector = lambda: force.copy()
        ratios = [.9, 1., 1.+1.e-10]*2 + [1.1]*49 + [1.] + [1.1]*50
        for step, ratio in enumerate(ratios):
            force[2] = ratio*ref.coefficient*grip
            ra, rb = ref.check_load(dt, grip), cand.check_load(dt, grip)
            event_delta = 0.
            if ref.events[-1]["event"] == "overload":
                event_delta = max(abs(ref.events[-1][key]-cand.events[-1][key]) for key in ("force", "limit", "overload_s"))
            check("pinch_capacity_continuity", max(event_delta, abs(ref.overload_s-cand.overload_s),
                  abs(ref.peak_force-cand.peak_force)), ra==rb and ref.active==cand.active and ref.broken==cand.broken,
                  anchor=name, step=step, capacity_N=ref.coefficient*grip, force_N=float(force[2]))
        ref.release(); cand.release()
        check("pinch_release", discrete=ref.active==cand.active and ref.events[-1]==cand.events[-1], anchor=name)

    ref, cand = pinch.PinchSpring(), C.WarpPinchSpring()
    for eligible in (False, True):
        for gap0 in (.020, .040, .095):
            ref.arm(gap0, eligible); cand.arm(gap0, eligible)
            for gap in (gap0, gap0-.010+1.e-12, gap0-.010, gap0-.010-1.e-12, gap0-.015):
                check("spring", abs(ref.force(gap)-cand.force(gap)), eligible=eligible, reference_gap=gap0, gap=gap)
            ref.release(); cand.release()
            check("spring_release", abs(ref.force(0.)-cand.force(0.)))
    save(out / "summary.json", dict(passed=True, tolerance=1.e-12, rows=rows,
         cases=len(rows), max_abs_diff=max(r["max_abs_diff"] for r in rows)))
    print("operators passed", len(rows), "max diff", max(r["max_abs_diff"] for r in rows), flush=True)
    return True


def execute(job, candidate, trace=False):
    import model
    import protocols as P
    import top_grasp as T
    import warp_cpu as C
    k = model.Kappa.from_dict(job["kappa"]) if "kappa" in job else model.ANCHORS[job["category"]]
    with C.enabled() if candidate else nullcontext():
        test = job["test"]
        if test.startswith("r1"):
            return P.r1(k, test[-1])
        if test == "r2":
            return P.r2(k)
        if test == "r3":
            return P.r3(k)
        if test == "r4":
            return T.measure_r4(k)
        if test == "e4":
            return T.evaluate(k, job["grasp"], job["seed"], trace=trace)
        return dict(rest=P.rest(k), grasp=T.evaluate(k, job["grasp"], job["seed"], trace=trace))


def response_comparison(job, a, b, tolerance=.01):
    test = job["test"]
    metrics = (["stiffness", "lock_strain"] if test.startswith("r1") else
               ["droop"] if test == "r2" else
               ["slip_mm", "com_swing_mm", "com_decay_s", "force_p2p"] if test == "r3" else
               ["measured_pinch_coeff"])
    rows = []
    for key in metrics:
        av, bv = a.get(key), b.get(key)
        error = None
        delta = nominal = allowed = None
        resolution = {"lock_strain": .0025, "com_decay_s": .01}.get(key, 0.)
        resolution_used = False
        record_only = bool(job["category"]=="hard" and test.startswith("r1") and key=="stiffness"
                           and a.get("lower_bound") and b.get("lower_bound"))
        if av is None or bv is None:
            passed = av is None and bv is None
        else:
            delta = abs(bv-av)
            nominal = tolerance*abs(av) if av else 1.e-12
            allowed = max(nominal, resolution)
            # A mathematically exact one-grid difference can round just above
            # its grid size. This is machine precision, not physical slack.
            at_boundary = np.isclose(delta, allowed, rtol=8*np.finfo(float).eps, atol=0.)
            passed = bool(np.isfinite(delta) and (delta<=allowed or at_boundary))
            error = delta/abs(av) if av else delta
            resolution_used = passed and delta>nominal and resolution>nominal
        if record_only:
            passed = True
        rows.append(dict(metric=key, reference=av, candidate=bv, relative_error=error, abs_diff=delta,
                         nominal_absolute_tolerance=nominal, grid_step=resolution,
                         effective_absolute_tolerance=allowed, resolution_used=resolution_used,
                         record_only=record_only,
                         exclusion_reason="reviewed hard-object capped stiffness lower bound" if record_only else None,
                         passed=passed))
    flags = {key:dict(reference=a.get(key), candidate=b.get(key), passed=a.get(key)==b.get(key))
             for key in ("fail", "capped", "lower_bound", "slipped", "dropped", "suspended", "table_touch", "outcome")
             if key in a or key in b}
    return dict(passed=all(r["passed"] for r in rows) and all(v["passed"] for v in flags.values()), metrics=rows, flags=flags)


def gpu_first_review(out):
    previous = BASE/"s4_gpu_responses"
    meta = json.loads((previous/"metadata.json").read_text())
    mechanics = ("mjwarp_backend.py","warp_mechanics.py","warp_effects.py","warp_cpu.py")
    if any(digest(ROOT/name)!=meta["source_sha256"][name] for name in mechanics):
        raise RuntimeError("GPU mechanics changed; saved first response cannot be reused")
    case = previous/"hard_r1y"
    a = json.loads((case/"reference.json").read_text())
    b = json.loads((case/"candidate.json").read_text())
    comparison = response_comparison(dict(category="hard",test="r1y"),a,b,tolerance=.02)
    save(out/"summary.json",dict(simulation_runs=0,source=str(case),
         reference_sha256=digest(case/"reference.json"),candidate_sha256=digest(case/"candidate.json"),
         hard_task_layer_not_yet_validated=True,**comparison))
    print("hard R1y lower bound recorded only; reviewed",comparison["passed"],flush=True)
    return comparison["passed"]


def reassess(out):
    previous = BASE/"s2_responses"
    metadata = json.loads((previous/"metadata.json").read_text())
    mechanics = ("warp_mechanics.py", "warp_effects.py", "warp_cpu.py")
    if any(digest(ROOT/name)!=metadata["source_sha256"][name] for name in mechanics):
        raise RuntimeError("candidate mechanics changed; saved response results cannot be reused")
    original = json.loads((previous/"summary.json").read_text())
    rows = []
    for row in original["rows"]:
        case = previous/f"case_{row['index']:04d}"
        a = json.loads((case/"reference.json").read_text())
        b = json.loads((case/"candidate.json").read_text())
        rows.append(dict(index=row["index"], job=row["job"], source=str(case),
                    reference_sha256=digest(case/"reference.json"), candidate_sha256=digest(case/"candidate.json"),
                    **response_comparison(row["job"],a,b)))
    passed = all(row["passed"] for row in rows)
    save(out/"summary.json",dict(passed=passed, completed=len(rows), simulation_runs=0,
         reviewed_numeric_responses_accepted=True, unexecuted_original_configurations=original["planned"]-len(rows), rows=rows))
    print("saved responses reassessed", len(rows), "passed", passed, flush=True)
    return passed


def responses(out):
    jobs = regression_manifest()[:15]
    save(out / "manifest.json", dict(jobs=jobs, tolerance_relative=.01, trajectory_comparison=False))
    rows = []
    for index, job in enumerate(jobs):
        case = out / f"case_{index:04d}"
        case.mkdir(); os.chdir(case)
        a = execute(job, False); save(case/"reference.json", a)
        b = execute(job, True); save(case/"candidate.json", b)
        row = dict(index=index, job=job, **response_comparison(job, a, b))
        rows.append(row); save(case/"comparison.json", row)
        print(index, job["category"], job["test"], "passed", row["passed"], json.dumps(row["metrics"]), flush=True)
        if not row["passed"]:
            break
    passed = len(rows)==len(jobs) and all(r["passed"] for r in rows)
    save(out/"summary.json", dict(passed=passed, planned=len(jobs), completed=len(rows), rows=rows))
    return passed


def critical_evidence(result, kappa):
    import top_grasp as T
    threshold = T.THRESHOLD[result["cls"]]
    capacity = []
    for sample in result["trace"]:
        if sample["active"]:
            margin = kappa["pinch_coeff"]*sample["grip_force"]-float(np.linalg.norm(sample["weld_force"]))
            capacity.append(dict(t=sample["t"], phase=sample["phase"], margin_N=margin))
    nearest = min(capacity,key=lambda r:abs(r["margin_N"])) if capacity else None
    return dict(outcome=result["outcome"], reason=result["reason"],
                over_squeeze_margin_N=threshold-result["max_grip_force_N"],
                pinch_events=result["pinch_events"], nearest_recorded_capacity_margin=nearest,
                actual_depth_m=result["actual_depth_m"], press_stop=result["press_stop"],
                note="10ms sampled margins describe proximity; they do not prove the cause of a branch difference")


def classification_pair(payload):
    index, job, folder = payload
    case = Path(folder); case.mkdir(); os.chdir(case)
    try:
        import model
        kappa = job.get("kappa", model.ANCHORS[job["category"]].to_dict())
        a = execute(job, False, trace=True); save(case/"reference.json", a)
        b = execute(job, True, trace=True); save(case/"candidate.json", b)
        aa, bb = (a["grasp"], b["grasp"]) if job["test"]=="e5" else (a, b)
        row = dict(index=index, job=job, reference=aa["outcome"], candidate=bb["outcome"],
                   agreed=aa["outcome"]==bb["outcome"], execution_error=False)
        if not row["agreed"]:
            row["boundary_evidence"] = dict(reference=critical_evidence(aa,kappa), candidate=critical_evidence(bb,kappa))
    except Exception:
        row = dict(index=index, job=job, agreed=False, execution_error=True, traceback=traceback.format_exc())
    save(case/"comparison.json", row)
    return row


def classifications(out, workers, smoke=False):
    jobs = []
    for original in regression_manifest()[15:]:
        for repeat in range(3):
            jobs.append(dict(**{key:value for key,value in original.items() if key!="seed"},
                             seed=original["seed"]+repeat, repeat=repeat))
    if smoke:
        jobs = [next(job for job in jobs if job["category"]==name and job["test"]=="e4")
                for name in ("hard", "bendable", "compressible")]
    save(out/"manifest.json", dict(jobs=jobs, required_agreement=.95, workers=workers, entry_smoke=smoke))
    rows, groups = [], {}
    for kind in ("e4", "e5"):
        payloads = [(index,job,str(out/f"case_{index:04d}")) for index,job in enumerate(jobs) if job["test"]==kind]
        if not payloads:
            continue
        part = []
        with get_context("spawn").Pool(workers) as pool:
            for row in pool.imap_unordered(classification_pair,payloads):
                rows.append(row); part.append(row)
                print(row["index"],kind,row.get("reference"),row.get("candidate"),
                      "error",row["execution_error"],flush=True)
                mismatches = sum(not r["agreed"] for r in part)
                upper = (len(payloads)-mismatches)/len(payloads)
                impossible = not smoke and upper<.95
                if row["execution_error"] or impossible:
                    pool.terminate()
                    groups[kind] = dict(planned=len(payloads), completed=len(part), mismatches=mismatches,
                         maximum_possible_agreement=upper, passed=False, early_stop=True)
                    break
        if kind not in groups:
            rate = sum(r["agreed"] for r in part)/len(part)
            groups[kind] = dict(planned=len(payloads),completed=len(part),agreement=rate,
                                passed=rate>=.95 if not smoke else not any(r["execution_error"] for r in part))
        save(out/"summary.json",dict(passed=False,groups=groups,completed=len(rows),planned=len(jobs),
            mismatches=[r for r in rows if not r["agreed"]],rows=sorted(rows,key=lambda r:r["index"])))
        if not groups[kind]["passed"]:
            break
    passed = all(g["passed"] for g in groups.values()) and len(rows)==len(jobs)
    save(out/"summary.json", dict(passed=passed, groups=groups, completed=len(rows),planned=len(jobs),
         mismatches=[r for r in rows if not r["agreed"]],rows=sorted(rows,key=lambda r:r["index"])))
    return passed


def gpu_rest(out, seconds, nworld, anchors):
    import mujoco_warp as mjw
    import warp as wp
    import model
    import protocols as P
    from mjwarp_backend import BatchStepper
    from validate_acceleration import state
    rows = []
    for name in anchors:
        case = out/name; case.mkdir()
        cpu, host = P.make(model.ANCHORS[name],[]), P.make(model.ANCHORS[name],[])
        (case/"model.xml").write_text(host.built.xml)
        dt = host.m.opt.timestep
        checkpoint = round(.01/dt)
        if round(seconds/dt)%checkpoint:
            raise ValueError("GPU resting duration must be a multiple of the original 10ms check interval")
        with wp.ScopedDevice("cuda:0"):
            warm = BatchStepper(host,nworld)
            warm.step(); wp.synchronize()
            del warm
            batch = BatchStepper(host,nworld)
            with wp.ScopedCapture() as capture:
                for _ in range(checkpoint):
                    batch.step()
            samples, failure = [], None
            for step in range(checkpoint,round(seconds/dt)+1,checkpoint):
                for _ in range(checkpoint):
                    cpu.step()
                wp.capture_launch(capture.graph)
                overflow = int(batch.overflow.numpy()[0])
                if overflow or int(batch.data.nacon.numpy()[0])>batch.data.naconmax or np.any(batch.data.nefc.numpy()>batch.data.njmax):
                    raise RuntimeError("GPU contact/constraint buffer overflow; no physical result is accepted")
                gpu_failures, maximum = [], 0.
                for env in range(nworld):
                    mjw.get_data_into(host.d,host.m,batch.data,world_id=env)
                    error = host.failure()
                    gpu_failures.append(error)
                    if np.isfinite(host.d.qvel).all():
                        maximum = max(maximum,float(np.max(np.abs(host.d.qvel))))
                    if error and failure is None:
                        failure = dict(world=env,step=step,t=step*dt,reason=error,state=state(host))
                cpu_error = cpu.failure()
                row = dict(step=step,t=step*dt,cpu_failure=cpu_error,gpu_failures=gpu_failures,
                           gpu_max_abs_qvel=maximum,active_welds=batch.data.eq_active.numpy().sum(axis=1).tolist())
                samples.append(row)
                if failure or cpu_error:
                    print(name,"rest failure",json.dumps(row),flush=True)
                    break
            row = dict(anchor=name,passed=failure is None and cpu_error is None, nworld=nworld,
                       simulation_seconds=step*dt,requested_seconds=seconds,timestep=dt,
                       damped_tendons=int(np.count_nonzero(host.m.tendon_damping)),
                       first_gpu_failure=failure,samples=samples,
                       backend="native MJWarp step1/2, shared float64 Warp face forces and compaction; no CPU force bridge",
                       scope="resting only; batched grasp protocol is not yet validated")
            save(case/"result.json",row); rows.append(row)
            print(name,"GPU rest passed",row["passed"],"seconds",row["simulation_seconds"],flush=True)
            if not row["passed"]:
                break
    passed = len(rows)==len(anchors) and all(row["passed"] for row in rows)
    save(out/"summary.json",dict(passed=passed,rows=rows,remaining_anchors=anchors[len(rows):],throughput_measured=False))
    return passed


def gpu_readback(out):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    from mjwarp_backend import synchronize_contact_ids
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1"/>'
        '<body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>'
        '</worldbody></mujoco>')
    initial = mujoco.MjData(m)
    mujoco.mj_forward(m,initial)
    result = mujoco.MjData(m)
    with wp.ScopedDevice("cuda:0"):
        gm = mjw.put_model(m)
        gd = mjw.put_data(m,initial,nworld=1,nconmax=16,njmax=64)
        mjw.step(gm,gd)
        mjw.get_data_into(result,m,gd)
    modern = result.contact.geom.copy()
    before = np.column_stack((result.contact.geom1,result.contact.geom2))
    synchronize_contact_ids(result)
    after = np.column_stack((result.contact.geom1,result.contact.geom2))
    passed = result.ncon>0 and np.array_equal(modern,after)
    row = dict(passed=bool(passed),ncon=result.ncon,modern=modern.tolist(),
               legacy_before=before.tolist(),legacy_after=after.tolist(),
               reproduced_stale_aliases=not np.array_equal(modern,before))
    save(out/"summary.json",row)
    print("native GPU contact readback",json.dumps(row),flush=True)
    return passed


def gpu_responses(out, anchors, tests):
    import model
    import protocols as P
    from mjwarp_backend import DiagnosticSim
    rows = []
    for name in anchors:
        for test in tests:
            case = out/f"{name}_{test}"; case.mkdir(); os.chdir(case)
            job = dict(category=name,test=test)
            baseline = execute(job,False); save(case/"reference.json",baseline)
            original = P.Sim
            holder = []
            class ObservedSim(DiagnosticSim):
                def __init__(self,*args,**kwargs):
                    super().__init__(*args,**kwargs)
                    holder.append(self)
                    self.observed_steps = 0
                def step(self):
                    super().step()
                    self.observed_steps += 1
                    if self.observed_steps%2500==0:
                        print(name,test,"GPU diagnostic steps",self.observed_steps,
                              "max_abs_qvel",float(np.max(np.abs(self.d.qvel))),flush=True)
            P.Sim = ObservedSim
            try:
                k = model.ANCHORS[name]
                candidate = P.r1(k,test[-1]) if test.startswith("r1") else P.r2(k) if test=="r2" else P.r3(k)
            finally:
                P.Sim = original
            # Preserve invalid simulation numbers as explicit JSON strings;
            # they remain failures and must not become an encoding exception.
            recorded = json.loads(json.dumps(candidate),parse_constant=lambda value:value)
            save(case/"candidate.json",recorded)
            row = dict(job=job,**response_comparison(job,baseline,candidate,tolerance=.02),
                       scope="closed-loop GPU R1--R3 using frozen host controller; not batched grasp throughput")
            if baseline.get("fail") or candidate.get("fail"):
                row["passed"] = False
                row["confirmed_failure"] = dict(reference=baseline.get("fail"),candidate=candidate.get("fail"))
            row = json.loads(json.dumps(row),parse_constant=lambda value:value)
            save(case/"comparison.json",row); rows.append(row)
            (case/"model.xml").write_text(holder[-1].built.xml)
            print(name,test,"GPU response passed",row["passed"],json.dumps(row["metrics"]),flush=True)
            save(out/"summary.json",dict(passed=False,rows=rows,planned=len(anchors)*len(tests),completed=len(rows)))
            if not row["passed"]:
                return False
    save(out/"summary.json",dict(passed=True,rows=rows,planned=len(anchors)*len(tests),completed=len(rows)))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("smoke", "operators", "responses", "reassess", "classes-smoke", "classes", "gpu-rest", "gpu-readback", "gpu-responses", "gpu-first-review"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seconds",type=float,default=2.)
    ap.add_argument("--nworld",type=int,default=2)
    ap.add_argument("--anchors",nargs="+",default=["hard","bendable","compressible"])
    ap.add_argument("--tests",nargs="+",default=["r1y","r1z","r2","r3"])
    args = ap.parse_args()
    tag, runtime = reference()
    BASE.mkdir(exist_ok=True)
    out = BASE/args.run_id; out.mkdir(); os.chdir(out)
    files = ("warp_mechanics.py", "warp_effects.py", "warp_cpu.py")
    if args.action.startswith("gpu-"):
        files += ("mjwarp_backend.py",)
    sources = {name: ROOT / name for name in files}
    sources["test/content_model/validate_warp.py"] = Path(__file__).resolve()
    metadata = dict(before=environment(), runtime=str(runtime), arguments=vars(args),
                    frozen_archive_sha256=tag["archive_sha256"],
                    source_sha256={name:digest(path) for name,path in sources.items()})
    save(out/"metadata.json", metadata)
    with tarfile.open(out/"source.tar.gz","w:gz") as archive:
        for name,path in sources.items():
            archive.add(path,arcname=name)
    try:
        if args.action in ("smoke", "operators"):
            passed = operator_checks(out, args.action=="smoke")
        elif args.action=="responses":
            passed = responses(out)
        elif args.action=="reassess":
            passed = reassess(out)
        elif args.action=="gpu-rest":
            passed = gpu_rest(out,args.seconds,args.nworld,args.anchors)
        elif args.action=="gpu-responses":
            passed = gpu_responses(out,args.anchors,args.tests)
        elif args.action=="gpu-readback":
            passed = gpu_readback(out)
        elif args.action=="gpu-first-review":
            passed = gpu_first_review(out)
        else:
            passed = classifications(out,args.workers,args.action=="classes-smoke")
    except Exception:
        save(out/"execution_error.json", dict(traceback=traceback.format_exc()))
        raise
    finally:
        metadata["after"] = environment()
        metadata["frozen_archive_unchanged"] = digest(RELEASE/"v1.2.tar.gz")==tag["archive_sha256"]
        save(out/"metadata.json", metadata)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

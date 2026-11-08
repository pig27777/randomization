"""Audit saved acceleration evidence only; never advances a simulation."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SEGMENT = ROOT / "contents/segment"


def read(relative):
    return json.loads((HERE / relative).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_usage(metadata):
    before, after = metadata["before"], metadata["after"]
    a = np.array([int(v) for v in before["proc_stat"].split()[1:9]])
    b = np.array([int(v) for v in after["proc_stat"].split()[1:9]])
    delta = b-a
    busy = 1-float(delta[3]+delta[4])/float(delta.sum())
    return dict(start=before["time"], end=after["time"],
                loadavg_before=before["loadavg"], loadavg_after=after["loadavg"],
                system_busy_percent=busy*100, average_busy_logical_cpus=busy*len(before["affinity"]))


def main():
    reg = read("s2_bushing_regression/summary.json")
    failed = reg["rows"][-1]
    case = HERE / f"s2_bushing_regression/case_{failed['index']:04d}"
    a = np.load(case / "reference/trace.npz")["state"]
    b = np.load(case / "candidate/trace.npz")["state"]
    common = min(len(a), len(b))
    diff = np.abs(a[:common]-b[:common])
    maximum = diff.max(axis=1)
    first_exact = int(np.flatnonzero(maximum>0)[0])
    first_exceed = int(np.flatnonzero(maximum>1.e-9)[0])
    # R1-y: 36 free segment joints and 6 prescribed pad slide joints.
    nq, nv = 36*7+6, 36*6+6
    trace = dict(common_steps=common, reference_steps=len(a), candidate_steps=len(b),
                 first_nonzero_step=first_exact+1, first_nonzero_max_abs_diff=float(maximum[first_exact]),
                 first_exceeding_step=first_exceed+1, first_exceeding_t=(first_exceed+1)*.0004,
                 first_exceeding_max_abs_diff=float(maximum[first_exceed]),
                 common_prefix_max_abs_diff=float(maximum.max()),
                 qpos_max_abs_diff=float(diff[:, :nq].max()), qvel_max_abs_diff=float(diff[:, nq:nq+nv].max()),
                 eq_active_different_steps=int(np.count_nonzero(np.any(diff[:, nq+nv:] != 0, axis=1))))
    profile = read("s1_profile/summary.json")
    tag_path = SEGMENT / "results/pinch_effect_v1_2/v22_review/release_v1_2/v1.2.json"
    tag = json.loads(tag_path.read_text())
    active_changes = [name for name, digest in tag["source_sha256"].items()
                      if sha(ROOT/name) != digest]
    dependency = ROOT / "policy/env.py"
    dependency_hashes = {row["side_source_sha256"] for row in profile["rows"]}
    dependencies = HERE / "dependencies"
    dependencies.mkdir(exist_ok=True)
    if dependency_hashes == {sha(dependency)}:
        shutil.copyfile(dependency, dependencies / "policy_env.py")
    risk = read("risk_formal_checked/summary.json")
    result = dict(S1=dict(passed=True, rows=[dict(anchor=r["anchor"], kind=r["kind"],
                 wall_s=r["wall_s"], seconds=r["seconds"], shares=r["shares"],
                 outcome=r["result"].get("outcome"), sim_s=r["result"].get("sim_s")) for r in profile["rows"]]),
                 risk=dict(short_rest_passed=risk["passed"], simulation_seconds=.3,
                           native_gpu_backend_validated=False,
                           rows=[{key:r[key] for key in ("anchor", "damped_tendons", "steps", "ntendon", "neq", "passed")}
                                 for r in risk["rows"]]),
                 S2=dict(passed=False, completed=reg["completed"], planned=reg["planned"],
                         hard_bitwise_cases=sum(r["bitwise"] for r in reg["rows"]),
                         failing_case=failed["job"], trace=trace,
                         stiffness_reference=.5011508539825948, stiffness_candidate=.49658196416970884,
                         stiffness_abs_diff=.004568889812886001,
                         stiffness_relative_diff=.49658196416970884/.5011508539825948-1),
                 S3="not run: S2 failed; CPU speedup not established",
                 S4="not run: only the short resting diagnostic is complete",
                 S5="not run", S6="not run; no evaluate throughput claim", released_v1_2_1=False,
                 load={stage:cpu_usage(read(stage+"/metadata.json")) for stage in
                       ("risk_formal_checked", "s1_profile", "s2_bushing_regression")},
                 archive_unchanged=sha(tag_path.parent / "v1.2.tar.gz")==tag["archive_sha256"],
                 frozen_archive_sha256=tag["archive_sha256"], active_frozen_files_changed=active_changes,
                 side_dependency_unchanged=dependency_hashes=={sha(dependency)},
                 source_sha256={name:sha(SEGMENT/name) for name in ("validate_acceleration.py", "warp_mechanics.py")})
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(trace=trace, load=result["load"], archive_unchanged=result["archive_unchanged"],
                         active_frozen_files_changed=active_changes), indent=2))


if __name__ == "__main__":
    main()

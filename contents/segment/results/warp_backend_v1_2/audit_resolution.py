"""Audit the resolution-reviewed S2 and GPU-entry evidence; no simulation steps."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
SEGMENT = HERE.parents[1]
PROJECT = SEGMENT.parents[1]


def read(path):
    return json.loads((HERE/path).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usage(meta):
    a,b = meta["before"],meta["after"]
    av = [int(v) for v in a["proc_stat"].split()[1:9]]
    bv = [int(v) for v in b["proc_stat"].split()[1:9]]
    d = [y-x for x,y in zip(av,bv)]
    busy = 1-(d[3]+d[4])/sum(d)
    return dict(start=a["time"],end=b["time"],loadavg_before=a["loadavg"],loadavg_after=b["loadavg"],
                busy_percent=100*busy,average_busy_logical_cpus=32*busy)


def main():
    classes = read("s2_classes/summary.json")
    review = read("resolution_review/summary.json")
    rest = read("s4_gpu_rest/summary.json")
    response = read("s4_gpu_responses/summary.json")
    stages = ("resolution_review","classes_smoke","s2_classes","gpu_rest_smoke","s4_gpu_rest","s4_gpu_responses")
    provenance = {}
    for stage in stages:
        meta = read(stage+"/metadata.json")
        with tarfile.open(HERE/stage/"source.tar.gz") as archive:
            matches = {name:hashlib.sha256(archive.extractfile(name).read()).hexdigest()==value
                       for name,value in meta["source_sha256"].items()}
        provenance[stage] = dict(source_snapshot_matches=matches,load=usage(meta))
        if not all(matches.values()):
            raise RuntimeError(f"source snapshot differs from recorded source: {stage}")
    group_counts = {}
    for kind in ("e4","e5"):
        part = [row for row in classes["rows"] if row["job"]["test"]==kind]
        group_counts[kind] = dict(groups=len(part)//3,paired_runs=len(part),matches=sum(row["agreed"] for row in part),
            agreement=sum(row["agreed"] for row in part)/len(part),
            reference_outcomes=dict(Counter(row["reference"] for row in part)),
            candidate_outcomes=dict(Counter(row["candidate"] for row in part)))
        if kind=="e5":
            rest_failures = {}
            for version in ("reference","candidate"):
                rest_failures[version] = [row["index"] for row in part if
                    read(f"s2_classes/case_{row['index']:04d}/{version}.json")["rest"].get("fail")]
            group_counts[kind]["rest_failures"] = rest_failures
    release = SEGMENT/"results/pinch_effect_v1_2/v22_review/release_v1_2"
    tag = json.loads((release/"v1.2.json").read_text())
    changed = [name for name,value in tag["source_sha256"].items() if digest(PROJECT/name)!=value]
    runtime = SEGMENT/"results/cpu_equivalence_v1_2_1/reference_v1_2/contents/segment"
    sys.path.insert(0,str(runtime))
    import mujoco
    import model
    import protocols as P
    import mj_adapter as A
    # Rebuild only the static model, without stepping it, to check identical XML.
    k = model.ANCHORS["hard"]
    pads = [("pA",1,(0,-(k.size[1]/2+P.GAP+A.PAD_T),k.size[2]/2)),
            ("pB",1,(0,(k.size[1]/2+P.GAP+A.PAD_T),k.size[2]/2))]
    expected = P.make(k,pads)
    xml = (HERE/"s4_gpu_responses/hard_r1y/model.xml").read_text()
    m = mujoco.MjModel.from_xml_string(xml)
    config = dict(same_frozen_xml=expected.built.xml==xml,ntendon=m.ntendon,neq=m.neq,nbody=m.nbody,
        timestep=float(m.opt.timestep),solver=int(m.opt.solver),integrator=int(m.opt.integrator),
        iterations=int(m.opt.iterations),ls_iterations=int(m.opt.ls_iterations),tolerance=float(m.opt.tolerance))
    files = ("validate_warp.py","mjwarp_backend.py","warp_mechanics.py","warp_effects.py","warp_cpu.py")
    for name in files:
        ast.parse((SEGMENT/name).read_text())
    last = read("s4_gpu_responses/metadata.json")
    current = {name:digest(SEGMENT/name)==value for name,value in last["source_sha256"].items()}
    result = dict(S2=dict(reviewed_numeric_responses_accepted=review["passed"],
        originally_executed_numeric_cases=review["completed"], originally_unexecuted_numeric_cases=8,
        classifications_passed=classes["passed"],classes=group_counts,
        execution_errors=sum(row["execution_error"] for row in classes["rows"]),mismatches=classes["mismatches"]),
        S4=dict(resting_passed=rest["passed"],rest_worlds_per_anchor=2,rest_seconds=2,
        full_protocol_v22_and_remaining_stability_tests="not completed; numerical response gate failed"),
        S5_entry=dict(passed=response["passed"],completed=response["completed"],failure=response["rows"][0]),
        S6="not run; no evaluate/s or peak-VRAM result",candidate_training_ready=False,
        hard_r1y_model=config,provenance=provenance,
        frozen_archive_unchanged=digest(release/"v1.2.tar.gz")==tag["archive_sha256"],
        original_files_changed=changed,latest_sources_match=current,source_syntax_passed=True)
    (HERE/"audit_resolution.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(classifications=group_counts,hard_r1y_model=config,
        frozen_archive_unchanged=result["frozen_archive_unchanged"],original_files_changed=changed,
        latest_sources_match=current,load={stage:value["load"] for stage,value in provenance.items()}),indent=2))


if __name__=="__main__":
    main()

"""Audit reviewed GPU responses and the contact-readback repair; no stepping."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile

from audit_resolution import usage

HERE = Path(__file__).resolve().parent
SEGMENT = HERE.parents[1]
PROJECT = SEGMENT.parents[1]


def read(path):
    return json.loads((HERE/path).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    stages = ["hard_lower_bound_review", "s5_soft_responses", "s5_readback_repair",
              "s5_bendable_readback_fixed", "s5_compressible_responses"]
    provenance = {}
    responses = []
    for stage in stages:
        if not (HERE/stage).exists():
            continue
        meta = read(stage+"/metadata.json")
        if "after" not in meta:
            raise RuntimeError(f"run still active: {stage}")
        with tarfile.open(HERE/stage/"source.tar.gz") as archive:
            matches = {name:hashlib.sha256(archive.extractfile(name).read()).hexdigest()==value
                       for name,value in meta["source_sha256"].items()}
        if not all(matches.values()):
            raise RuntimeError(f"incorrect source archive: {stage}")
        load = usage(meta)
        if (datetime.fromisoformat(load["end"])-datetime.fromisoformat(load["start"])).total_seconds()<1:
            load.update(busy_percent=None,average_busy_logical_cpus=None,
                        note="subsecond read-only audit; CPU tick resolution is insufficient")
        provenance[stage] = dict(source_snapshot_matches=matches,load=load)
        summary = read(stage+"/summary.json")
        for row in summary.get("rows",[]):
            if stage=="s5_soft_responses" and row["job"]["test"]=="r2":
                continue  # Saved as invalid measurement evidence below, not a physics failure.
            responses.append(dict(stage=stage,**row))
    old = HERE/"s5_soft_responses/source.tar.gz"
    with tarfile.open(old) as archive:
        old_backend = archive.extractfile("mjwarp_backend.py").read().decode()
    current = (SEGMENT/"mjwarp_backend.py").read_text()
    core = current.split("\n\ndef synchronize_contact_ids(data):")[0].rstrip()+"\n"
    expected = old_backend.replace("            mjw.get_data_into(self.d,self.m,self.batch.data)\n",
        "            mjw.get_data_into(self.d,self.m,self.batch.data)\n            synchronize_contact_ids(self.d)\n")
    only_readback_repaired = core==expected
    if not only_readback_repaired:
        raise RuntimeError("backend changed beyond the recorded contact-readback repair")
    release = SEGMENT/"results/pinch_effect_v1_2/v22_review/release_v1_2"
    tag = json.loads((release/"v1.2.json").read_text())
    changed = [name for name,value in tag["source_sha256"].items() if digest(PROJECT/name)!=value]
    runtime = SEGMENT/"results/cpu_equivalence_v1_2_1/reference_v1_2"
    runtime_changed = [name for name,value in tag["source_sha256"].items() if digest(runtime/name)!=value]
    content_changed = [name for name in changed if name.startswith("contents/segment/")]
    files = ("validate_warp.py","mjwarp_backend.py","warp_mechanics.py","warp_effects.py","warp_cpu.py")
    for name in files:
        ast.parse((SEGMENT/name).read_text())
    latest = read(list(provenance)[-1]+"/metadata.json")
    sources_match = {name:digest(SEGMENT/name)==value for name,value in latest["source_sha256"].items()}
    physics_files = ("warp_mechanics.py","warp_effects.py","warp_cpu.py")
    original_meta = read("s5_soft_responses/metadata.json")
    physics_unchanged = {name:digest(SEGMENT/name)==original_meta["source_sha256"][name] for name in physics_files}
    result = dict(hard_lower_bound_review=read("hard_lower_bound_review/summary.json"),
        valid_soft_responses=responses,
        invalid_original_r2=dict(reason="stale legacy contact IDs after native MJWarp readback",
            comparison=read("s5_soft_responses/bendable_r2/comparison.json")),
        native_readback_reproduction=read("s5_readback_repair/summary.json"),
        only_contact_readback_repaired=only_readback_repaired,
        earlier_r1_reusable_reason="R1 uses modern contact.geom; readback alias repair does not affect integration or its force measurement",
        physics_sources_unchanged=physics_unchanged,provenance=provenance,
        original_files_checked=len(tag["source_sha256"]),original_files_changed=changed,
        frozen_runtime_files_changed=runtime_changed,original_content_files_changed=content_changed,
        frozen_archive_unchanged=digest(release/"v1.2.tar.gz")==tag["archive_sha256"],
        source_syntax_passed=True,latest_sources_match=sources_match,
        S5_complete=False,S6_evaluate_per_second=None,S6_peak_vram=None,
        candidate_training_ready=False)
    if runtime_changed or content_changed or not result["frozen_archive_unchanged"] or not all(sources_match.values()):
        raise RuntimeError("frozen runtime, original contents, archive, or candidate provenance mismatch")
    (HERE/"audit_gpu_review.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(valid_soft_responses=len(responses),
        failed=[dict(job=r["job"],stage=r["stage"]) for r in responses if not r["passed"]],
        frozen_archive_unchanged=result["frozen_archive_unchanged"],original_files_changed=changed,
        only_contact_readback_repaired=only_readback_repaired,latest_sources_match=sources_match,
        load={stage:value["load"] for stage,value in provenance.items()}),indent=2))


if __name__=="__main__":
    main()

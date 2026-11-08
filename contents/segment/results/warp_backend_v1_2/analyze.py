"""Summarize saved reviewed-S2 gates without advancing any simulation."""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SEGMENT = ROOT / "contents/segment"


def read(path):
    return json.loads((HERE/path).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usage(meta):
    a = [int(v) for v in meta["before"]["proc_stat"].split()[1:9]]
    b = [int(v) for v in meta["after"]["proc_stat"].split()[1:9]]
    d = [y-x for x,y in zip(a,b)]
    busy = 1-(d[3]+d[4])/sum(d)
    return dict(start=meta["before"]["time"], end=meta["after"]["time"],
                loadavg_before=meta["before"]["loadavg"], loadavg_after=meta["after"]["loadavg"],
                busy_percent=100*busy, average_busy_logical_cpus=32*busy)


def main():
    operators = read("s2_operators_boundaries/summary.json")
    responses = read("s2_responses/summary.json")
    groups = defaultdict(lambda:dict(count=0, max_abs_diff=0., passed=True))
    for row in operators["rows"]:
        group = groups[row["name"]]
        group["count"] += 1
        group["max_abs_diff"] = max(group["max_abs_diff"],row["max_abs_diff"])
        group["passed"] &= row["passed"]
    meta = read("s2_responses/metadata.json")
    tag_path = SEGMENT/"results/pinch_effect_v1_2/v22_review/release_v1_2/v1.2.json"
    tag = json.loads(tag_path.read_text())
    changed = [name for name,value in tag["source_sha256"].items() if digest(ROOT/name)!=value]
    source_matches = {name:digest(SEGMENT/name)==value for name,value in meta["source_sha256"].items()}
    for name in meta["source_sha256"]:
        ast.parse((SEGMENT/name).read_text())
    ast.parse(Path(__file__).read_text())
    failed = [row for row in responses["rows"] if not row["passed"]]
    result = dict(operator_gate=dict(passed=operators["passed"], cases=operators["cases"],
                  max_abs_diff=operators["max_abs_diff"], groups=dict(groups)),
                  response_gate=dict(passed=responses["passed"], completed=responses["completed"],
                  planned=responses["planned"], failed=failed),
                  E4_E5="not run: response gate failed", S4_S6="not run: S2 incomplete",
                  CPU_speed_target="cancelled by review", released_v1_2_1=False,
                  quantization=dict(lock_strain_step=.0025, reference=.15250000000000008,
                       candidate=.15500000000000008, relative_diff=.016393442622950824,
                       source="contents/segment/metrics.py:r1_metrics np.arange step"),
                  load={stage:usage(read(stage+"/metadata.json")) for stage in
                        ("s2_operators_boundaries", "s2_responses")},
                  frozen_archive_unchanged=digest(tag_path.parent/"v1.2.tar.gz")==tag["archive_sha256"],
                  original_files_changed=changed, candidate_sources_match_run=source_matches,
                  source_syntax_passed=True, dependency_installations=0)
    (HERE/"audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__=="__main__":
    main()

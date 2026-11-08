"""Print one line per round-1 run (C1 Newton, C2 MuJoCo) for the report."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = lambda v: round(v, 1) if isinstance(v, float) else v
for f in sys.argv[1:] or sorted(p.name for p in HERE.glob("c[12]_*.jsonl") if "_runs" not in p.name):
    print("==", f)
    for l in open(HERE / f):
        d = json.loads(l)
        if "test" in d:
            keys = ["fail", "speed_flag", "pad_gap_mm", "dropped", "pkg_com_rise_mm", "content_com_rise_mm", "bag_slip_mm",
                    "content_rel_max_mm", "droop_bag", "grip_force_hold", "outside_end"]
            name = d.get("content", d.get("fill"))
            print(name, d["test"], "rtf", round(d.get("rtf", d["sim_t"] / d["wall_s"]), 4), {k: R(d.get(k)) for k in keys if k in d})
        else:
            keys = ["reason", "max_grip_force_N", "slip_mm", "content_rel_mm", "lift_height_mm", "outside", "speed_flag"]
            print(d["content"], d["tag"], d["outcome"], "rtf", round(d["sim_s"] / d["wall_s"], 4), {k: R(d.get(k)) for k in keys})

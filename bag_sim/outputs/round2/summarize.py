"""One line per round-2 run (F1-F5). Usage: python summarize.py [file.jsonl ...]"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = lambda v: round(v, 1) if isinstance(v, float) else v
for f in sys.argv[1:] or sorted(p.name for p in HERE.glob("*.jsonl") if "_runs" not in p.name):
    print("==", f)
    for l in open(HERE / f):
        d = json.loads(l)
        rtf = d["sim_t"] / d["wall_s"] if "sim_t" in d else d["sim_s"] / d["wall_s"]
        head = f'{d.get("tag", "")} {d.get("content", "")} {d.get("test", d.get("outcome"))} dt={d.get("dt")} nv={d.get("nv")} rtf={rtf:.3f}'
        if "test" in d:
            keys = ["fail", "speed_flag", "pad_gap_mm", "dropped", "pkg_com_rise_mm", "content_com_rise_mm", "slide_mm",
                    "bag_slip_mm", "content_rel_max_mm", "grip_force_hold", "penetration_mm"]
        else:
            keys = ["reason", "max_grip_force_N", "slip_mm", "content_rel_mm", "lift_height_mm", "penetration_mm", "speed_flag"]
        print(head, {k: R(d.get(k)) for k in keys if k in d})

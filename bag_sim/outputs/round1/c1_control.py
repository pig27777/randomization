"""C1 control: the same hard boxes without a bag, contents/segment/protocols R2 (edge) and R3, MuJoCo 3.17 CPU,
surface friction = kraft 0.45 (pads). Usage: python -B c1_control.py loose|full r2|r3"""
import json, sys
from dataclasses import replace
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[2] / "contents" / "segment")]
import protocols
from loose_protocols import BOX
fill, test = sys.argv[1:3]
k = replace(BOX[fill], surface_friction=0.45)
res = protocols.r2(k, "edge") if test == "r2" else protocols.r3(k)
res.update(fill=fill, backend="mujoco-3.17 content model (no bag)")
print(json.dumps(res, default=str))

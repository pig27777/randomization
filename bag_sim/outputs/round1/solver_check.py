"""Content-only R2 edge / R3 (contents/segment/protocols) with the content's CG solver vs Newton, as used by
mj_loose for bagged scenes. Usage: python -B solver_check.py ANCHOR TEST CG|Newton"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "contents" / "segment"))
import mj_adapter
import protocols
from model import ANCHORS
anchor, test, solver = sys.argv[1:4]
mj_adapter.SOLVER = mj_adapter.SOLVER.replace('solver="CG"', f'solver="{solver}"')
res = protocols.r3(ANCHORS[anchor]) if test == "r3" else protocols.r2(ANCHORS[anchor], "edge")
res.update(anchor=anchor, solver=solver)
print(json.dumps(res, default=str))

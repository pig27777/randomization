"""F4: the tight-fit box with only an equivalent skin (surface friction = kraft 0.45, no bag), contents/segment
protocols R2 (edge) / R3. skin_compliance as given (default 0). Usage: python -B f4_skin.py r2|r3 [skin m/N]"""
import json, sys
from dataclasses import replace
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[2] / "contents" / "segment")]
import protocols
from loose_protocols import BOX
test = sys.argv[1]
skin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
k = replace(BOX["fit"], surface_friction=0.45, skin_compliance=skin)
res = protocols.r2(k, "edge") if test == "r2" else protocols.r3(k)
res.update(content="box_fit", skin_compliance=skin, tag="F4 skin")
print(json.dumps(res, default=str))

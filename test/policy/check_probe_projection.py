"""Score paired pure-RGB renderings using a calibrated oblique color camera."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.low_order_parcel import lift_probe
from probe_rgb import CAMERA_OFFSET, CAMERA_ROTATION


def rgb_box(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    y, x = np.nonzero((rgb[..., 0] > 180) & (rgb[..., 2] < 70))
    if len(x) == 0:
        return None
    return torch.tensor([[(.5*(x.max()+x.min())-320)/440, (.5*(y.max()+y.min())-240)/440,
                          (x.max()-x.min())/440]], dtype=torch.float64)


def auc(rows, column):
    positives = [r[column] for r in rows if r["loaded"] and r[column] is not None]
    negatives = [r[column] for r in rows if not r["loaded"] and r[column] is not None]
    if not positives or not negatives:
        return None
    scores = [float(a>b) + .5*(a==b) for a in positives for b in negatives]
    return sum(scores)/len(scores)


def analyze(root):
    rows = []
    for r in json.loads((root/"results.json").read_text()):
        if r["rule"] != "continue":
            continue
        p = root / r["trial"]
        d, t = r["probe"], r["decision"]
        if not t:
            rows.append(dict(trial=r["trial"], reached=False))
            continue
        stream = np.load(p/"boundary.npz")["time_clearance_support_speed_left_right"]
        idx = np.searchsorted(stream[:,0], t["time_s"])
        a, b = rgb_box(p/"rgb-start.png"), rgb_box(p/"rgb-end.png")
        ratio = None
        if a is not None and b is not None:
            # FK at the same image exposure; take its world z from the sampled
            # site observations bracketing the pause for this evaluation.
            raw = json.loads((p/"result.json").read_text())["micro_lift"]["observations"]
            start = next(sample for sample in reversed(raw) if sample[1]=="close")
            dz = raw[-1][4] - start[4]
            gamma, valid = lift_probe.monocular_lift_ratio(a, b, torch.tensor([[dz]]), .10,
                                                         CAMERA_OFFSET[2], torch.tensor(CAMERA_ROTATION)[2:3])
            if valid.item():
                ratio = gamma.item()
        captured = bool((stream[idx,4:6]>.5).all())
        rows.append(dict(trial=r["trial"], reached=True, bilateral=captured,
                         loaded=bool(captured and stream[idx,1]>.001 and stream[idx,2]==0),
                         clearance_mm=float(stream[idx,1]*1000), support_N=float(stream[idx,2]),
                         current_N=t["late_load_N"], gamma_old=d["gamma_rgb"], gamma_width=ratio))
    valid = [r for r in rows if r["reached"]]
    return dict(cases=rows, auc={k:auc(valid,k) for k in ("current_N","gamma_old","gamma_width")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2, allow_nan=False))

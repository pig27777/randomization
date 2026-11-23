"""Write the before/after engineering table, using the kernel's table convention."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent


def metrics(q, original):
    blocks = q["vertices"].reshape(-1, 4, 8, 3)
    if original:
        # Archived C5 has 28 mm unloaded half thickness. The 8 symmetric
        # corners recover each original block center to float32 precision.
        center = blocks[..., 2].mean(axis=2)
        half = .028 - .5 * q["indent"]
        lower, upper = (center - half).min(1), (center + half).max(1)
        vis_bottom = blocks[..., 2].min((1, 2))
        vis_top = blocks[..., 2].max((1, 2))
    else:
        lower, upper = q["core_z"].T
        vis_bottom, vis_top = q["skin_z"].T
    bottom = q["corners"][..., 2].min(axis=2)
    pad = q["pads"][..., 2]
    rail = q["load"] if original else q["rail"]
    motor = q["motor"]
    series = dict(frame=np.arange(len(upper)), video_s=np.arange(len(upper))/25,
                  phys_top_mm=upper*1000, phys_bot_mm=lower*1000,
                  vis_top_mm=vis_top*1000, vis_bot_mm=vis_bottom*1000, dz_mm=(vis_top-upper)*1000,
                  pad_bottom_L_mm=bottom[:,0]*1000, pad_bottom_R_mm=bottom[:,1]*1000,
                  pad_center_L_mm=pad[:,0]*1000, pad_center_R_mm=pad[:,1]*1000,
                  depth_mm=(upper-bottom.min(1))*1000,
                  vertical_N=q["press"][:,0] if original else q["press"],
                  rail_L_N=rail[:,0], rail_R_N=rail[:,1], motor_L_N=motor[:,0], motor_R_N=motor[:,1])
    return series


def main():
    old = np.load(RUN/"original.npz")
    new = np.load(RUN/"right/geometry.npz")
    source_hash = json.loads((RUN/"flex.json").read_text())["trajectory_sha256"]
    if hashlib.sha256(new["substeps"].tobytes()).hexdigest() != source_hash:
        raise ValueError("The rendered and native-replay input trajectories differ")
    summary = {}
    for before, data in [(True,old),(False,new)]:
        name = "original" if before else "repaired"
        values = metrics(data,before)
        with (RUN/f"{name}-frames.csv").open("w",newline="") as f:
            writer=csv.writer(f, lineterminator="\n")
            writer.writerow(values)
            writer.writerows(zip(*values.values(),strict=True))
        depth = values["depth_mm"]
        summary[name]=dict(deepest_frame=int(depth.argmax()), max_depth_mm=float(depth.max()),
            max_abs_dz_mm=float(abs(values["dz_mm"]).max()), min_pad_mm=float(min(
                values["pad_bottom_L_mm"].min(), values["pad_bottom_R_mm"].min())))
    baseline = np.load(RUN/"original.substeps.npz")["samples"]
    summary["original"].update(peak_2ms_N=float(baseline[:,0].max()), side_mask_frame84=baseline[1699,1:3].tolist(),
                               side_mask_frame116=baseline[2339,1:3].tolist())
    summary["repaired"].update(peak_2ms_N=float(new["substeps"][:,8].max()),
        min_finger_skin_mm=float(new["clearance"].min()*1000), captured_frames=int(new["capture"].all(1).sum()))
    # Independent substep check of all eight CAD corners per finger using R_xyzw.
    from itertools import product
    poses = new["substeps"][:,13:].reshape(-1,2,7)
    quaternion = poses[:,:,None,3:6]
    size=np.array([.0144,.0034,.0164])
    center=np.array([[.02389881,-.04676265,-.08164032],[-.02389810,.04676242,-.08164010]])
    local=center[:,None]+np.array(list(product((-1,1), repeat=3)))*size/2
    cross=2*np.cross(quaternion,local)
    world=poses[:,:,None,:3]+local+poses[:,:,None,6:7]*cross+np.cross(quaternion,cross)
    lowest=world[...,2].min((1,2))
    bottom,top=new["core_bounds_2ms"].T
    summary["repaired"].update(max_2ms_depth_mm=float(1000*(top-lowest).max()),
                               min_2ms_table_mm=float(1000*lowest.min()), min_2ms_parcel_bottom_mm=float(1000*(lowest-bottom).min()))
    assert len(values["frame"])==180 and len(lowest)==3600
    assert 2 <= new["substeps"][:,8].max() <= 4 and 15 <= (top-lowest).max()*1000 <= 25
    assert (lowest > 0).all() and (lowest > bottom).all()
    assert summary["repaired"]["max_abs_dz_mm"] < .5 and abs(values["vis_bot_mm"][0]) < .001
    assert summary["repaired"]["min_finger_skin_mm"] >= 1.5
    (RUN/"audit.json").write_text(json.dumps(summary, indent=2))


if __name__=="__main__":
    main()

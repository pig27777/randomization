"""Compare the archived PE loading window with fixed-mesh checkpoint readouts."""
import argparse
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "test/runs/2026-11-20/bag"
RUN = ARCHIVE / "rss2027-stage1-material-load-ramps-r1"
sys.path.insert(0, str(ARCHIVE / "rss2027-stage1-kraft-held-load-step-r1"))
import replay_held_load as replay
from yam_platform.protocol import tips


def fit(x, y):
    k, b = np.polyfit(x, y, 1)
    return dict(k_N_m=float(k), r2=float(1.-sum((y-(k*x+b))**2)/sum((y-y.mean())**2)),
                span_mm=float(np.ptp(x)*1000))


def path_motion(edges, first, last):
    """Vertical-motion identity on one fixed, shortest-hop material path."""
    xyz, pad, com = first
    start, end = (int(np.argmin(np.linalg.norm(xyz - marker, axis=1))) for marker in (com, pad))
    adjacent = [[] for _ in xyz]
    for a, b in edges:
        adjacent[a].append(b)
        adjacent[b].append(a)
    parent = {start: None}
    queue = deque([start])
    while end not in parent and queue:
        source = queue.popleft()
        for node in adjacent[source]:
            if node not in parent:
                parent[node] = source
                queue.append(node)
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    edge_z, lengths = [], []
    for points, _, _ in (first, last):
        vectors = np.diff(points[path], axis=0)
        lengths.append(np.linalg.norm(vectors, axis=1))
        edge_z.append(vectors[:, 2] / lengths[-1])
    rotation = np.sum(np.mean(lengths, axis=0) * np.subtract(edge_z[1], edge_z[0]))
    stretch = np.sum(np.mean(edge_z, axis=0) * np.subtract(lengths[1], lengths[0]))
    delta = (last[1][2] - last[2][2]) - (pad[2] - com[2])
    surface = (last[0][end, 2] - last[0][start, 2]) - (xyz[end, 2] - xyz[start, 2])
    np.testing.assert_allclose(rotation + stretch, surface, atol=1e-12)
    return dict(path_vertices=list(map(int,path)), separation_change_mm=float(delta*1000),
                direction_change_mm=float(rotation*1000), length_change_mm=float(stretch*1000),
                endpoint_relative_motion_mm=float((delta-surface)*1000))


def read_window(case, path, native):
    # The held-state contains the entire integration state. The earlier
    # preparation file is unnecessary and is absent in this checkout.
    scene = replay.backend.scene
    with patch.object(replay.backend, "scene", lambda cfg: scene(replace(cfg, initial_state=None))):
        robot, _, sha = replay.held_scene()
    sim = robot.sim
    m, d = sim.m, sim.d
    if case == "half_solref":
        m.flex_solref[0, 0] *= .5
    if case == "half_radius":
        m.flex_radius[0] *= .5
    start_load = json.loads((replay.SOURCE / "load-check.json").read_bytes())["load_start_s"]
    keep = []
    strain = []
    markers = []
    ref = sim.parcel.envelope["reference"]
    edges = m.flex_edge.reshape(-1, 2)
    rest = np.linalg.norm(ref[edges[:, 0]] - ref[edges[:, 1]], axis=1)
    np.testing.assert_allclose(rest,m.flexedge_length0,rtol=1e-8)
    for i in range(round((native["t_s"][-1]-d.time)/m.opt.timestep)+1):
        t = float(d.time)
        d.xfrc_applied[int(sim.payload_bodies[0]), 2] = -max(0., 5*(t-start_load))
        sim.step()
        if i % 80 or t < native["t_s"][0]-1e-9:
            continue
        force = replay.backend.measure(robot)
        # Split-step contact readout uses start-of-step positions.
        x = d.flexvert_xpos
        markers.append((x.copy(), tips(robot).mean(axis=0), sim.com().copy()))
        strain.append(np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/rest-1.)
        normal, sliding = replay.inner_loads(replay.release_contacts(robot))
        keep.append((t, tips(robot)[:,2].mean()-sim.com()[2], -sum(a[2] for a in force["force_on_fingers_N"]),
                     force["penetration_mm"],force["support_N"], *normal,*sliding))
    arr = np.asarray(keep)
    np.testing.assert_allclose(arr[:,0], native["t_s"],atol=1e-9,rtol=0)
    np.savetxt(path / f"{case}.csv",arr,delimiter=",",comments="",
               header="time_s,separation_m,tension_N,penetration_mm,support_N,left_inner_N,right_inner_N,left_slide_N,right_slide_N")
    strain = np.array(strain)
    np.savez_compressed(path/f"{case}-edges.npz", edges=edges, rest_length_m=rest, time_s=arr[:,0], axial_strain=strain)
    qualified = (arr[:,4]<=.05)&(arr[:,5:7]>.001).all(1)&(arr[:,7:]<.5*arr[:,5:7]).all(1)&(arr[:,2]>.001)
    return dict(**fit(arr[:,1],arr[:,2]), path_motion=path_motion(edges, markers[0], markers[-1]), archive_max_force_difference_N=float(np.abs(arr[:,2]-native["tension_N"]).max()),
               archive_max_separation_difference_mm=float(1000*np.abs(arr[:,1]-(native["tip_z_m"]-native["payload_com_z_m"])).max()),
               max_abs_edge_strain=float(np.abs(strain).max()), max_edge_strain_change=float(np.ptp(strain,axis=0).max()),
               median_edge_strain_span=float(np.median(np.ptp(strain,axis=0))),
               penetration_range_mm=[float(arr[:,3].min()),float(arr[:,3].max())],qualified_frames=int(qualified.sum()),
               warnings=int(d.warning.number.sum()), checkpoint=sha, radius_m=float(m.flex_radius[0]),
               flex_solref=m.flex_solref[0].tolist())


def main(output):
    output.mkdir(parents=True,exist_ok=False)
    replay.SOURCE = RUN/"pe/recording"
    replay.CHECKPOINT = ARCHIVE/"rss2027-stage1-pe-held-load-step-r1/held-state.npz"
    source = np.genfromtxt(replay.SOURCE/"load-curve.csv",delimiter=",",names=True)
    window = json.loads((RUN/"pe/parameter-observations.json").read_bytes())["pre_slip_stiffness_candidates"][0]
    native = source[(source["t_s"]>=window["start_s"]) & (source["t_s"]<=window["end_s"])]
    metrics = {"archive":fit(native["tip_z_m"]-native["payload_com_z_m"],native["tension_N"])}
    for case in ("baseline","half_solref","half_radius"):
        metrics[case] = read_window(case,output,native)
        print(case,metrics[case],flush=True)
    (output/"stiffness.json").write_text(json.dumps(metrics,indent=2)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    main(parser.parse_args().output)

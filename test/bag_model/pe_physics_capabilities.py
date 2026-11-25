"""Measure independent CPU and CUDA Flex features at small, prescribed states."""
import argparse
from importlib.metadata import version
from pathlib import Path
import platform
import traceback

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from pe_physics import BASE, save_json


def world(dimension=2, elastic="stretch", *, edge=False, contact=None):
    counts = {1: "4 1 1", 2: "3 3 1", 3: "2 2 2"}
    elasticity = f'<elasticity young="2e7" poisson=".3" thickness=".00006" elastic2d="{elastic}"/>'
    if dimension != 2:
        elasticity = '<elasticity young="2e4" poisson=".3"/>' if dimension == 3 else ""
    if edge:
        elasticity = '<edge equality="true"/>'
    contact_xml = ('<contact selfcollide="auto" friction=".5 .005 .001" condim="3"/>'
                   if contact == "self" else '<contact selfcollide="none" friction=".5 .005 .001" condim="3"/>')
    plane = '<geom name="table" type="plane" size="1 1 .01" condim="3" friction=".5 .005 .001"/>'
    if contact == "self":
        # A flex with two distinct triangulated layers covers the self-contact path.
        v = [(x, y, z) for z in (.01, .0104) for x, y in ((0, 0), (.01, 0), (.01, .01), (0, .01))]
        points = " ".join(str(c) for p in v for c in p)
        mesh = (f'type="direct" point="{points}" element="0 1 2 0 2 3 4 5 6 4 6 7"')
        elasticity = ""
        plane = ""
    else:
        mesh = f'type="grid" count="{counts[dimension]}" spacing=".01 .01 .01"'
        if contact != "table":
            plane = ""
    height = ".000125" if contact == "table" else "0"
    integrator = "implicitfast" if edge else "discrete"
    xml = f'''
    <mujoco model="feature_probe"><option timestep=".00005" gravity="0 0 0" cone="elliptic"
        integrator="{integrator}" solver="Newton" iterations="30"/>
      <worldbody>{plane}<flexcomp name="film" {mesh} dim="{dimension}" radius=".00025"
        mass=".001" pos="0 0 {height}">{contact_xml}{elasticity}</flexcomp></worldbody>
    </mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    dofs = m.body_dofadr[m.flex_vertbodyid]
    if contact is None:
        component = 2 if elastic == "bend" else 0
        d.qpos[dofs[-1] + component] = .0002
    elif contact == "table":
        d.qvel[dofs] = .02
    mujoco.mj_forward(m, d)
    return m, d


def cpu_probe(m, d):
    reaction = np.array([0., 0.])
    count = 0
    force = np.zeros(6)
    for i, contact in enumerate(d.contact):
        if contact.efc_address >= 0:
            mujoco.mj_contactForce(m, d, i, force)
            reaction += [force[0], np.linalg.norm(force[1:3])]
            count += int(force[0] > 0)
    return dict(dim=int(m.flex_dim[0]), finite=bool(np.isfinite(d.qacc).all()),
                acceleration_norm=float(np.linalg.norm(d.qacc)), equality_constraints=int(d.ne),
                equality_force_norm=float(np.linalg.norm(d.efc_force[:d.ne])), contacts=count,
                normal_force_N=float(reaction[0]), tangential_force_N=float(reaction[1]))


def gpu_probe(m, d, *, worlds=1, graph=False):
    with wp.ScopedDevice("cuda:0"):
        gm = mjw.put_model(m)
        gd = mjw.put_data(m, d, nworld=worlds, nconmax=256, njmax=1024)
        # Distinguish both worlds with a physical initial perturbation.
        if worlds > 1:
            start = gd.qpos.numpy()
            start[1, -1] += .0001
            gd.qpos.assign(start)
        mjw.forward(gm, gd)
        wp.synchronize_device()
        actual_count = min(int(gd.nacon.numpy()[0]), gd.naconmax)
        forces = wp.zeros(actual_count, dtype=wp.spatial_vector)
        if actual_count:
            ids = wp.array(np.arange(actual_count), dtype=int)
            mjw.contact_force(gm, gd, ids, False, forces)
        accel = gd.qacc.numpy()
        reactions = forces.numpy()
        limit = int(d.ne)
        details = dict(device=str(gd.qpos.device), dims=m.flex_dim.tolist(), nworld=worlds,
                       finite=bool(np.isfinite(accel).all()), acceleration_norm=float(np.linalg.norm(accel[0])),
                       equality_force_norm=float(np.linalg.norm(gd.efc.force.numpy()[0, :limit])),
                       counts=int(np.count_nonzero(reactions[:, 0] > 0.)),
                       normal_force_N=float(reactions[:, 0].sum()),
                       tangential_force_N=float(np.linalg.norm(reactions[:, 1:3], axis=1).sum()),
                       overflow=gd.overflow.numpy().tolist())
        if graph:
            mjw.step(gm, gd)
            before = gd.time.numpy()
            with wp.ScopedCapture() as capture:
                mjw.step(gm, gd)
            wp.capture_launch(capture.graph)
            wp.synchronize_device()
            increment = gd.time.numpy() - before
            details.update(graph_increment_s=increment.tolist(),
                           capture_replay_pass=bool(np.allclose(increment, m.opt.timestep, rtol=1e-4)),
                           distinct_worlds=bool(not np.allclose(gd.qpos.numpy()[0], gd.qpos.numpy()[1])))
        return details


def parcel_probe():
    from bag_sim.physical_parcel_gpu import Batch

    with wp.ScopedDevice("cuda:0"):
        bag = Batch(2, BASE, randomize=False, contact_capacity=2048, constraint_capacity=8192)
        graph = bag.graph(steps=4)
        initial = bag.d.time.numpy()
        wp.capture_launch(graph)
        wp.synchronize_device()
        times = (bag.d.time.numpy() - initial).tolist()
        return dict(device=str(bag.d.qpos.device), nworld=bag.nworld,
                    vertices=int(bag.parcel.m.nflexvert), graph_increment_s=times,
                    capture_replay_pass=bool(np.allclose(times, 4 * BASE.timestep, rtol=1e-4)),
                    finite=bool(np.isfinite(bag.d.qpos.numpy()).all()), **bag.diagnostics())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parcel-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    names = {
        "dim1_edge": dict(dimension=1, edge=True),
        "dim2_stretch": {},
        "dim2_native_bend": dict(elastic="bend"),
        "dim2_no_elasticity": dict(elastic="none"),
        "dim3_elasticity": dict(dimension=3),
        "dim2_edge": dict(edge=True),
        "dim2_self_contact": dict(contact="self"),
        "dim2_table_friction": dict(contact="table"),
        "dim2_two_world_cuda_graph": {},
    }
    source = Path(mjw.__file__).parent / "_src"
    result = dict(host=platform.node(), versions={p: version(p) for p in ("mujoco", "mujoco-warp", "warp-lang")},
                  source={name: str(source / name) for name in ("io.py", "forward.py", "passive.py",
                                                                 "support.py", "constraint.py")},
                  cases={})
    if args.parcel_only:
        try:
            result["cases"]["existing_parcel_cuda_graph"] = parcel_probe()
        except (ValueError, RuntimeError, NotImplementedError, AttributeError, TypeError):
            result["cases"]["existing_parcel_cuda_graph"] = {"cuda_error": traceback.format_exc()}
        save_json(args.output / "capabilities.json", result)
        print(result["cases"], flush=True)
        return
    for name, kwargs in names.items():
        row = result["cases"][name] = {}
        try:
            model, data = world(**kwargs)
            row["cpu"] = cpu_probe(model, data)
        except (ValueError, RuntimeError):
            row["cpu_error"] = traceback.format_exc()
        else:
            try:
                graph = name.endswith("_graph")
                row["cuda"] = gpu_probe(model, data, worlds=2 if graph else 1, graph=graph)
            except (ValueError, RuntimeError, NotImplementedError, AttributeError, TypeError):
                row["cuda_error"] = traceback.format_exc()
        save_json(args.output / "capabilities.json", result)
        print(name, row, flush=True)


if __name__ == "__main__":
    main()

"""Check one fixed continuum B, geometric/material X and CUDA plastic history.

Material probes use free-body shear release, not an alternative grasp executor.
Run GPU checks under the workspace flock; outputs refuse existing directories.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "contents"))
from continuum.model import Parameters, Template, build, geometry, preset, sample, shape_parameters, topology_signature


def construction():
    b = Template()
    rows = []
    for name in ("hard", "bendable", "compressible"):
        for shape in ("box", "rounded", "pillow", "tapered"):
            p = replace(preset(name), **shape_parameters(shape))
            built = build(p, b)
            m, d, g = built.model, built.data, built.geometry
            np.testing.assert_allclose(np.ptp(d.flexvert_xpos, axis=0), p.size, atol=1e-12)
            np.testing.assert_allclose(m.body_mass[m.flex_vertbodyid].sum(), g.mass, rtol=1e-12)
            assert m.nflex == 1 and m.nplastic == 6 * m.nflexelem and m.neq == 0
            rows.append(dict(material=name, shape=shape, nv=m.nv, nplastic=m.nplastic,
                             topology=topology_signature(m), volume_m3=g.volume, mass_kg=g.mass))
    assert len({r["topology"] for r in rows}) == 1
    assert len({round(r["volume_m3"],12) for r in rows}) == 4
    rng = np.random.default_rng(20261109)
    for _ in range(24):
        p = sample(rng)
        assert geometry(p,b).volume > 0
    rejected = 0
    for options in ({"young":-1}, {"density":0}, {"yield_strain":0}, {"poisson":.5},
                    {"size":[.1,0,.1]}, {"roundness":2}, {"bulge":1}, {"taper":1}):
        try:
            Parameters(**options)
        except ValueError:
            rejected += 1
    assert rejected == 8
    return dict(instances=rows, sampled_shapes=24, rejected_inputs=rejected, template=asdict(b))


def shear_data(built, amount):
    m, d = built.model, built.data
    z = d.flexvert_xpos[:,2] - np.mean(d.flexvert_xpos[:,2])
    addresses = [m.jnt_qposadr[m.body_jntadr[body]] for body in m.flex_vertbodyid]
    d.qpos[addresses] += amount * z


def advance(sim, steps):
    for _ in range(steps):
        sim.step()
        sim.assert_valid()


def materials(output):
    from continuum.gpu import Simulation
    rows = []
    for name in ("hard", "bendable", "compressible"):
        built = build(preset(name), gravity=(0,0,0), table=False)
        shear_data(built, .04)
        sim = Simulation(built.model, built.data)
        restoring = float(np.linalg.norm(sim.data.qfrc_passive.numpy()[0]))
        sim.step(external=np.zeros((built.model.nbody,6)))
        sim.assert_valid()
        advance(sim,19)
        state = sim.state()
        np.savez_compressed(output/f"{name}.npz", integration_state=state,
                            integration_state_spec=sim.state_spec, plastic=sim.data.plastic.numpy())
        rows.append(dict(name=name, restoring_force_norm_N=restoring,
                         health=sim.health(), software=sim.software()))
    assert rows[0]["restoring_force_norm_N"] > rows[1]["restoring_force_norm_N"] > rows[2]["restoring_force_norm_N"] > 0
    return rows


def plastic_release(output):
    from continuum.gpu import Simulation
    rows = []
    for name, threshold in (("elastic_control", 1e3), ("plastic", .003)):
        p = replace(preset("bendable"), yield_strain=threshold)
        built = build(p, gravity=(0,0,0), table=False)
        shear_data(built,.08)
        sim = Simulation(built.model,built.data)
        advance(sim,1)
        first = float(np.linalg.norm(sim.data.plastic.numpy()))
        advance(sim,399)
        final = float(np.linalg.norm(sim.data.plastic.numpy()))
        q = sim.data.qpos.numpy()[0]
        vertices = sim.rest + q[sim.addresses[:,None]+np.arange(3)]
        edges = built.model.flex_edge
        original = sim.rest[edges[:,1]]-sim.rest[edges[:,0]]
        deformed = vertices[edges[:,1]]-vertices[edges[:,0]]
        metric_difference = float(np.linalg.norm(np.sum(deformed**2,1)-np.sum(original**2,1)))
        np.savez_compressed(output/f"{name}.npz", integration_state=sim.state(),
                            integration_state_spec=sim.state_spec, vertices=vertices,
                            plastic=sim.data.plastic.numpy())
        rows.append(dict(name=name, first_plastic_norm=first, final_plastic_norm=final,
                         edge_metric_change_m2=metric_difference, health=sim.health()))
    assert rows[0]["first_plastic_norm"] == rows[0]["final_plastic_norm"] == 0
    assert rows[1]["first_plastic_norm"] > 1e-12 and rows[1]["final_plastic_norm"] > 1e-12
    return rows


def state_and_isolation(output):
    from continuum.gpu import Simulation
    import mujoco_warp as mjw
    p = replace(preset("bendable"), yield_strain=.003)
    built = build(p, gravity=(0,0,0), table=False)
    sim = Simulation(built.model,built.data,nworld=2)
    q = sim.data.qpos.numpy()
    q[0,sim.addresses] += .08 * (sim.rest[:,2]-np.mean(sim.rest[:,2]))
    sim.data.qpos.assign(q)
    mjw.forward(sim.model,sim.data)
    advance(sim,20)
    plastic = sim.data.plastic.numpy()
    assert np.linalg.norm(plastic[0]) > 1e-12
    assert np.linalg.norm(plastic[1]) == 0
    saved = sim.state()
    advance(sim,5)
    expected_q, expected_p = sim.data.qpos.numpy(), sim.data.plastic.numpy()
    sim.restore(saved)
    np.testing.assert_array_equal(saved,sim.state())
    advance(sim,5)
    dq = float(np.max(np.abs(sim.data.qpos.numpy()-expected_q)))
    dp = float(np.max(np.abs(sim.data.plastic.numpy()-expected_p)))
    assert dq <= 1e-6 and dp <= 1e-6
    np.savez_compressed(output/"state_isolation.npz", checkpoint=saved,
                        replay=sim.state(), integration_state_spec=sim.state_spec)
    return dict(world0_plastic_norm=float(np.linalg.norm(plastic[0])), world1_plastic_norm=0.,
                replay_qpos_difference=dq, replay_plastic_difference=dp, health=sim.health())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("construction","materials","plastic","state","all"), default="all")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    paths = [Path(__file__), *sorted((ROOT/"contents/continuum").glob("*.py")), ROOT/"bag_sim/flex_residual.py"]
    sources = {}
    for i,path in enumerate(paths):
        payload = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
        (args.output/f"source_{i}_{path.name}").write_bytes(payload)
    report = dict(hostname=platform.node(), mode=args.mode, sources=sources, results={}, valid=True)
    checks = {"construction": construction, "materials": lambda:materials(args.output),
              "plastic":lambda:plastic_release(args.output), "state":lambda:state_and_isolation(args.output)}
    for name, function in checks.items():
        if args.mode not in (name,"all"):
            continue
        try:
            report["results"][name] = dict(passed=True, data=function())
        except Exception as error:
            report["results"][name] = dict(passed=False,error=str(error),traceback=traceback.format_exc())
            report["valid"] = False
            break
    report["sources_unchanged"] = all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha for path,sha in sources.items())
    report["valid"] &= report["sources_unchanged"]
    (args.output/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"valid":report["valid"], "checks":{k:v["passed"] for k,v in report["results"].items()},
                      "output":str(args.output)}),flush=True)
    return int(not report["valid"])


if __name__ == "__main__":
    raise SystemExit(main())

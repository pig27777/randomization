"""Native 2D stretch-only membrane containing unbonded rigid sphere grains."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bag_sim"))
from wrap import point_triangle_distance, winding_number

DT = .0005
RADIUS = .008
FILM_RADIUS = .0005


def shell_mesh(n):
    points, faces, lookup = [], [], {}
    for axis in range(3):
        uv = [a for a in range(3) if a != axis]
        for side in (0, n):
            grid = {}
            for i in range(n + 1):
                for j in range(n + 1):
                    p = [0, 0, 0]
                    p[axis], p[uv[0]], p[uv[1]] = side, i, j
                    key = tuple(p)
                    if key not in lookup:
                        lookup[key] = len(points)
                        points.append(p)
                    grid[i, j] = lookup[key]
            for i in range(n):
                for j in range(n):
                    a, b, c, d = [grid[k] for k in ((i, j), (i+1, j), (i+1, j+1), (i, j+1))]
                    for tri in ((a, b, c), (a, c, d)):
                        p = np.asarray(points)[list(tri)]
                        normal = np.cross(p[1]-p[0], p[2]-p[0])[axis]
                        faces.append(tri if normal * (1 if side else -1) > 0 else tri[::-1])
    vertices = (np.asarray(points, float) / n - .5) * [.24, .18, .16]
    vertices[:, 2] += .0825
    faces = np.asarray(faces, int)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if not np.all(counts == 2):
        raise ValueError("shell must be a closed two-manifold")
    return vertices, faces


def make_model(count, divisions, solver):
    vertices, faces = shell_mesh(divisions)
    side = round(count ** (1/3))
    if side ** 3 != count:
        raise ValueError("grain count must be a perfect cube")
    x = (np.arange(side) - (side - 1)/2) * .0175
    centers = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    centers[:, 2] += .083
    vec = lambda a: " ".join(format(float(v), ".12g") for v in np.asarray(a).ravel())
    grains = "\n".join(f'<body name="grain{i}" pos="{vec(p)}"><freejoint/>'
        f'<geom name="grain_geom{i}" type="sphere" size="{RADIUS}" density="600" '
        'contype="4" conaffinity="6" rgba=".8 .6 .15 1"/></body>' for i, p in enumerate(centers))
    xml = f'''<mujoco model="flex_granular"><compiler angle="radian"/>
      <option integrator="discrete" solver="{solver}" timestep="{DT}" iterations="100"
        tolerance="1e-6" ls_iterations="20" cone="elliptic" gravity="0 0 -9.81"/>
      <default><geom condim="3" friction=".5 .005 .0001" solref=".001 1" solimp=".95 .99 .001"/></default>
      <worldbody><light pos="0 0 1"/>
        <geom name="floor" type="plane" size=".5 .5 .01" contype="1" conaffinity="2" rgba=".8 .8 .8 1"/>
        <flexcomp name="film" type="direct" dim="2" dof="full" point="{vec(vertices)}"
          element="{' '.join(map(str, faces.ravel()))}" radius="{FILM_RADIUS}" mass=".012" rgba=".1 .65 .7 .4">
          <elasticity young="5000000" poisson=".3" thickness=".0002" damping=".0001" elastic2d="stretch"/>
          <contact selfcollide="auto" contype="2" conaffinity="7" condim="3"
             friction=".5 .005 .0001" solref=".001 1" solimp=".95 .99 .001"/>
        </flexcomp>{grains}
      </worldbody></mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bodies = np.array([m.body(f"grain{i}").id for i in range(count)])
    addresses = m.jnt_qposadr[m.body_jntadr[bodies]]
    film_addresses = m.jnt_qposadr[m.body_jntadr[m.flex_vertbodyid]]
    metadata = dict(grains=count, divisions=divisions, vertices=len(vertices), triangles=len(faces),
                    nq=m.nq, nv=m.nv, grain_radius_m=RADIUS, film_radius_m=FILM_RADIUS,
                    grain_mass_kg=float(m.body_mass[bodies].sum()), film_mass_kg=.012,
                    membrane_only=True, self_collision="auto", solver=solver, timestep=DT,
                    contact_time_constant_s=.001)
    return m, d, vertices, faces, addresses, film_addresses, metadata, xml


def geometry(qpos, rest, faces, grain_q, film_q):
    vertices = rest + qpos[film_q[:, None] + np.arange(3)]
    centers = qpos[grain_q[:, None] + np.arange(3)]
    if not np.isfinite(vertices).all() or not np.isfinite(centers).all():
        return dict(valid=False, outside=len(centers), protrusion_mm=None)
    winding = np.abs(winding_number(vertices, faces, centers))
    distances = point_triangle_distance(centers, vertices, faces)
    outside = int(np.count_nonzero(winding < .5))
    protrusion = float(np.maximum(RADIUS - distances, 0).max())
    return dict(valid=outside == 0 and protrusion <= .001, outside=outside,
                protrusion_mm=protrusion*1000, winding_min=float(winding.min()))


def warnings(d):
    return {name:int(d.warning[value].number) for name,value in mujoco.mjtWarning.__members__.items()
            if name != "mjNWARNING" and d.warning[value].number}


def run_cpu(m, rest, faces, grain_q, film_q, steps):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    samples = []
    wall = 0.
    failure = None
    peak_contacts = peak_constraints = peak_iterations = 0
    peak_penetration = 0.
    started = time.perf_counter()
    executed = 0
    for first in range(0, steps, 20):
        block = min(20, steps-first)
        t = time.perf_counter()
        # Keep per-step contact/capacity evidence; geometry is sampled every 10 ms.
        for _ in range(block):
            mujoco.mj_step(m, d)
            peak_contacts = max(peak_contacts, d.ncon)
            peak_constraints = max(peak_constraints, d.nefc)
            peak_iterations = max(peak_iterations, int(d.solver_niter.max()))
            if d.ncon:
                peak_penetration = max(peak_penetration, float(max(0., -d.contact.dist.min())))
        wall += time.perf_counter()-t
        executed += block
        g = geometry(d.qpos, rest, faces, grain_q, film_q)
        finite = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(d.qacc).all())
        samples.append(dict(step=executed, time=executed*DT, geometry=g, ncon=d.ncon, nefc=d.nefc))
        if warnings(d) or not finite or not g["valid"] or peak_penetration > .001 or peak_iterations>=m.opt.iterations:
            failure = dict(warnings=warnings(d), finite=finite, geometry=g,
                           peak_penetration_mm=peak_penetration*1000,iteration_limit=peak_iterations>=m.opt.iterations)
            break
    return dict(valid=failure is None and executed==steps, failure=failure, steps=executed,
                wall_s=wall, instrumented_wall_s=time.perf_counter()-started,
                env_steps_per_s=executed/wall, realtime_factor=executed*DT/wall,
                peak_contacts=peak_contacts, peak_constraints=peak_constraints, peak_iterations=peak_iterations,
                peak_penetration_mm=peak_penetration*1000, samples=samples), d


def run_gpu(m, rest, faces, grain_q, film_q, steps, worlds, ncon, njmax,
            record_path=None, continue_solver_warnings=False):
    import mujoco_warp as mjw
    import warp as wp
    from gpu_checks import state_check, contact_check, counters
    wp.set_device("cuda:0")
    started_setup = time.perf_counter()
    wm = mjw.put_model(m)
    # Disposable initial state compiles before formal data is uploaded.
    scratch = mjw.put_data(m, mujoco.MjData(m), nworld=worlds, nconmax=ncon, njmax=njmax)
    mjw.step(wm, scratch)
    wp.synchronize()
    del scratch
    data = mjw.put_data(m, mujoco.MjData(m), nworld=worlds, nconmax=ncon, njmax=njmax)
    flags = wp.zeros(worlds, dtype=int)
    peak = wp.zeros(4, dtype=int)
    penetration = wp.zeros(1, dtype=float)

    def one():
        mjw.step(wm, data)
        wp.launch(state_check, (worlds,max(m.nq,m.nv)), inputs=[data.qpos,data.qvel,data.qacc,flags])
        wp.launch(counters, worlds, inputs=[data.overflow,data.nefc,data.solver_niter,data.nacon,flags,peak])
        wp.launch(contact_check, data.naconmax, inputs=[data.contact.dist,data.nacon,penetration])

    # Compile check kernels on disposable state, then restore formal state.
    one()
    wp.synchronize()
    data = mjw.put_data(m, mujoco.MjData(m), nworld=worlds, nconmax=ncon, njmax=njmax)
    flags.zero_(); peak.zero_(); penetration.zero_()
    with wp.ScopedCapture() as capture:
        for _ in range(20):
            one()
    if not np.array_equal(data.qpos.numpy()[0], mujoco.MjData(m).qpos.astype(np.float32)):
        raise RuntimeError("graph capture advanced formal state")
    wp.synchronize()
    setup_wall = time.perf_counter()-started_setup
    samples = []
    recorded_qpos = [data.qpos.numpy()[0].copy()] if record_path else []
    recorded_time = [0.] if record_path else []
    failure = None
    executed = 0
    wall = 0.
    started = time.perf_counter()
    for _ in range(steps//20):
        t = time.perf_counter()
        wp.capture_launch(capture.graph)
        wp.synchronize()
        wall += time.perf_counter()-t
        executed += 20
        status = flags.numpy()
        depth = float(penetration.numpy()[0])
        q = data.qpos.numpy()
        if record_path:
            recorded_qpos.append(q[0].copy())
            recorded_time.append(executed*DT)
        g = geometry(q[0], rest, faces, grain_q, film_q)
        raw_flags = int(peak.numpy()[3])
        samples.append(dict(step=executed,time=executed*DT,geometry=g,flags_or=int(np.bitwise_or.reduce(status)),
                            peak_penetration_mm=depth*1000))
        allowed = 1536 if continue_solver_warnings else 0
        hard_status = np.any(status & 1) or (raw_flags & ~allowed) != 0
        if hard_status or depth > .001 or not g["valid"]:
            failure = samples[-1]
            break
        if record_path and executed % 500 == 0:
            print(json.dumps(dict(recorded_sim_time_s=executed*DT,wall_s=wall,
                                  raw_flags=raw_flags)),flush=True)
    final_geometry = [geometry(row,rest,faces,grain_q,film_q) for row in data.qpos.numpy()]
    if not all(g["valid"] for g in final_geometry):
        failure = dict(reason="final geometry failed",bad_worlds=[i for i,g in enumerate(final_geometry) if not g["valid"]])
    stats = peak.numpy().tolist()
    if failure is None and stats[3]:
        failure = dict(reason="recorded solver budget warnings",raw_flags=stats[3])
    if record_path:
        with record_path.open("xb") as stream:
            np.savez_compressed(stream,time=np.asarray(recorded_time),qpos=np.asarray(recorded_qpos),
                                rest=rest,faces=faces,grain_q=grain_q,film_q=film_q)
    summary = dict(valid=failure is None and executed==steps,failure=failure,steps=executed,nworld=worlds,
        wall_s=wall,instrumented_wall_s=time.perf_counter()-started,setup_wall_s=setup_wall,
        env_steps_per_s=worlds*executed/wall,realtime_factor=executed*DT/wall,
        peak_contacts_total=stats[0],peak_constraints=stats[1],peak_iterations=stats[2],
        overflow_or=stats[3],peak_penetration_mm=float(penetration.numpy()[0])*1000,
        samples=samples,final_bad_worlds=sum(not g["valid"] for g in final_geometry),
        nconmax_per_world=ncon,njmax=njmax)
    return summary, None


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--backend",choices=("cpu","gpu"),default="cpu")
    p.add_argument("--solver",choices=("CG","Newton"),default="Newton")
    p.add_argument("--grains",type=int,default=64)
    p.add_argument("--divisions",type=int,default=4)
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--worlds",type=int,default=1)
    p.add_argument("--repeats",type=int,default=1)
    p.add_argument("--nconmax",type=int,default=4096)
    p.add_argument("--njmax",type=int,default=12288)
    p.add_argument("--record",action="store_true",help="save 100 Hz GPU states for video")
    p.add_argument("--continue-solver-warnings",action="store_true",
                   help="record-only: retain warnings and continue; result still invalid")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.steps<=0 or a.steps%20 or a.repeats<=0 or a.worlds<=0:
        p.error("positive steps divisible by 20; positive repeats and worlds required")
    if a.record and (a.backend!="gpu" or a.worlds!=1 or a.repeats!=1):
        p.error("recording requires one GPU world and one repeat")
    if a.continue_solver_warnings and not a.record:
        p.error("continuing solver warnings is only for recorded diagnostic runs")
    if a.output.exists():
        p.error("refusing to overwrite result")
    m,d,rest,faces,grain_q,film_q,metadata,xml=make_model(a.grains,a.divisions,a.solver)
    initial=geometry(d.qpos,rest,faces,grain_q,film_q)
    if not initial["valid"] or warnings(d):
        raise RuntimeError(f"invalid initial state: {initial}; {warnings(d)}")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_suffix(".xml").write_text(xml)
    sources=[Path(__file__).resolve(),ROOT/"bag_sim/wrap.py"]
    if a.backend=="gpu": sources.append(Path(__file__).with_name("gpu_checks.py"))
    source_sha={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
    trials=[]
    for _ in range(a.repeats):
        if a.backend=="cpu":
            trial,last=run_cpu(m,rest,faces,grain_q,film_q,a.steps)
        else:
            trial,last=run_gpu(m,rest,faces,grain_q,film_q,a.steps,a.worlds,a.nconmax,a.njmax,
                a.output.with_suffix(".trajectory.npz") if a.record else None,a.continue_solver_warnings)
        trials.append(trial)
        print(json.dumps({k:v for k,v in trial.items() if k not in ("samples",)},ensure_ascii=False),flush=True)
        if not trial["valid"]: break
    unchanged=source_sha=={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
    valid=all(t["valid"] for t in trials) and len(trials)==a.repeats and unchanged
    result=dict(host=platform.node(),backend=a.backend,model=metadata,initial_geometry=initial,
        nworld=a.worlds,repeats=a.repeats,valid=valid,trials=trials,
        median_env_steps_per_s=float(np.median([t["env_steps_per_s"] for t in trials])) if valid else None,
        median_realtime_factor=float(np.median([t["realtime_factor"] for t in trials])) if valid else None,
        source_sha256=source_sha,sources_unchanged=unchanged,
        recorded=a.record,continue_solver_warnings=a.continue_solver_warnings,
        software=dict(mujoco=mujoco.__version__,numpy=np.__version__,
            mujoco_warp=importlib.metadata.version("mujoco-warp"),warp=importlib.metadata.version("warp-lang")),
        timing="step+per-step numerical/contact counters; CPU geometry/readback separately in instrumented_wall_s",
        limitations="unmeasured stretch-only film; homogeneous GPU copies; no grasping or independent solver residual acceptance")
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    if last is not None:
        np.savez_compressed(a.output.with_suffix(".npz"),qpos=last.qpos,rest=rest,faces=faces,
                            grain_q=grain_q,film_q=film_q)
    if not valid: raise SystemExit(2)


if __name__=="__main__":
    main()

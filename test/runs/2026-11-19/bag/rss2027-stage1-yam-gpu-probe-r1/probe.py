"""Probe archived full-YAM parcel frames with CPU MuJoCo and Warp."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[4]
ARCHIVE = ROOT / "test/runs/2026-11-16/bag/physical-parcel-mj4-yam-video-r3/trial"
sys.path[:0] = [str(ROOT / "test/bag_model"), str(ROOT), str(ROOT / "experience")]
import physical_parcel_yam as backend
from bag_sim.physical_parcel_gpu import bending
from yam_platform.provenance import source_files
from yam_platform.robot import asset_fingerprint


def memory_mib():
    return int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    ).splitlines()[0])


def gpu_finger_forces(m, d, finger_sets):
    n = int(d.nacon.numpy()[0])
    if not n:
        return np.zeros((2, 3)), n, [0, 0]
    ids = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32)
    forces = wp.empty(n, dtype=wp.spatial_vector)
    mjw.contact_force(m, d, ids, True, forces)
    forces = forces.numpy()[:, :3]
    geoms = d.contact.geom.numpy()[:n]
    flex = d.contact.flex.numpy()[:n]
    addresses = d.contact.efc_address.numpy()[:n, 0]
    active = (flex == 0).any(axis=1) & (addresses >= 0)
    mapped = []
    counts = []
    for group in finger_sets:
        first, second = np.isin(geoms.T, tuple(group))
        mapped.append(forces[active & second].sum(axis=0) - forces[active & first].sum(axis=0))
        counts.append(int((active & (first | second)).sum()))
    return np.asarray(mapped), n, counts


def probe_case(robot, qpos, label):
    m = robot.sim.m
    cpu = robot.sim.d = mujoco.MjData(m)
    cpu.qpos[:] = qpos
    robot.qtarget = cpu.qpos[robot.qadr].copy()
    robot.steps = 0
    mujoco.mj_forward(m, cpu)
    target = cpu.site_xpos[robot.site] + [0, 0, -.0005]
    gap = max(0., robot.gap() - .0002)
    parcel = robot.sim.parcel
    dofs = parcel.node_dofs(m)
    kwargs = {"nworld": 1, "nconmax": 2048, "naconmax": 2048, "njmax": 8192}
    clock = time.perf_counter()
    gpu_model = mjw.put_model(m)
    gpu = mjw.put_data(m, cpu, **kwargs)
    inputs = [wp.array(value, dtype=dtype) for value, dtype in (
        (dofs[:, 0], int), (parcel.hinges, wp.vec4i), (parcel.rest_angles, float),
        (parcel.hinge_stiffness, float), (parcel.hinge_damping, float))]
    wp.synchronize()
    upload_s = time.perf_counter() - clock
    memory_after_upload = memory_mib()
    rows = []
    wall_start = time.perf_counter()
    for step in range(10):
        robot.command(target, gap)
        gpu.ctrl.assign(cpu.ctrl.reshape(1, -1).astype(np.float32))
        mujoco.mj_step1(m, cpu)
        cpu_before = cpu.qfrc_passive[dofs].copy()
        parcel.apply_internal(m, cpu, dofs)
        cpu_hinge_force = cpu.qfrc_passive[dofs] - cpu_before
        mujoco.mj_step2(m, cpu)
        clock = time.perf_counter()
        mjw.step1(gpu_model, gpu)
        gpu_before = gpu.qfrc_passive.numpy()[0, dofs]
        wp.launch(bending, (1, len(parcel.hinges)),
                  [gpu.flexvert_xpos, gpu.qvel, *inputs, gpu.qfrc_passive])
        gpu_hinge_force = gpu.qfrc_passive.numpy()[0, dofs] - gpu_before
        mjw.step2(gpu_model, gpu)
        wp.synchronize()
        gpu_step_with_readback_s = time.perf_counter() - clock
        cpu_force = np.asarray(backend.measure(robot)["force_on_fingers_N"])
        gpu_force, ncon, counts = gpu_finger_forces(gpu_model, gpu, robot.all_finger_sets)
        gpu_qpos, gpu_qvel = gpu.qpos.numpy()[0], gpu.qvel.numpy()[0]
        # Distinct translational node DOFs and the corresponding qpos addresses.
        film_qpos = m.jnt_qposadr[m.body_jntadr[m.flex_vertbodyid, None] + np.arange(3)]
        rows.append({
            "step": step + 1,
            "finite": all(np.isfinite(a).all() for a in (cpu.ctrl, gpu.ctrl.numpy(), cpu.qpos, cpu.qvel,
                        gpu_qpos, gpu_qvel, cpu_force, gpu_force, cpu_hinge_force, gpu_hinge_force)),
            "cpu_time_s": float(cpu.time), "gpu_time_s": float(gpu.time.numpy()[0]),
            "cpu_ncon": int(cpu.ncon), "gpu_ncon": ncon,
            "gpu_active_finger_flex_contacts": counts,
            "gpu_overflow": int(gpu.overflow.numpy()[0]),
            "gpu_solver_iterations": int(gpu.solver_niter.numpy()[0]),
            "cpu_solver_iterations": int(cpu.solver_niter.max()),
            "ctrl": cpu.ctrl.tolist(),
            "cpu_finger_force_N": cpu_force.tolist(), "gpu_finger_force_N": gpu_force.tolist(),
            "finger_force_diff_N": np.linalg.norm(gpu_force - cpu_force, axis=1).tolist(),
            "cpu_hinge_force_L2_N": float(np.linalg.norm(cpu_hinge_force)),
            "gpu_hinge_force_relative_L2_error": float(
                np.linalg.norm(gpu_hinge_force - cpu_hinge_force) / max(np.linalg.norm(cpu_hinge_force), 1e-10)),
            "film_max_position_diff_m": float(np.linalg.norm((gpu_qpos - cpu.qpos)[film_qpos], axis=1).max()),
            "robot_max_joint_diff_rad": float(np.max(np.abs((gpu_qpos - cpu.qpos)[robot.qadr]))),
            "gpu_step_with_readback_diagnostic_s": gpu_step_with_readback_s,
        })
    start, end = rows[0], rows[-1]
    checks = {
        "finite_ten_steps": len(rows) == 10 and all(row["finite"] for row in rows),
        "zero_overflow": all(not row["gpu_overflow"] for row in rows),
        "first_hinge_force": start["gpu_hinge_force_relative_L2_error"] <= .005,
        "last_film_position": end["film_max_position_diff_m"] <= .0001,
        "last_robot_joints": end["robot_max_joint_diff_rad"] <= .001,
    }
    if label == "close":
        tolerance = np.maximum(.20, .25 * np.linalg.norm(start["cpu_finger_force_N"], axis=1))
        checks["first_two_finger_forces"] = bool(np.all(start["finger_force_diff_N"] <= tolerance))
    return {"checks": checks, "upload_s": upload_s,
            "device_mib_sampled_after_upload": memory_after_upload, "device_mib_sampled_after_steps": memory_mib(),
            "wall_s": time.perf_counter() - wall_start, "steps": rows}


def run():
    source = json.loads((ARCHIVE / "result.json").read_text())
    robot = backend.scene(backend.Config(**source["config"]))
    assembled = robot.sim.m
    model = mujoco.MjModel.from_binary_path(str(ARCHIVE / "model.mjb"))
    for kind in ("body", "joint", "geom", "actuator"):
        attr = {"body": "nbody", "joint": "njnt", "geom": "ngeom", "actuator": "nu"}[kind]
        assert [getattr(model, kind)(i).name for i in range(getattr(model, attr))] == [
            getattr(assembled, kind)(i).name for i in range(getattr(assembled, attr))], kind
    assert (model.nq, model.nv, model.nflexvert, model.nflexelem) == (615, 614, 200, 396)
    assert model.opt.timestep == 25e-6
    robot.sim.m = model
    robot.kin = mujoco.MjData(model)
    snapshots = {**source_files(), **backend.source_files(), "gpu_parcel": ROOT / "bag_sim/physical_parcel_gpu.py"}
    (RUN / "sources").mkdir()
    fingerprints = {}
    for name, path in snapshots.items():
        contents = Path(path).read_bytes()
        (RUN / "sources" / f"{name}.py").write_bytes(contents)
        fingerprints[name] = hashlib.sha256(contents).hexdigest()
    memory_start = memory_mib()
    result = {
        "host": platform.node(), "mujoco": mujoco.__version__, "mjwarp": mjw.__version__, "warp": wp.__version__,
        "model_sha256": hashlib.sha256((ARCHIVE / "model.mjb").read_bytes()).hexdigest(),
        "trajectory_sha256": hashlib.sha256((ARCHIVE / "trajectory.npz").read_bytes()).hexdigest(),
        "dimensions": {"nq": model.nq, "nv": model.nv, "nu": model.nu,
                       "film_vertices": model.nflexvert, "film_triangles": model.nflexelem},
        "dt_s": model.opt.timestep, "cold_device_mib_sampled_whole_card": memory_start,
        "asset_fingerprint": asset_fingerprint(), "sources_sha256": fingerprints,
        "timing_is_throughput": False, "cases": {},
    }
    with np.load(ARCHIVE / "trajectory.npz") as states, wp.ScopedDevice("cuda:0"):
        times = states["time"]
        for label, time_s in (("settle", .100), ("close", 2.740)):
            i = int(np.argmin(np.abs(times - time_s)))
            print(f"{label} frame {i} at {times[i]:.6f} s", flush=True)
            case = probe_case(robot, states["qpos"][i], label)
            result["cases"][label] = {"archive_time_s": float(times[i]), "archive_frame": i, **case}
            (RUN / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(case["checks"], flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        (RUN / "failure.txt").write_text(traceback.format_exc())
        raise

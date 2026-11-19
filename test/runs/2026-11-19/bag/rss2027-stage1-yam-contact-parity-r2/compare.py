"""Inventory and pair CPU/Warp contacts from one identical YAM starting state."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[4]
ARCHIVE = ROOT / "test/runs/2026-11-16/bag/physical-parcel-mj4-yam-video-r3/trial"
PREVIOUS = RUN.parent / "rss2027-stage1-yam-gpu-probe-r1" / "results.json"
sys.path[:0] = [str(ROOT / "test/bag_model"), str(ROOT), str(ROOT / "experience")]
import physical_parcel_yam as backend
from bag_sim.physical_parcel_gpu import bending

DISTANCE_LIMIT_M = .0001
ANGLE_LIMIT_DEG = 10.
FRIC_ROWS = ("pos", "margin", "D", "vel", "aref", "force", "state")
CONTACT_FIELDS = ("geom", "flex", "elem", "vert", "frame", "dist", "pos",
                  "dim", "includemargin", "solref", "solimp", "friction")


def record(m, contact, ident, address, world_force, local_force, constraint_rows):
    """Normalize normals and world reactions to the same exact object order."""
    objs = [("geom", int(g)) if g >= 0 else ("flex", int(f))
            for g, f in zip(contact["geom"], contact["flex"])]
    reverse = objs[0] > objs[1]
    sign = -1 if reverse else 1
    order = [1, 0] if reverse else [0, 1]
    pair = [objs[i] for i in order]
    obj_types = {"geom": mujoco.mjtObj.mjOBJ_GEOM, "flex": mujoco.mjtObj.mjOBJ_FLEX}
    names = [mujoco.mj_id2name(m, obj_types[kind], idx) or f"{kind}_{idx}" for kind, idx in pair]
    frame = np.asarray(contact["frame"]).reshape(3, 3)
    return {
        "id": ident, "objects": pair, "names": names,
        "elem": [int(contact["elem"][i]) for i in order],
        "vert": [int(contact["vert"][i]) for i in order],
        "solver_ids": np.atleast_1d(address).tolist(), "active": int(np.atleast_1d(address)[0]) >= 0,
        "local_force_N": np.asarray(local_force).tolist(), "force_on_second_N": (sign * world_force).tolist(),
        "position_m": np.asarray(contact["pos"]).tolist(), "normal": (sign * frame[0]).tolist(),
        "distance_m": float(contact["dist"]), "include_margin_m": float(contact["includemargin"]),
        "dim": int(contact["dim"]), "solref": np.asarray(contact["solref"]).tolist(),
        "solimp": np.asarray(contact["solimp"]).tolist(), "friction": np.asarray(contact["friction"]).tolist(),
        "constraint": constraint_rows,
    }


def cpu_records(m, d):
    rows = []
    for i, c in enumerate(d.contact):
        local_force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, local_force)
        world_force = c.frame.reshape(3, 3).T @ local_force[:3]
        address = c.efc_address
        indices = slice(address, address + c.dim)
        constraints = {field: getattr(d, f"efc_{field}")[indices].tolist()
                       for field in FRIC_ROWS} if address >= 0 else {}
        rows.append(record(m, {name: getattr(c, name) for name in CONTACT_FIELDS}, i,
                           address, world_force, local_force[:3], constraints))
    return rows


def gpu_records(m, w_model, d):
    count = int(d.nacon.numpy()[0])
    ids = wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32)
    world_force = wp.empty(count, dtype=wp.spatial_vector)
    local_force = wp.empty(count, dtype=wp.spatial_vector)
    mjw.contact_force(w_model, d, ids, True, world_force)
    mjw.contact_force(w_model, d, ids, False, local_force)
    world = world_force.numpy()[:, :3]
    local = local_force.numpy()[:, :3]
    values = {name: getattr(d.contact, name).numpy()[:count] for name in CONTACT_FIELDS}
    addresses = d.contact.efc_address.numpy()[:count]
    constraint_arrays = {name: getattr(d.efc, name).numpy()[0] for name in FRIC_ROWS}
    rows = []
    for i, addrs in enumerate(addresses):
        active = addrs[:int(values["dim"][i])]
        active = active[active >= 0]
        constraints = {key: value[active].tolist() for key, value in constraint_arrays.items()} if active.size else {}
        rows.append(record(m, {name: field[i] for name, field in values.items()}, i,
                           addrs, world[i], local[i], constraints))
    return rows


def dimensions(a, b):
    dist = float(np.linalg.norm(np.subtract(a["position_m"], b["position_m"])))
    angle = float(np.rad2deg(np.arccos(np.clip(np.dot(a["normal"], b["normal"]), -1., 1.))))
    return dist, angle


def match(cpu, gpu):
    """Match exact feature IDs, then mutually nearest geometries within each exact object pair."""
    candidates = []
    for a in cpu:
        for b in gpu:
            if a["objects"] != b["objects"]:
                continue
            distance, angle = dimensions(a, b)
            if distance > DISTANCE_LIMIT_M or angle > ANGLE_LIMIT_DEG:
                continue
            feature = (max(a["elem"] + a["vert"]) >= 0 and a["elem"] == b["elem"] and a["vert"] == b["vert"])
            candidates.append((distance, angle, a["id"], b["id"], feature))
    candidates.sort()
    pairs, used_cpu, used_gpu = [], set(), set()

    def add(entry, method):
        distance, angle, a_id, b_id, _ = entry
        pairs.append({"cpu": a_id, "gpu": b_id, "distance_m": distance, "angle_deg": angle, "method": method})
        used_cpu.add(a_id)
        used_gpu.add(b_id)

    for entry in candidates:
        if entry[4] and entry[2] not in used_cpu and entry[3] not in used_gpu:
            add(entry, "feature_id")
    remaining = [p for p in candidates if p[2] not in used_cpu and p[3] not in used_gpu]
    cpu_nearest, gpu_nearest = {}, {}
    for entry in remaining:
        cpu_nearest.setdefault(entry[2], entry)
        gpu_nearest.setdefault(entry[3], entry)
    for entry in remaining:
        if entry == cpu_nearest[entry[2]] == gpu_nearest[entry[3]]:
            add(entry, "mutual_nearest")
    return pairs, used_cpu, used_gpu


def group_object(obj, robot):
    kind, idx = obj
    if kind == "flex":
        return "bag"
    if idx in robot.all_finger_sets[0]:
        return "left_finger"
    if idx in robot.all_finger_sets[1]:
        return "right_finger"
    m = robot.sim.m
    name = m.geom(idx).name
    if name.startswith(("payload_geom", "plate_")):
        return "payload"
    if name in ("table", "floor"):
        return "support"
    if robot.robot_mask[idx]:
        return "other_robot"
    return name or f"geom_{idx}"


def classify(cpu, gpu, robot):
    counters = defaultdict(lambda: [0, 0, 0, 0])
    exact = defaultdict(lambda: [0, 0, 0, 0])
    for axis, rows in ((0, cpu), (2, gpu)):
        for entry in rows:
            group = tuple(sorted(group_object(obj, robot) for obj in entry["objects"]))
            pair = tuple(entry["names"])
            for values in (counters[group], exact[pair]):
                values[axis] += 1
                values[axis + 1] += int(entry["active"])

    def named(counts):
        return {" × ".join(key): {"cpu_candidates": a, "cpu_active": b, "gpu_candidates": c, "gpu_active": d}
                for key, (a, b, c, d) in sorted(counts.items())}

    return named(counters), named(exact)


def reactions(rows, robot, include):
    result = np.zeros((2, 3))
    for entry in rows:
        if not include(entry["id"]) or not entry["active"] or ("flex", 0) not in entry["objects"]:
            continue
        for side, geoms in enumerate(robot.all_finger_sets):
            for slot, (kind, idx) in enumerate(entry["objects"]):
                if kind == "geom" and idx in geoms:
                    result[side] += (2 * slot - 1) * np.asarray(entry["force_on_second_N"])
    return result


def source_fingerprints():
    paths = {**backend.source_files(), "gpu_model": ROOT / "bag_sim/physical_parcel_gpu.py",
             "controller": ROOT / "experience/yam_platform/robot.py",
             "compare": Path(__file__), "archive_model": ARCHIVE / "model.mjb",
             "archive_state": ARCHIVE / "trajectory.npz", "previous": PREVIOUS}
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()}


def main():
    if (RUN / "results.json").exists():
        raise FileExistsError("Contact comparison output already exists")
    prior = json.loads(PREVIOUS.read_text())["cases"]["close"]
    archive_cfg = json.loads((ARCHIVE / "result.json").read_text())["config"]
    robot = backend.scene(backend.Config(**archive_cfg))
    m = mujoco.MjModel.from_binary_path(str(ARCHIVE / "model.mjb"))
    assembled = robot.sim.m
    assert [m.geom(i).name for i in range(m.ngeom)] == [assembled.geom(i).name for i in range(assembled.ngeom)]
    robot.sim.m = m
    cpu = robot.sim.d = mujoco.MjData(m)
    with np.load(ARCHIVE / "trajectory.npz") as states:
        cpu.qpos[:] = states["qpos"][prior["archive_frame"]]
    mujoco.mj_forward(m, cpu)  # Prepare the same initial solver data as the last run.
    cpu.ctrl[:] = prior["steps"][0]["ctrl"]
    parcel = robot.sim.parcel
    dofs = parcel.node_dofs(m)
    with wp.ScopedDevice("cuda:0"):
        w_model = mjw.put_model(m)
        gpu = mjw.put_data(m, cpu, nworld=1, nconmax=2048, naconmax=2048, njmax=8192)
        exact_gpu_input = (np.array_equal(cpu.qpos.astype(np.float32), gpu.qpos.numpy()[0])
                           and np.array_equal(cpu.ctrl.astype(np.float32), gpu.ctrl.numpy()[0])
                           and not np.any(gpu.qvel.numpy()))
        wp_inputs = [wp.array(value, dtype=dtype) for value, dtype in (
            (dofs[:, 0], int), (parcel.hinges, wp.vec4i), (parcel.rest_angles, float),
            (parcel.hinge_stiffness, float), (parcel.hinge_damping, float))]
        mujoco.mj_step1(m, cpu)
        parcel.apply_internal(m, cpu, dofs)
        mujoco.mj_step2(m, cpu)
        mjw.step1(w_model, gpu)
        wp.launch(bending, (1, len(parcel.hinges)),
                  [gpu.flexvert_xpos, gpu.qvel, *wp_inputs, gpu.qfrc_passive])
        mjw.step2(w_model, gpu)
        wp.synchronize()
        overflow = int(gpu.overflow.numpy()[0])
        cpu_rows, gpu_rows = cpu_records(m, cpu), gpu_records(m, w_model, gpu)
        solver_iterations = [int(cpu.solver_niter.max()), int(gpu.solver_niter.numpy()[0])]
    matched, used_cpu, used_gpu = match(cpu_rows, gpu_rows)
    groups, pairs = classify(cpu_rows, gpu_rows, robot)
    cpu_force = reactions(cpu_rows, robot, lambda i: True)
    gpu_force = reactions(gpu_rows, robot, lambda i: True)
    cpu_matched = reactions(cpu_rows, robot, used_cpu.__contains__)
    gpu_matched = reactions(gpu_rows, robot, used_gpu.__contains__)
    same_first_step = prior["steps"][0]
    checks = {
        "same_gpu_input": exact_gpu_input,
        "zero_gpu_overflow": overflow == 0,
        "first_step_contact_counts_reproduced": len(cpu_rows) == same_first_step["cpu_ncon"]
            and len(gpu_rows) == same_first_step["gpu_ncon"],
        "first_step_force_reproduced": np.allclose(cpu_force, same_first_step["cpu_finger_force_N"], atol=.01, rtol=0)
            and np.allclose(gpu_force, same_first_step["gpu_finger_force_N"], atol=.01, rtol=0),
    }
    data = {
        "checks": checks, "hostname": platform.node(), "dt_s": m.opt.timestep,
        "versions": {"mujoco": mujoco.__version__, "warp": wp.__version__, "mujoco_warp": mjw.__version__},
        "archive_frame": prior["archive_frame"], "archive_time_s": prior["archive_time_s"],
        "source_sha256": source_fingerprints(), "gpu_overflow": overflow, "cpu_gpu_solver_iterations": solver_iterations,
        "contacts_by_object_group": groups, "contacts_by_exact_objects": pairs,
        "cpu_finger_force_N": cpu_force.tolist(), "gpu_finger_force_N": gpu_force.tolist(),
        "cpu_finger_matched_N": cpu_matched.tolist(), "gpu_finger_matched_N": gpu_matched.tolist(),
        "cpu_finger_unmatched_N": (cpu_force - cpu_matched).tolist(),
        "gpu_finger_unmatched_N": (gpu_force - gpu_matched).tolist(),
        "cpu_unmatched": sorted(set(range(len(cpu_rows))) - used_cpu),
        "gpu_unmatched": sorted(set(range(len(gpu_rows))) - used_gpu),
        "matched_contact_ids": matched, "cpu_contacts": cpu_rows, "gpu_contacts": gpu_rows,
    }
    (RUN / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"checks": checks, "contact_counts": groups, "matched_count": len(matched),
                      "cpu_minus_gpu_N": (cpu_force - gpu_force).tolist()}))
    return int(not all(checks.values()))


if __name__ == "__main__":
    raise SystemExit(main())

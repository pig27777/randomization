"""Compare CPU and Warp payload contacts at identical saved configurations."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from physical_parcel_checks import film_box_geometry, triangle_box_intersections


def payload_contacts(data, geom):
    contacts = []
    for contact in data.contact:
        if contact.efc_address < 0 or geom not in contact.geom or 0 not in contact.flex:
            continue
        slot = list(contact.flex).index(0)
        contacts.append(dict(element=int(contact.elem[slot]), vertex=int(contact.vert[slot]),
                             distance_m=float(contact.dist), point_m=contact.pos.tolist(),
                             normal_on_film=((2 * slot - 1) * contact.frame[:3]).tolist(),
                             normal_velocity_m_s=float(data.efc_vel[contact.efc_address])))
    return sorted(contacts, key=lambda c: (c["element"], c["vertex"], c["distance_m"]))


def device_payload_contacts(gpu, geom):
    count = int(gpu.nacon.numpy()[0])
    geoms = gpu.contact.geom.numpy()[:count]
    flex = gpu.contact.flex.numpy()[:count]
    addresses = gpu.contact.efc_address.numpy()[:count, 0]
    elements = gpu.contact.elem.numpy()[:count]
    vertices = gpu.contact.vert.numpy()[:count]
    distances = gpu.contact.dist.numpy()[:count]
    points = gpu.contact.pos.numpy()[:count]
    frames = gpu.contact.frame.numpy()[:count]
    velocities = gpu.efc.vel.numpy()[0]
    active = (geoms == geom).any(axis=1) & (flex == 0).any(axis=1) & (addresses >= 0)
    contacts = []
    for i in np.flatnonzero(active):
        slot = int(np.flatnonzero(flex[i] == 0)[0])
        contacts.append(dict(element=int(elements[i, slot]), vertex=int(vertices[i, slot]),
                             distance_m=float(distances[i]), point_m=points[i].tolist(),
                             normal_on_film=((2 * slot - 1) * frames[i, 0]).tolist(),
                             normal_velocity_m_s=float(velocities[addresses[i]])))
    return sorted(contacts, key=lambda c: (c["element"], c["vertex"], c["distance_m"]))


def compare_pose(model, warp_model, qpos, geom, watched, film_velocity):
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = qpos
    cpu.qvel[model.body_dofadr[model.flex_vertbodyid] + 2] = film_velocity
    mujoco.mj_forward(model, cpu)
    warp_data = mjw.put_data(model, cpu, nworld=1, nconmax=2048, naconmax=2048, njmax=8192)
    mjw.step1(warp_model, warp_data)
    if int(warp_data.overflow.numpy()[0]):
        raise RuntimeError("Warp contact buffers overflowed")
    host = mujoco.MjData(model)
    mjw.get_data_into(host, model, warp_data)
    if not np.isfinite(host.flexvert_xpos).all():
        raise RuntimeError("nonfinite exported film positions")

    faces = model.flex_elem.reshape(-1, 3)
    intersecting = np.flatnonzero(triangle_box_intersections(
        cpu.flexvert_xpos, faces, model.geom_size[geom], cpu.geom_xpos[geom],
        cpu.geom_xmat[geom].reshape(3, 3)))
    cpu_contacts = payload_contacts(cpu, geom)
    gpu_contacts = payload_contacts(host, geom)
    device_contacts = device_payload_contacts(warp_data, geom)
    elements = [{c["element"] for c in contacts if c["element"] >= 0} for contacts in (cpu_contacts, gpu_contacts)]
    return dict(
        geometry=film_box_geometry(model, cpu, faces, geom),
        intersecting_elements=intersecting.tolist(),
        cpu_only_elements=sorted(elements[0] - elements[1]), gpu_only_elements=sorted(elements[1] - elements[0]),
        watched_elements=sorted(watched | set(intersecting.tolist())),
        max_vertex_position_difference_m=float(np.max(np.linalg.norm(cpu.flexvert_xpos - host.flexvert_xpos, axis=1))),
        cpu_contacts=cpu_contacts, gpu_contacts=gpu_contacts, device_contacts=device_contacts,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("warp", type=Path, help="GPU recording directory containing trial/ and penetration-peaks.json")
    parser.add_argument("cpu", type=Path, help="reference CPU trial directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--film-vertical-velocity", type=float, default=0., help="uniform film velocity, in m/s")
    args = parser.parse_args()
    if not np.isfinite(args.film_vertical_velocity):
        parser.error("the added film velocity must be finite")
    trial = args.warp / "trial"
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    geom = model.geom("payload_geom_0").id
    peak = json.loads((args.warp / "penetration-peaks.json").read_text())[-1]
    watched = {side["element"] for side in peak["sides"] if "element" in side and side["element"] >= 0}
    rows = json.loads((trial / "trace.json").read_text())
    with np.load(trial / "trajectory.npz") as states:
        if not np.array_equal(states["time"], [row["t"] for row in rows]):
            raise ValueError("GPU state and trajectory sample times differ")
        t_failure = rows[-1]["t"]
        end_close = max(i for i, row in enumerate(rows) if row["phase"] == "close")
        frames = {"warp_end_close": (rows[end_close]["t"], states["qpos"][end_close]),
                  "warp_before_failure": (rows[-2]["t"], states["qpos"][-2]),
                  "warp_failure": (t_failure, states["final_qpos"])}
    with np.load(args.cpu / "trajectory.npz") as states:
        index = int(np.argmin(np.abs(states["time"] - t_failure)))
        frames["cpu_near_failure_time"] = (float(states["time"][index]), states["qpos"][index])

    with wp.ScopedDevice("cuda:0"):
        device_model = mjw.put_model(model)
        result = {name: dict(t_s=t, **compare_pose(model, device_model, qpos, geom, watched,
                                                  args.film_vertical_velocity))
                  for name, (t, qpos) in frames.items()}
    provenance = dict(warp_recording=str(args.warp), cpu_trial=str(args.cpu),
                      imposed_film_vertical_velocity_m_s=args.film_vertical_velocity,
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), frames=result)
    with args.output.open("x") as stream:
        stream.write(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()

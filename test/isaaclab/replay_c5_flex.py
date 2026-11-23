"""Replay measured C5 finger motion on the unified YAM / MuJoCo flex platform."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "test/bag_model")]
from physical_parcel_yam import Config as Trial, ParcelSim, attach_robot, compile_yam_model, measure
from bag_sim.physical_parcel import Config, Parcel, parcel_profile, vec


def scene():
    center = (-.03, -.225)
    sheet = (.28, .22)
    content = (.18, .60 * .24 * .18 * .056 / (.18 * .054), .054)
    material = parcel_profile("pe")["shell"]
    film_mass = 2 * np.prod(sheet) * material["areal_mass"]
    parcel = Parcel(Config(mode="compressible", geometry="geometry/kraft_edge_sealed_35.npz",
        bag_size=sheet, payload_size=content, payload_mass=.50-film_mass, center=center,
        timestep=.0002, **material))
    world_xml = ET.fromstring(parcel.xml)
    world = world_xml.find("worldbody")
    world.remove(world.find("body[@name='contact_probe']"))
    cfg = Trial(dt=parcel.cfg.timestep, tilt=0., yaw=-90., contact_time=.001, pickup_only=True)
    built = SimpleNamespace(parcel=parcel, cfg=cfg, desc=SimpleNamespace(), xml=ET.tostring(world_xml, encoding="unicode"))
    abc = json.loads((ROOT / "configs/route_b_yam_scene.json").read_text())

    def compile_workcell(xml):
        tree = ET.fromstring(xml)
        world = tree.find("worldbody")
        world.find("body[@name='arm']").set("pos", vec(abc["arm_bases_m"][1]))
        world.find("body[@name='arm']").set("quat", "1 0 0 0")
        world.remove(world.find("geom[@name='yam_pedestal']"))
        for side in ("left", "right"):
            body = tree.find(f".//body[@name='{side}_pad']")
            for geom in list(body.findall("geom")):
                if geom.get("name") == f"{side}_pad":
                    geom.set("pos", "0 -.0017 -.1362")
                    geom.set("size", ".0072 .0017 .0082")
                else:
                    body.remove(geom)
        return compile_yam_model(parcel, built, ET.tostring(tree, encoding="unicode"))

    built.compile_model = compile_workcell
    return attach_robot(cfg, built, ParcelSim, np.array(center), .075, np.r_[sheet, content[2]], .5)


def matrices(xyzw):
    matrices = np.empty((len(xyzw), 3, 3))
    for m, q in zip(matrices, xyzw, strict=True):
        mj.mju_quat2Mat(m.ravel(), np.roll(q, 1))
    return matrices


def registered_joints(robot, trace):
    """Register the hardware variants by their two CAD faces on the shared clock."""
    cfg = json.loads((ROOT / "configs/route_b_yam_scene.json").read_text())
    faces = np.array([p["box_center_m"] for p in cfg["pads"]])
    faces[:, 1] += np.array([1, -1]) * cfg["pad_box_size_m"][1] / 2
    poses = trace[:, 13:].reshape(-1, 2, 7)
    rotations = matrices(poses[..., 3:].reshape(-1, 4)).reshape(-1, 2, 3, 3)
    wanted = poses[..., :3] + (rotations @ faces[..., None]).squeeze(-1)
    m, data = robot.sim.m, robot.kin
    joint_rows = trace[:, :8].copy()
    # Menagerie slides measure opening, whereas ABC measures closing.
    joint_rows[:, 6:] *= -1
    ids = np.r_[robot.qadr, robot.fingers]

    def points():
        return np.stack([data.body(s+"_pad").xpos + data.body(s+"_pad").xmat.reshape(3,3) @ [0,0,-.1362]
                         for s in ("left", "right")])

    # ABC's joint-2 axis is 113.5 mm above its base; Menagerie's is
    # 104 mm. Keep the table/base fixed and register the chain at its mount.
    m.body_pos[m.body("link_1").id, 2] += .0095
    data.qpos[ids] = joint_rows[0]
    mj.mj_kinematics(m, data)
    offsets = wanted[0] - points()
    for shift, name in zip(offsets, ("left_finger", "right_finger"), strict=True):
        body = m.body(name).id
        parent = m.body_parentid[body]
        m.body_pos[body] += data.xmat[parent].reshape(3, 3).T @ shift
    mj.mj_setConst(m, data)
    max_error = 0.
    for row, pair in zip(joint_rows, wanted, strict=True):
        data.qpos[ids] = row
        mj.mj_kinematics(m, data)
        max_error = max(max_error, np.linalg.norm(points() - pair, axis=1).max())
    if max_error > .0005:
        raise ValueError(f"Pad registration error {1000*max_error:.3f} mm")
    return joint_rows, max_error


def run(geometry, output):
    ref = np.load(geometry)
    trace = ref["substeps"]
    last = trace[-1].copy()
    last[:8] = ref["final_q"]
    last[13:] = ref["finger_pose"][-1].ravel()
    robot = scene()
    q, err = registered_joints(robot, np.vstack((trace, last)))
    m, d = robot.sim.m, robot.sim.d
    qids = np.r_[robot.qadr, robot.fingers]
    dofs = m.jnt_dofadr[[*robot.joints, m.joint("left_finger").id, m.joint("right_finger").id]]
    vertices, contacts = [], []
    for i in range(len(trace)*10 + 1):
        index, sub = divmod(i, 10)
        delta = q[min(index+1, len(q)-1)] - q[index]
        d.qpos[qids] = q[index] + sub/10 * delta
        d.qvel[dofs] = delta/.002
        d.ctrl[robot.act] = d.qpos[robot.qadr]
        d.ctrl[robot.gripper] = d.qpos[robot.fingers].mean()
        if i and i % 200 == 0:
            mj.mj_forward(m, d)
            vertices.append(d.flexvert_xpos.copy())
            contacts.append(measure(robot)["penetration_mm"])
        if i < len(trace)*10:
            robot.sim.step()
        if d.warning.number.any():
            raise RuntimeError(f"MuJoCo solver warning at step {i}")
    p = robot.sim.parcel
    np.savez_compressed(output, vertices=vertices, triangles=p.envelope["faces"], penetration_mm=contacts)
    output.with_suffix(".json").write_text(json.dumps(dict(engine="MuJoCo flex", dt=p.cfg.timestep,
        trajectory_sha256=hashlib.sha256(trace.tobytes()).hexdigest(),
        pad_registration_max_mm=err*1000, initial_film_size_m=np.ptp(p.envelope["current"],axis=0).tolist(),
        sheet_m=p.cfg.bag_size, content_m=p.cfg.payload_size, mass_kg=p.cfg.payload_mass+p.envelope["masses"].sum(),
        peak_sampled_penetration_mm=max(contacts), frames=len(vertices)),indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    run(args.geometry,args.output)

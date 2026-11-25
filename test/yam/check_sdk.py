"""Offline FK comparison against pinned official YAM v1 and linear_4310 XML."""

import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from yam_platform.robot import rotation_error
from yam_platform.yam_model import ARM_JOINTS, build_yam_model, load_config

REV = "ea116fe4079e991958b22f879874402674ee6590"
RAW = f"https://raw.githubusercontent.com/i2rt-robotics/i2rt/{REV}/i2rt/robot_models"
FILES = ("arm/yam/v1/yam.xml", "gripper/linear_4310/linear_4310.xml")


def compare():
    content = {}
    for name in FILES:
        with urlopen(f"{RAW}/{name}", timeout=15) as source:
            content[name] = source.read()
    arm, linear = (ET.fromstring(content[name]) for name in FILES)
    arm.remove(arm.find("asset"))
    for body in arm.iter("body"):
        for g in body.findall("geom"):
            body.remove(g)
    actuator = arm.find("actuator")
    if actuator is not None:
        arm.remove(actuator)
    cfg = load_config("nominal")
    base = arm.find("worldbody/body")
    base.set("pos", " ".join(map(str, cfg["base_m"])))
    base.set("quat", " ".join(map(str, cfg["base_quat_wxyz"])))
    mount = arm.find(".//body[@name='gripper']")
    gripper = linear.find("worldbody/body")
    offset = ET.SubElement(mount, "frame", pos=gripper.attrib["pos"], quat=gripper.attrib["quat"])
    offset.extend(gripper.findall("site"))
    reference = mujoco.MjModel.from_xml_string(ET.tostring(arm, encoding="unicode"))
    _, model = build_yam_model()
    nominal_ranges = np.array([model.joint(j).range for j in ARM_JOINTS])
    official_ranges = np.array([reference.joint(j).range for j in ARM_JOINTS])
    assert np.array_equal(nominal_ranges, official_ranges)
    d, d_ref = mujoco.MjData(model), mujoco.MjData(reference)
    joint_rows, tool_rows = [], []
    for q in ([0.] * 6, cfg["ready_joints_rad"], [-.2, 2.2, 1.2, -1.3, .34, .16]):
        for name, value in zip(ARM_JOINTS, q, strict=True):
            d.joint(name).qpos[:] = value
            d_ref.joint(name).qpos[:] = value
        mujoco.mj_kinematics(model, d)
        mujoco.mj_kinematics(reference, d_ref)
        axes = d_ref.xaxis[:6]
        displacement = d.xanchor[:6] - d_ref.xanchor[:6]
        joint_rows.append(dict(axes_dot=np.sum(d.xaxis[:6] * axes, axis=1).tolist(),
                               axis_line_offset_mm=(1000 * np.linalg.norm(np.cross(displacement, axes), axis=1)).tolist()))
        sim, sdk = d.site("grasp_site"), d_ref.site("grasp_site")
        r, r_ref = sim.xmat.reshape(3, 3), sdk.xmat.reshape(3, 3)
        tool_rows.append(dict(sdk_origin_in_sim_tool_mm=(1000 * r.T @ (sdk.xpos - sim.xpos)).tolist(),
                              sdk_axes_in_sim_tool=(r.T @ r_ref).tolist(),
                              tool_angle_difference_deg=float(np.rad2deg(np.linalg.norm(rotation_error(r, r_ref))))))
    return dict(commit=REV, xml_sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in content.items()},
                matching_limits=True, joints=joint_rows, tools=tool_rows)


if __name__ == "__main__":
    with Path(sys.argv[1]).open("x") as out:
        json.dump(compare(), out, indent=2)
        out.write("\n")

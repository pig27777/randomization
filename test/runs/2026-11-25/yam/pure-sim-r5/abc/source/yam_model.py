"""Single Yam_B from the platform's Menagerie asset, with direct effort inputs."""

import json
from os.path import relpath
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from .robot import ASSET

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
RAILS = ("left_finger", "right_finger")
HERE = Path(__file__).resolve().parent


def load_config(name, overrides=None):
    config = json.loads((HERE / f"yam_{name}.json").read_text())
    if overrides:
        if unknown := overrides.keys() - config.keys():
            raise ValueError(f"Unknown YAM {name} parameters: {sorted(unknown)}")
        config.update(overrides)
    return config


def _vec(values):
    return " ".join(map(str, values))


def build_yam_model(config=None, *, add_scene=None, compile_model=mujoco.MjModel.from_xml_string):
    """Return native MJCF and its compiled model. add_scene(root) may append a fixture."""
    cfg = load_config("nominal", config)
    root = ET.parse(ASSET / "yam_linear_4310.xml").getroot()
    root.set("model", "Yam_B")
    root.find("compiler").set("meshdir", str(ASSET / "assets"))
    root.find("option").set("timestep", str(cfg["physics_dt_s"]))
    root.find("option").set("cone", "elliptic")
    arm = root.find("worldbody/body[@name='arm']")
    arm.set("pos", _vec(cfg["base_m"]))
    arm.set("quat", _vec(cfg["base_quat_wxyz"]))
    for name, friction in zip(ARM_JOINTS, cfg["joint_frictionloss_Nm"], strict=True):
        arm.find(f".//joint[@name='{name}']").set("frictionloss", str(friction))
    # The upstream gravity flags identify the SDK's compensated masses. The
    # controller applies this gravity term exactly once, through the motors.
    gravity_bodies = [b.get("name") for b in arm.iter("body") if b.attrib.pop("gravcomp", None) == "1"]
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "text", name="gravity_bodies", data=" ".join(gravity_bodies))

    actuators = root.find("actuator")
    actuators.clear()
    for name, limit in zip(ARM_JOINTS, cfg["torque_limit_Nm"], strict=True):
        ET.SubElement(actuators, "motor", name=name, joint=name, ctrlrange=f"{-limit} {limit}")
    # With tied rails, the gap is twice one rail displacement. Gear 2 makes
    # ctrl the generalized force on the full gap, equal to the ideal force per pad.
    limit = cfg["gripper_force_limit_N"]
    ET.SubElement(actuators, "motor", name="gripper", joint=RAILS[0], gear="2", ctrlrange=f"{-limit} {limit}")
    if add_scene is not None:
        add_scene(root)
    xml = ET.tostring(root, encoding="unicode")
    model = compile_model(xml)
    return xml, model


def save_yam_xml(xml, destination):
    """Save a replay model with mesh references relative to the output file."""
    root = ET.fromstring(xml)
    compiler = root.find("compiler")
    compiler.set("meshdir", relpath(compiler.attrib["meshdir"], destination.parent))
    destination.write_text(ET.tostring(root, encoding="unicode"))


def initial_state(model, config):
    data = mujoco.MjData(model)
    for name, target in zip(ARM_JOINTS, config["ready_joints_rad"], strict=True):
        if not model.joint(name).range[0] <= target <= model.joint(name).range[1]:
            raise ValueError(f"Initial {name} outside model range")
        data.joint(name).qpos[:] = target
    for name in RAILS:
        data.joint(name).qpos[:] = config["ready_gap_m"] / 2
    mujoco.mj_forward(model, data)
    return data


def arm_indices(model):
    """Resolve policy order even if the scene inserts unrelated joints."""
    joints = np.array([model.joint(name).id for name in (*ARM_JOINTS, *RAILS)], dtype=np.int32)
    return joints, model.jnt_qposadr[joints], model.jnt_dofadr[joints]

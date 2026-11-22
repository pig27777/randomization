"""ABC workcell and linear 4310 frames for the Route B YAM URDF."""

from copy import deepcopy
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET


SCENE_FILE = Path(__file__).resolve().parents[2] / "configs" / "route_b_yam_scene.json"
YAM_USD = Path(__file__).with_name("assets") / "yam.usdc"
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
RAIL_JOINTS = ("joint7", "joint8")


def load_scene():
    """Load table-relative positions and local optical frames (right, down, forward)."""
    return json.loads(SCENE_FILE.read_text())


def prepare_urdf(source: Path, destination: Path, scene):
    """Add housing and pad collisions to a conversion input without altering the source."""
    root = ET.parse(source).getroot()
    for mesh in root.findall(".//mesh"):
        path = (source.parent / mesh.attrib["filename"]).resolve(strict=True)
        mesh.set("filename", str(path))

    for link in root.findall("link"):
        if link.find("collision") is not None:
            continue
        for visual in link.findall("visual"):
            collision = ET.SubElement(link, "collision")
            for element in visual:
                if element.tag in ("origin", "geometry"):
                    collision.append(deepcopy(element))

    for pad in scene["pads"]:
        parent = root.find(f"link[@name='{pad['parent']}']")
        if parent is None:
            raise ValueError(f"Missing pad parent in {source}: {pad['parent']}")
        collision = ET.SubElement(parent, "collision", name=f"{pad['parent']}_pad")
        ET.SubElement(collision, "origin", xyz=_xyz(pad["box_center_m"]),
                      rpy=_xyz((0., 0., pad["box_yaw_rad"])))
        ET.SubElement(ET.SubElement(collision, "geometry"), "box", size=_xyz(scene["pad_box_size_m"]))
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def filter_gripper_collisions(usd):
    """Exclude the mating rail housings as in linear 4310, keeping pad-to-pad contact."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd))
    shapes = {prim.GetName(): prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)}
    for parent, mates in (
        ("gripper_1", ("tip_left_1", "tip_right_1", "tip_left_pad", "tip_right_pad")),
        ("tip_left_1", ("tip_right_1", "tip_right_pad")),
        ("tip_right_1", ("tip_left_pad",)),
    ):
        exclude = UsdPhysics.FilteredPairsAPI.Apply(shapes[parent]).CreateFilteredPairsRel()
        exclude.SetTargets([shapes[mate].GetPath() for mate in mates])
    stage.GetRootLayer().Save()


def _xyz(values):
    return " ".join(map(str, values))


def arm_cfg(index, scene, usd, prim_path):
    """Build a fixed-base I2RT articulation for a single or batched scene."""
    from isaaclab import sim as sim_utils
    from isaaclab.actuators import IdealPDActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    def drive(names, stiffness, damping, effort):
        return IdealPDActuatorCfg(
            joint_names_expr=list(names), stiffness=stiffness, damping=damping,
            effort_limit=effort, effort_limit_sim=effort, velocity_limit_sim=1.,
        )

    initial = scene["ready_arm_joints_rad"][index] + [scene["ready_rail_m"]] * 2
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd), fix_root_link=True,
            articulation_props=[PhysxArticulationCfg(enabled_self_collisions=True)],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(scene["arm_bases_m"][index]), rot=_xyzw(scene["arm_base_quat_wxyz"]),
            joint_pos=dict(zip(ARM_JOINTS + RAIL_JOINTS, initial, strict=True)),
        ),
        actuators={
            "proximal": drive(ARM_JOINTS[:3], 80., 5., 28.),
            "distal": drive(ARM_JOINTS[3:], 10., 1.5, 10.),
            # Near critical for the 69.657 g URDF sliders at 2 ms with explicit PD.
            "rails": drive(RAIL_JOINTS, 1000., 16.7, 50.),
        },
    )


def table_cfg(scene, prim_path):
    """Build the collision-only static table at the parcel kernel's plane."""
    from isaaclab import sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    table = scene["table"]
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(size=tuple(table["size_m"]),
                                  collision_props=sim_utils.UsdPhysicsCollisionCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(table["center_m"])),
    )


def abc_scene_cfg(num_envs, *, dual_arm, usd=YAM_USD, env_spacing=2.0):
    """Use the same ABC workcell in single- and dual-arm DirectRLEnv batches."""
    from isaaclab.scene import InteractiveSceneCfg

    usd = Path(usd).resolve(strict=True)
    scene = load_scene()
    cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    cfg.table = table_cfg(scene, "{ENV_REGEX_NS}/Table")
    cfg.arm_a = arm_cfg(0, scene, usd, "{ENV_REGEX_NS}/Yam_A")
    if dual_arm:
        cfg.arm_b = arm_cfg(1, scene, usd, "{ENV_REGEX_NS}/Yam_B")
    return cfg


class YamSceneFrames:
    """Project the ABC optical and linear 4310 contact frames on the PhysX device."""

    def __init__(self, scene, body_names, like):
        import torch

        wrist = scene["wrist_camera"]
        pads = scene["pads"]
        self.body_ids = torch.tensor(
            [body_names.index(frame["parent"]) for frame in (wrist, *pads)],
            dtype=torch.long, device=like.device,
        )
        self.wrist_quat = like.new_tensor(_xyzw(wrist["quat_wxyz"]))
        self.top_position = like.new_tensor(scene["top_camera"]["position_m"])
        self.top_quat = like.new_tensor(_xyzw(scene["top_camera"]["quat_wxyz"]))
        half_pad = .5 * scene["pad_box_size_m"][1]
        offsets = [wrist["position_m"]]
        for pad in pads:
            x, y, z = pad["box_center_m"]
            yaw = pad["box_yaw_rad"]
            offsets.append([x - half_pad * math.sin(yaw), y + half_pad * math.cos(yaw), z])
        self.offsets = like.new_tensor(offsets)

    def top_camera(self, batch):
        from isaaclab.utils.math import matrix_from_quat
        from src.low_order_parcel.virtual_rgb import CameraPose

        return CameraPose(self.top_position.expand(batch, -1),
                          matrix_from_quat(self.top_quat).expand(batch, -1, -1))

    def wrist_camera_and_pads(self, link_pose_w, env_origins):
        from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_mul
        from src.low_order_parcel.virtual_rgb import CameraPose

        links = link_pose_w.index_select(1, self.body_ids)
        pos, quat = links.split((3, 4), dim=-1)
        relative = pos - env_origins[:, None] + quat_apply(quat, self.offsets)
        optical_quat = quat_mul(quat[:, 0], self.wrist_quat.expand_as(quat[:, 0]))
        return CameraPose(relative[:, 0], matrix_from_quat(optical_quat)), relative[:, 1:]


def _xyzw(quaternion_wxyz):
    w, x, y, z = quaternion_wxyz
    return x, y, z, w


if __name__ == "__main__":
    import argparse
    from tempfile import TemporaryDirectory

    from urdf_usd_converter import Converter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix="abc_yam_") as scratch:
        urdf = Path(scratch) / "yam.urdf"
        prepare_urdf(args.urdf, urdf, load_scene())
        asset = Converter(layer_structure=False, scene=False).convert(urdf, args.output_dir)
        filter_gripper_collisions(asset.path)
        print(asset.path, flush=True)

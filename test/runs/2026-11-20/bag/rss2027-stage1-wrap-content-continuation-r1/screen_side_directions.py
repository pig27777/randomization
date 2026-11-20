"""Screen the same 60% parcel from alternative sides for a reachable YAM path."""

import json
from contextlib import nullcontext
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
CASE = RUN / "equilibrated-entry/sample-061"
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-top-local-mesh-r1")))
from refine_parcel import backend, physical_parcel, prepared_wrap  # noqa: E402


def shifted_target(translation):
    attach = backend.attach_robot

    def shifted(*args, approach=None, **kwargs):
        if approach is None:
            raise ValueError("A translated approach requires a side grasp")
        approach = approach | {"contact_point": np.asarray(approach["contact_point"]) + translation}
        return attach(*args, approach=approach, **kwargs)

    return patch.object(backend, "attach_robot", shifted)


def screen(config, translation=None):
    cfg = backend.Config(**config)
    placement = shifted_target(translation) if translation is not None else nullcontext()
    with placement, patch.object(physical_parcel, "wrap_generic", prepared_wrap(cfg.geometry)):
        robot = backend.scene(cfg)
    start = robot.initial_target
    insertion = start + .110 * robot.approach
    targets = np.vstack((np.linspace(start, insertion, 34),
                         np.linspace(insertion, insertion + [0., 0., cfg.lift_height], 71)[1:]))
    jacp, jacr = np.zeros((3, robot.sim.m.nv)), np.zeros((3, robot.sim.m.nv))
    rows = []
    for target in targets:
        result = robot.solve(target)
        robot.residual(robot.qtarget, target)
        mujoco.mj_jacSite(robot.sim.m, robot.kin, jacp, jacr, robot.site)
        j = np.vstack((4. * jacp[:, robot.dofs], jacr[:, robot.dofs]))
        margin = np.minimum(robot.qtarget - robot.lo, robot.hi - robot.qtarget)
        robot.kin.qpos[robot.fingers] = cfg.grasp_opening_m / 2.
        mujoco.mj_fwdPosition(robot.sim.m, robot.kin)
        self_contact = []
        for contact in robot.kin.contact:
            a, b = contact.geom
            if a >= 0 and b >= 0 and robot.robot_mask[a] and robot.robot_mask[b] and contact.dist < -1e-6:
                self_contact.append((float(-contact.dist * 1000.), robot.sim.m.geom(a).name, robot.sim.m.geom(b).name))
        worst = max(self_contact, default=None)
        rows.append(dict(target_m=target.tolist(), joint_target_rad=robot.qtarget.tolist(),
                         smallest_jacobian_value=float(np.linalg.svd(j, compute_uv=False)[-1]),
                         joint_margin_rad=margin.tolist(), self_contact=worst, **result))
    valid = all(row["position_mm"] <= 1. and row["rotation_deg"] <= 1. for row in rows)
    worst = max((row["self_contact"] for row in rows if row["self_contact"] is not None), default=None)
    status = "path_unreachable" if not valid else "arm_self_collision" if worst else "reachable"
    wrist_margin = float(np.min(np.array([row["joint_margin_rad"] for row in rows])[:, 3:]))
    return dict(status=status, worst_arm_self_collision=worst,
                minimum_wrist_joint_margin_rad=wrist_margin,
                min_jacobian_value=min(row["smallest_jacobian_value"] for row in rows),
                maximum_translation_error_mm=max(row["position_mm"] for row in rows),
                maximum_orientation_error_deg=max(row["rotation_deg"] for row in rows), trace=rows)


def main():
    config = json.loads((CASE / "side-top-edge/config.json").read_bytes())
    results = []
    orientations = [(float(yaw), tilt) for yaw in range(70, 116, 5) for tilt in (5., 15., 25.)]
    for yaw, tilt in orientations:
        try:
            result = screen(config | {"yaw": yaw, "tilt": tilt})
        except RuntimeError as error:
            if not str(error).startswith(("YAM initial pose unreachable", "initial film contact penetration")):
                raise
            result = dict(status="initial_pose_unavailable", reason=str(error))
        results.append(dict(yaw_deg=yaw, tilt_deg=tilt, **result))
    valid = [row for row in results if row["status"] == "reachable"]
    choice = max(valid, key=lambda row: (row["minimum_wrist_joint_margin_rad"], row["min_jacobian_value"]), default=None)
    case = CASE / "side-wrist-clearance"
    case.mkdir(exist_ok=True)
    report = dict(original_yaw_deg=config["yaw"], original_tilt_deg=config["tilt"],
                  selected=choice, paths=[{k: v for k, v in row.items() if k != "trace"} for row in results])
    with (case / "collision-screen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if choice:
        for name, seconds in (("config", 0.), ("smoke", .02)):
            cfg = config | {"yaw": choice["yaw_deg"], "tilt": choice["tilt_deg"], "smoke_seconds": seconds}
            with (case / f"{name}.json").open("x") as stream:
                stream.write(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    main()

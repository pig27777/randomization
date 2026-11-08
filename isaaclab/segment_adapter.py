"""L2 native D6 construction probe using the Isaac Lab Newton builder.

The probe preserves every grid edge and records unsupported loop mappings. A
successful allocation alone must not be counted as a complete model migration.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import warnings

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager


def add_content(builder, desc, loop_mode="native"):
    """Add the frozen description with a spanning articulation and all loop edges."""
    if desc["mobile"] is not None:
        raise NotImplementedError("L2 anchors have no mobile mass; mobile coupling is not implemented")
    segments, joints = desc["segments"], desc["joints"]
    offset = np.array([0.0, 0.0, desc["kappa"]["size"][2] / 2 + 1e-4])
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, ke=3e5)
    bodies, shapes = [], []
    for i, segment in enumerate(segments):
        body = builder.add_link(
            xform=wp.transform(np.asarray(segment["pos"]) + offset, wp.quat_identity()),
            mass=segment["mass"], com=wp.vec3(*segment["com"]),
            inertia=wp.mat33(*np.diag(segment["inertia"]).ravel()),
            lock_inertia=True, label=f"segment_{i}",
        )
        bodies.append(body)
        shapes.append(builder.add_shape_box(body, hx=segment["half"][0], hy=segment["half"][1],
                                            hz=segment["half"][2], cfg=cfg))
    adjacency = [[] for _ in segments]
    for index, joint in enumerate(joints):
        adjacency[joint["a"]].append((joint["b"], index))
        adjacency[joint["b"]].append((joint["a"], index))
    tree, seen = [], {0}

    def visit(parent):
        for child, index in adjacency[parent]:
            if child not in seen:
                seen.add(child)
                tree.append((parent, child, index))
                visit(child)

    visit(0)
    if len(seen) != len(segments):
        raise ValueError("description graph is disconnected")

    def add_joint(parent, child, index, child_body=None):
        joint = joints[index]
        pa, pb = np.array(segments[parent]["pos"]), np.array(segments[child]["pos"])
        midpoint = (pa + pb) / 2
        kwargs = dict(parent_xform=wp.transform(midpoint - pa, wp.quat_identity()),
                      child_xform=wp.transform(midpoint - pb, wp.quat_identity()), label=f"edge_{index}")
        if joint["fixed"]:
            return builder.add_joint_fixed(bodies[parent], bodies[child] if child_body is None else child_body, **kwargs)
        axis = joint["axis"]
        transverse = [a for a in range(3) if a != axis]
        linear, angular = [], []
        for a in range(3):
            vector = np.eye(3)[a]
            if a == axis:
                vector *= np.sign(pb[a] - pa[a])
                stiffness, lower, upper = joint["k_comp"], -joint["lock_comp"] * joint["rest"], 1e6
                rotation_k, inertia = joint["k_twist"], joint["i_eff"][2]
                angle_limit = 1e6
            else:
                index_t = transverse.index(a)
                stiffness, lower, upper = joint["k_shear"][index_t], -1e6, 1e6
                rotation_k, inertia = joint["k_bend"][index_t], joint["i_eff"][index_t]
                angle_limit = joint["lock_bend"]
            linear.append(builder.JointDofConfig(
                axis=wp.vec3(*vector), limit_lower=lower, limit_upper=upper,
                target_ke=stiffness, damping=2 * joint["damping_ratio"] * math.sqrt(stiffness * joint["m_eff"]),
            ))
            angular.append(builder.JointDofConfig(
                axis=wp.vec3(*np.eye(3)[a]), limit_lower=-angle_limit, limit_upper=angle_limit,
                target_ke=rotation_k, damping=2 * joint["damping_ratio"] * math.sqrt(rotation_k * inertia),
            ))
        return builder.add_joint_d6(bodies[parent], bodies[child] if child_body is None else child_body,
                                    linear_axes=linear, angular_axes=angular, **kwargs)

    tree_ids = [builder.add_joint_free(bodies[0], label="content_free")]
    tree_indices = {index for _, _, index in tree}
    loops = [(index, joint) for index, joint in enumerate(joints) if index not in tree_indices]
    proxies = {}
    if loop_mode == "proxy":
        for index, joint in loops:
            if not joint["fixed"]:
                proxies[index] = builder.add_link(
                    xform=wp.transform(np.asarray(segments[joint["b"]]["pos"]) + offset, wp.quat_identity()),
                    mass=1e-5, inertia=wp.mat33(*np.diag([1e-9] * 3).ravel()),
                    lock_inertia=True, label=f"loop_proxy_{index}")
    children = [[] for _ in segments]
    for parent, child, index in tree:
        children[parent].append((child, index))

    def add_branch(parent):
        for index, joint in loops:
            if index in proxies and joint["a"] == parent:
                tree_ids.append(add_joint(parent, joint["b"], index, proxies[index]))
        for child, index in children[parent]:
            tree_ids.append(add_joint(parent, child, index))
            add_branch(child)

    add_branch(0)
    builder.add_articulation(tree_ids, label="content_tree")
    for index, proxy in proxies.items():
        # Public custom-value path for MuJoCo-specific equality rows; WELD enum = 1.
        builder.add_custom_values(**{
            "mujoco:equality_constraint_type": 1,
            "mujoco:equality_constraint_objtype": 1,
            "mujoco:equality_constraint_body1": proxy,
            "mujoco:equality_constraint_body2": bodies[joints[index]["b"]],
            "mujoco:equality_constraint_torquescale": 1.0,
            "mujoco:equality_constraint_relpose": wp.transform_identity(),
            "mujoco:equality_constraint_world": builder.current_world,
            "mujoco:equality_constraint_label": f"loop_weld_{index}",
            "mujoco:eq_solref": wp.vec2(0.001, 2.0),
            "mujoco:eq_solimp": wp.types.vector(length=5, dtype=wp.float32)(0.95, 0.99, 0.0005, 0.5, 2.0),
        })
    loop_ids = ([] if loop_mode == "tree" else
                [add_joint(joint["a"], joint["b"], index) for index, joint in loops if index not in proxies])
    builder.shape_collision_filter_pairs.extend((shapes[a], shapes[b]) for a, b in desc["exclude"])
    return {"bodies": bodies, "tree_edges": len(tree), "loop_edges": len(loops),
            "expected_edges": len(joints), "fixed_edges": sum(j["fixed"] for j in joints),
            "loop_mode": loop_mode, "proxy_bodies": len(proxies), "proxy_welds": len(proxies),
            "proxy_added_mass_kg": len(proxies) * 1e-5,
            "represented_edges": len(tree) if loop_mode == "tree" else len(joints)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loop-mode", choices=("native", "proxy"), default="native")
    parser.add_argument("--rest", action="store_true")
    parser.add_argument("--njmax", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    desc = json.loads(args.description.read_text())["description"]
    wp.config.enable_backward = False
    cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(integrator="implicitfast"))
    builder = NewtonMJWarpManager.create_builder(physics_cfg=cfg, gravity=(0.0, 0.0, -9.81))
    builder.begin_world()
    mapping = add_content(builder, desc, args.loop_mode)
    builder.end_world()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, ke=3e5))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = builder.finalize(device="cuda:0")
        solver = SolverMuJoCo(model, integrator="implicitfast", iterations=100, ls_iterations=20,
                              tolerance=1e-8, njmax=args.njmax, nconmax=4096)
    messages = [str(w.message) for w in caught]
    skipped = [message for message in messages if "Skipping loop closure" in message]
    rest = None
    if args.rest and not skipped:
        state, next_state, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        initial = state.body_q.numpy()[mapping["bodies"], :3].copy()
        finite = True
        for step in range(750):
            state.clear_forces()
            solver.step(state, next_state, control, None, 0.0004)
            state, next_state = next_state, state
            finite = finite and np.isfinite(state.body_q.numpy()).all() and np.isfinite(state.body_qd.numpy()).all()
            if not finite:
                break
        positions = state.body_q.numpy()[mapping["bodies"], :3]
        rest = {"duration_s": 0.3, "dt_s": 0.0004, "finite": bool(finite),
                "completed_steps": step + 1,
                "first_nonfinite_t_s": None if finite else (step + 1) * 0.0004,
                "max_segment_displacement_mm": float(np.max(np.linalg.norm(positions - initial, axis=1))) * 1000 if finite else None,
                "max_segment_speed_m_s": float(np.max(np.linalg.norm(state.body_qd.numpy()[mapping["bodies"], :3], axis=1))) if finite else None}
    result = {"category": desc["kappa"]["category"], **mapping, "warnings": messages,
              "skipped_loop_mappings": len(skipped), "native_full_graph_represented": not skipped,
              "dynamics_run": rest is not None, "rest": rest,
              "njmax": args.njmax,
              "actual_total_mass_kg": float(model.body_mass.numpy().sum()),
              "actual_proxy_mass_kg": float(model.body_mass.numpy()[len(mapping["bodies"]):].sum()),
              "actual_proxy_inertias": model.body_inertia.numpy()[len(mapping["bodies"]):].tolist(),
              "unmatched_cpu_mechanisms": ["compression/tension asymmetry", "coupled fiber limits",
                                            "confinement-gated compaction welds"],
              "note": "Construction probe only; no R1-R3 consistency or migration claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("category", "loop_mode", "proxy_bodies", "skipped_loop_mappings",
                                            "dynamics_run", "rest", "actual_total_mass_kg")}), flush=True)
    if skipped:
        raise SystemExit(2)
    if rest is not None and not rest["finite"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

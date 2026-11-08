"""L4 isolated physical responses in a batch of otherwise identical worlds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import newton
from newton.solvers import SolverMuJoCo
import numpy as np
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager


@wp.kernel
def set_forces(forces: wp.array[wp.spatial_vector], values: wp.array[wp.spatial_vector]):
    i = wp.tid()
    forces[i] = values[i]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_envs < 2 or args.output.exists():
        parser.error("need at least two worlds and a new output path")
    wp.config.enable_backward = False
    rng = np.random.default_rng(args.seed)
    n = args.num_envs
    parameters = {
        "mass_kg": rng.uniform(0.05, 0.4, n),
        "stiffness_N_m": rng.uniform(20, 80, n),
        "damping_N_s_m": rng.uniform(0.2, 4.0, n),
        "upper_limit_m": rng.uniform(0.005, 0.04, n),
        "friction": rng.uniform(0.05, 0.9, n),
    }
    cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(integrator="implicitfast"))
    builder = NewtonMJWarpManager.create_builder(physics_cfg=cfg, gravity=(0, 0, -9.81))
    bodies, shape_ids, joint_ids, q_starts, qd_starts = [], [], [], [], []
    kinds = ("mass", "stiffness", "damping", "limit", "friction")
    for world in range(n):
        builder.begin_world()
        world_bodies, world_shapes, world_joints, world_q, world_qd = [], [], [], [], []
        for index, kind in enumerate(kinds):
            origin = wp.vec3(index * 0.25, 0, 0.0201 if kind == "friction" else 0.5)
            mass = float(parameters["mass_kg"][world]) if kind == "mass" else 0.2
            body = builder.add_link(xform=wp.transform(origin, wp.quat_identity()), mass=mass,
                                    inertia=wp.mat33(*np.diag([mass * 0.04**2 / 6] * 3).ravel()),
                                    lock_inertia=True, label=kind)
            shape = builder.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02,
                        cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=float(parameters["friction"][world])
                                                            if kind == "friction" else 0.5))
            if kind == "friction":
                joint = builder.add_joint_free(body, label=kind + "_joint")
            else:
                stiffness = float(parameters["stiffness_N_m"][world]) if kind == "stiffness" else (40 if kind == "damping" else 0)
                damping = float(parameters["damping_N_s_m"][world]) if kind == "damping" else (2 if kind == "stiffness" else 0)
                limit = float(parameters["upper_limit_m"][world]) if kind == "limit" else 1e6
                joint = builder.add_joint_prismatic(-1, body, axis=wp.vec3(1, 0, 0),
                            parent_xform=wp.transform(origin, wp.quat_identity()),
                            target_ke=stiffness, damping=damping, limit_lower=-limit, limit_upper=limit,
                            custom_attributes={"mujoco:solreflimit": wp.vec2(0.001, 2.0),
                                               "mujoco:solreflimit_mode": 1},
                            label=kind + "_joint")
            builder.add_articulation([joint], label=kind + "_fixture")
            q = builder.joint_q_start[joint]
            qd = builder.joint_qd_start[joint]
            if kind == "damping":
                builder.joint_q[q] = 0.02
            if kind == "friction":
                builder.joint_qd[qd] = 1.0
            world_bodies.append(body)
            world_shapes.append(shape)
            world_joints.append(joint)
            world_q.append(q)
            world_qd.append(qd)
        builder.end_world()
        bodies.append(world_bodies)
        shape_ids.append(world_shapes)
        joint_ids.append(world_joints)
        q_starts.append(world_q)
        qd_starts.append(world_qd)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
    model = builder.finalize(device="cuda:0")
    solver = SolverMuJoCo(model, integrator="implicitfast", iterations=100, ls_iterations=20, tolerance=1e-8)
    state, next_state, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    bodies, shape_ids, q_starts, qd_starts = map(np.array, (bodies, shape_ids, q_starts, qd_starts))
    readback = {
        "mass_kg": model.body_mass.numpy()[bodies[:, 0]],
        "stiffness_N_m": model.joint_target_ke.numpy()[qd_starts[:, 1]],
        "damping_N_s_m": model.joint_damping.numpy()[qd_starts[:, 2]],
        "upper_limit_m": model.joint_limit_upper.numpy()[qd_starts[:, 3]],
        "friction": model.shape_material_mu.numpy()[shape_ids[:, 4]],
    }
    force_values = np.zeros((model.body_count, 6), np.float32)
    force_values[bodies[:, 0], 0] = 0.1
    force_values[bodies[:, 1], 0] = 1
    force_values[bodies[:, 3], 0] = 10
    initial_forces = wp.array(force_values, dtype=wp.spatial_vector, device=model.device)
    force_values[bodies[:, 0], 0] = 0
    later_forces = wp.array(force_values, dtype=wp.spatial_vector, device=model.device)
    initial_x = state.body_q.numpy()[bodies, 0]
    mass_x, damping_energy = None, None
    dt = 0.0004
    for step in range(2500):
        wp.launch(set_forces, dim=model.body_count,
                  inputs=[state.body_f, initial_forces if step < 250 else later_forces], device=model.device)
        solver.step(state, next_state, control, None, dt)
        state, next_state = next_state, state
        if step == 249:
            mass_x = state.body_q.numpy()[bodies[:, 0], 0] - initial_x[:, 0]
        if step == 499:
            x = state.body_q.numpy()[bodies[:, 2], 0] - 0.5
            v = state.body_qd.numpy()[bodies[:, 2], 0]
            damping_energy = 0.5 * 40 * x**2 + 0.5 * 0.2 * v**2
    positions = state.body_q.numpy()[bodies, 0] - initial_x
    result = {
        "num_envs": n, "seed": args.seed, "dt_s": dt, "duration_s": 1.0,
        "parameters": {k: v.tolist() for k, v in parameters.items()},
        "newton_readback": {k: v.tolist() for k, v in readback.items()},
        "readback_max_abs_error": {k: float(np.max(np.abs(parameters[k] - readback[k]))) for k in parameters},
        "responses": {"mass_displacement_at_0_1s_m": mass_x.tolist(),
                      "spring_displacement_at_1s_m": positions[:, 1].tolist(),
                      "damping_energy_at_0_2s_J": damping_energy.tolist(),
                      "limited_displacement_at_1s_m": positions[:, 3].tolist(),
                      "friction_sliding_distance_at_1s_m": positions[:, 4].tolist()},
        "finite": bool(np.isfinite(state.body_q.numpy()).all() and np.isfinite(state.body_qd.numpy()).all()),
        "engine_body_mass": solver.mjw_model.body_mass.numpy().tolist(),
        "engine_dof_damping": solver.mjw_model.dof_damping.numpy().tolist(),
        "engine_joint_range": solver.mjw_model.jnt_range.numpy().tolist(),
        "engine_geom_friction": solver.mjw_model.geom_friction.numpy().tolist(),
        "engine_actuator_gainprm": solver.mjw_model.actuator_gainprm.numpy().tolist(),
        "note": "Isolated fixtures verify parameter effects; not content R1-R3 or completed grasps",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("num_envs", "seed", "finite", "readback_max_abs_error")}))
    if not result["finite"]:
        raise RuntimeError("nonfinite fixture state")


if __name__ == "__main__":
    main()

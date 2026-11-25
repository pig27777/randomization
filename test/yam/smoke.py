"""Small single/batched acceptance runs through the public Yam_B GPU interface."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from importlib.metadata import version
import os
from pathlib import Path
import socket
import subprocess
from threading import Event, Thread
import time
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import warp as wp

from yam_platform import yam_warp
from yam_platform.abc_workcell import add_abc_workcell
from yam_platform.robot import asset_fingerprint, rotation_error
from yam_platform.yam_interface import RealYAMInterface
from yam_platform.yam_model import ARM_JOINTS, arm_indices, build_yam_model, initial_state, load_config, save_yam_xml
from yam_platform.yam_warp import SimYAMInterface

ROOT = Path(__file__).resolve().parents[2]


def values(array):
    return " ".join(map(str, array))


def fixture(case):
    scene = json.loads((ROOT / "configs/route_b_yam_scene.json").read_text())
    workcell = json.loads((ROOT / "configs/route_b_workcell.yaml").read_text())
    bin_spec = json.loads((ROOT / "configs/route_b_single_arm_bin.yaml").read_text())["bin"]

    def build(root):
        world = root.find("worldbody")

        def box(parent, name, pos, half, color, **attrs):
            return ET.SubElement(parent, "geom", name=name, type="box", pos=values(pos),
                                 size=values(half), rgba=color, **attrs)

        parked = workcell["arms"]["arm_b"]
        parked_state = (parked["abc_index"], parked["ready_joints_rad"]) if case == "abc" else None
        add_abc_workcell(root, scene, parked_arm=parked_state)
        if case in ("soft", "rigid"):
            stage = ET.SubElement(world, "body", name="test_surface", pos="-.03 -.225 .049")
            if case == "soft":
                ET.SubElement(stage, "joint", name="soft_deflection", type="slide", axis="0 0 1",
                              stiffness="900", damping="8", range="-.035 .005")
            box(stage, "press_surface", [0, 0, 0], [.060, .043, .018],
                ".38 .77 .67 1" if case == "soft" else ".41 .60 .80 1", mass=".035",
                solref=".006 1")
        elif case in ("grasp", "abc"):
            payload = ET.SubElement(world, "body", name="payload", pos="-.03 -.225 .0325")
            ET.SubElement(payload, "freejoint", name="payload_free")
            box(payload, "payload_geom", [0, 0, 0], [.026, .024, .0325], ".80 .44 .19 1",
                mass=".12", friction="1 .005 .0005")
            center = np.array(bin_spec["center_m"])
            hx, hy = bin_spec["inner_half_xy_m"]
            t = bin_spec["wall_thickness_m"]
            z = bin_spec["rim_height_m"] / 2
            box(world, "bin_floor", center + [0, 0, .002], [hx, hy, .002], ".16 .35 .46 1")
            for k, (x, y) in enumerate(((hx, 0), (-hx, 0), (0, hy), (0, -hy))):
                half = [t / 2, hy + t, z] if x else [hx, t / 2, z]
                box(world, f"bin_wall_{k}", center + [x, y, z], half, ".20 .44 .56 1")
    return build


class Trial:
    def __init__(self, case, count=1, *, sim=None, **kwargs):
        self.sim = sim if sim is not None else SimYAMInterface(count, add_scene=fixture(case), **kwargs)
        self.count = count
        self.model = m = self.sim.model
        self.q = initial_state(m, self.sim.config)
        self.site = m.site("grasp_site").id
        self.joints, self.pos, self.dof = arm_indices(m)
        self.start = self.q.site_xpos[self.site].copy()
        self.rot = self.q.site_xmat[self.site].reshape(3, 3).copy()
        self.ik = self.q.qpos[self.pos[:6]].copy()
        self.ref = np.r_[self.ik, self.sim.config["ready_gap_m"]]
        self.trace = []
        self.phase = ""

    def ik_path(self, pos):
        m, d = self.model, self.q
        jacp, jacr = np.empty((3, m.nv)), np.empty((3, m.nv))
        low, high = m.jnt_range[self.joints[:6]].T
        for _ in range(25):
            d.qpos[self.pos[:6]] = self.ik
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            dp = pos - d.site_xpos[self.site]
            rot_error = rotation_error(d.site_xmat[self.site].reshape(3, 3), self.rot)
            error = np.r_[4 * dp, rot_error]
            if np.linalg.norm(dp) < 0.00008 and np.linalg.norm(rot_error) < .001:
                return self.ik.copy()
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            j = np.vstack((4 * jacp[:, self.dof[:6]], jacr[:, self.dof[:6]]))
            delta = j.T @ np.linalg.solve(j @ j.T + .0001 * np.eye(6), error)
            self.ik = np.clip(self.ik + np.clip(delta, -.15, .15), low, high)
        raise RuntimeError(f"Unreachable fixture waypoint {pos}: residual {error}")

    def log(self):
        r, m = self.sim, self.model
        d = r.snapshot()
        obs = r.obs.numpy()[0]
        diag = r.get_diagnostics()
        act = [m.actuator(name).id for name in (*ARM_JOINTS, "gripper")]
        applied = d.actuator_force[act]
        forces = {}
        wrench = np.zeros(6)
        for i, contact in enumerate(d.contact):
            if contact.efc_address >= 0:
                mujoco.mj_contactForce(m, d, i, wrench)
                a, b = sorted(m.geom(g).name for g in contact.geom)
                forces[(a, b)] = forces.get((a, b), 0.) + wrench[0]
        def with_geom(name):
            return sum(force for pair, force in forces.items() if name in pair)
        def grip_pad(name):
            return forces.get(tuple(sorted((name, "payload_geom"))), 0.)
        payload = d.body("payload").xpos.copy() if m.nbody > 1 and mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "payload") >= 0 else np.zeros(3)
        self.trace.append(dict(
            time=d.time, phase=self.phase, observation=obs, qpos=d.qpos.copy(),
            tau_command=diag["tau_command"].numpy()[0], tau_actuator=applied[:6].copy(),
            tau_feedback=obs[12:18], tau_external=r.contact_torque(snapshot=d),
            gripper_force_N=applied[6], press_force_N=with_geom("press_surface"),
            pads_N=[grip_pad("left_pad"), grip_pad("right_pad")], payload_xyz=payload,
            tcp=d.site_xpos[self.site].copy(), rotation=d.site_xmat[self.site].reshape(3, 3).copy(),
            overflow=int(r.data.overflow.numpy()[0]),
        ))

    def step(self, target=None, action=None):
        if target is not None:
            self.ref = target
            self.sim.send_joint_command(target)
        if action is not None:
            self.sim.send_cartesian_action(action)
        self.sim.step()
        self.log()

    def hold(self, seconds):
        for _ in range(round(seconds / self.sim.policy_period)):
            self.step()

    def goto(self, target, gap, seconds, phase):
        self.phase = phase
        before = self.trace[-1]["tcp"] if not hasattr(self, "reference_tcp") and self.trace else self.start
        before = getattr(self, "reference_tcp", before)
        g0 = self.ref[-1]
        for a in np.linspace(0, 1, 1 + round(seconds / self.sim.policy_period))[1:]:
            a = a * a * (3 - 2 * a)
            self.step(np.r_[self.ik_path(before + a * (target - before)), g0 + a * (gap - g0)])
        self.reference_tcp = target.copy()

    def array(self, name, phase=None):
        return np.asarray([item[name] for item in self.trace if phase is None or item["phase"] == phase])

    def save(self, output, case):
        np.savez_compressed(output / f"{case}.npz", **{k: self.array(k) for k in self.trace[0]})
        save_yam_xml(self.sim.xml, output / f"{case}.xml")


def tracking(output):
    trial = Trial("tracking")
    init = trial.ref.copy()
    trial.phase = "settle"
    trial.hold(.6)
    directions = []
    for j in range(6):
        trial.phase = f"axis{j + 1}"
        start = trial.trace[-1]["observation"][j]
        target = init.copy()
        target[j] += .15
        trial.step(target)
        trial.hold(.76)
        end = trial.trace[-1]["observation"][j]
        directions.append(float(end - start))
        trial.step(init)
        trial.hold(.60)
    trial.phase = "cartesian"
    start = trial.trace[-1]
    for _ in range(24):
        trial.step(action=np.array([.65, 0., 0., 0., 0., .35, .68]))
    finish = trial.trace[-1]
    trial.phase = "final"
    trial.hold(.52)
    all_obs = trial.array("observation")
    final = trial.array("observation", "final")[-10:]
    effort = trial.array("tau_actuator")
    limits = np.asarray(trial.sim.config["torque_limit_Nm"])
    cap_fraction = float(np.mean(np.any(abs(effort) >= limits - 1e-5, axis=1)))
    error = float(np.max(abs(final[:, :6] - final[:, 18:24])))
    tail_delta = float(np.max(np.ptp(final[:, :6], axis=0)))
    low, high = trial.model.jnt_range[trial.joints[:6]].T
    limit_error = float(max(0, np.max(low - all_obs[:, :6]), np.max(all_obs[:, :6] - high)))
    displacement = float(finish["tcp"][0] - start["tcp"][0])
    rotation = float(rotation_error(start["rotation"], finish["rotation"])[2])
    assert np.array_equal(trial.array("tau_feedback"), effort)
    assert np.all(trial.array("overflow") == 0)
    trial.save(output, "tracking")
    return dict(pass_gate=min(directions) > 0 and error < .05 and limit_error <= 1e-4
                and cap_fraction < .05 and tail_delta < .015 and displacement > 0.003 and rotation > .015,
                positive_axis_motion_rad=directions, final_error_rad=error, final_excursion_rad=tail_delta,
                joint_limit_violation_rad=limit_error, saturated_fraction=cap_fraction,
                cartesian_x_m=displacement, cartesian_rz_rad=rotation)


def contact(output):
    traces = {}
    for case in ("free", "soft", "rigid"):
        trial = Trial(case)
        trial.phase = "settle"
        trial.hold(.40)
        trial.goto(trial.start + [0, 0, -.062], trial.ref[-1], 1.36, "press")
        trial.hold(.28)
        trial.goto(trial.start, trial.ref[-1], .80, "retract")
        assert np.array_equal(trial.array("tau_feedback"), trial.array("tau_actuator"))
        assert np.all(trial.array("overflow") == 0)
        trial.save(output, case)
        use = "press"
        obs = trial.array("observation", use)
        traces[case] = (trial.array("press_force_N", use), obs[:, :6] - obs[:, 18:24],
                        obs[:, 6:12], trial.array("tau_feedback", use))
    from itertools import combinations
    pairwise = {}
    for a, b in combinations(traces, 2):
        pairwise[f"{a}_{b}"] = [float(np.max(np.linalg.norm(x - y, axis=1))) for x, y in
                                  zip(traces[a][1:], traces[b][1:])]
    peaks = {case: float(trace[0].max()) for case, trace in traces.items()}
    return dict(pass_gate=peaks["free"] == 0 and min(peaks["soft"], peaks["rigid"]) > 1
                and all(min(row) > 1e-3 for row in pairwise.values()),
                peak_native_force_N=peaks, peak_pairwise_q_dq_eff_delta=pairwise)


def check_abc(trial):
    m, d = trial.model, trial.q
    cfg = json.loads((ROOT / "configs/route_b_yam_scene.json").read_text())
    base_a = d.body("Yam_A").xpos
    base_b = d.body("arm").xpos
    assert np.allclose(np.stack((base_a, base_b)), cfg["arm_bases_m"])
    assert m.body("Yam_A").jntnum[0] == 0 and m.nu == 7
    assert np.allclose(d.camera("abc_top").xpos, cfg["top_camera"]["position_m"])
    assert np.allclose(d.camera("abc_wrist_b").xpos, d.body("wrist_camera").xpos)
    assert np.isclose(m.geom("table").pos[2] + m.geom("table").size[2], 0.)


def grasp(output, *, abc=False):
    empty = Trial("empty")
    empty.goto(empty.start, 0., .88, "close")
    empty.hold(.48)
    empty.save(output, "empty")
    empty_gap = float(empty.array("observation")[-1, 24])

    trial = Trial("abc" if abc else "grasp", capacities={"njmax": 256, "nconmax": 64})
    if abc:
        check_abc(trial)
    pick = np.array([-.03, -.225, .052])
    lift = pick + [0, 0, .132]
    corridor = np.array([-.11, pick[1], lift[2]])
    target = np.array([-.11, -.045, lift[2]])
    placed = target.copy()
    placed[2] = .061
    trial.goto(pick, .08, 1.0, "approach")
    trial.goto(pick, 0., .84, "close")
    trial.hold(.24)
    trial.goto(lift, 0., 1.36, "lift")
    trial.goto(corridor, 0., 1.24, "transport")
    trial.goto(target, 0., 2.20, "transport")
    trial.goto(placed, 0., 1.28, "place")
    trial.goto(placed, .08, .88, "release")
    trial.hold(.56)
    trial.goto(target, .08, 1.28, "retreat")
    trial.save(output, "grasp")
    closed = trial.array("observation", "close")[-1]
    lifted = trial.array("payload_xyz", "lift").max(axis=0)
    moved = trial.array("payload_xyz", "transport")[-1]
    final = trial.array("payload_xyz")[-1]
    pads = trial.array("pads_N", "lift")
    contact_frames = int(np.count_nonzero(np.min(pads, axis=1) > 1.0))
    peak = float(np.max(abs(trial.array("gripper_force_N"))))
    released = float(np.max(abs(trial.array("pads_N")[-7:]))) < .05
    still = float(np.max(np.ptp(trial.array("payload_xyz")[-12:], axis=0))) < .002
    saturated = peak >= 49.99
    c_pass = empty_gap <= .002 and .042 < closed[24] < .060 and abs(closed[26]) > .2 and saturated and released
    d_pass = (contact_frames >= 5 and lifted[2] - .0325 >= .04 and moved[1] + .225 >= .08
              and abs(final[0] + .03) < .21 and abs(final[1] - .03) < .151 and .035 < final[2] < .039
              and still and released)
    assert peak <= 50.00001 and np.all(trial.array("overflow") == 0)
    return dict(gripper_pass=bool(c_pass), pick_place_pass=bool(d_pass), empty_gap_m=empty_gap,
                blocked_gap_m=float(closed[24]), blocked_effort_motor_Nm=float(closed[26]), peak_gap_force_N=peak,
                saturated=saturated, lifting_dual_contact_frames=contact_frames, peak_payload_z_m=float(lifted[2]),
                transported_payload_xyz_m=moved.tolist(), final_payload_xyz_m=final.tolist(),
                released=released, still=still, abc_workcell=abc)


@contextmanager
def memory_samples():
    done, started, samples = Event(), Event(), []
    def poll():
        while not done.is_set():
            result = subprocess.run([
                "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits",
            ], text=True, capture_output=True, check=True)
            memory = [int(mem) for line in result.stdout.splitlines()
                      for pid, mem in [line.split(", ")] if pid == str(os.getpid()) and mem.isdecimal()]
            if memory:
                samples.append(sum(memory))
            started.set()
            done.wait(.20)
    worker = Thread(target=poll, daemon=True)
    worker.start()
    try:
        yield samples
    finally:
        started.wait(1)
        done.set()
        worker.join()


def gpu(worlds, randomize, steps):
    with memory_samples() as memory:
        trial = Trial("batch", worlds, randomization=randomize, try_capacity=32)
        r = trial.sim
        r.history.start_try(r.all_worlds)
        act = np.tile([.35, 0, 0, 0, .20, 0, .7], (worlds, 1))
        act[::2, 0] *= -1
        actions = wp.array(act, dtype=float, device=r.device)
        for _ in range(5):
            r.send_cartesian_action(actions)
            r.step()
        wp.synchronize_device(r.device)
        begin = time.perf_counter()
        for _ in range(steps):
            r.send_cartesian_action(actions)
            r.step()
        wp.synchronize_device(r.device)
        elapsed = time.perf_counter() - begin

        recorded = r.history.read()
        assert np.all(recorded["valid_mask"].numpy())
        assert np.all(np.diff(recorded["measurement_timestamp"].numpy(), axis=1) > 0)
        assert np.array_equal(recorded["action_history"].numpy()[:, -1], act.astype(np.float32))
        assert np.allclose(recorded["proprio_history"].numpy()[:, -1], r.obs.numpy())
        assert np.all(r.history.read(full_try=True)["total_frames"].numpy() == steps + 5)
        assert np.allclose(r.data.time.numpy() - r.measurement_timestamp.numpy(), r.delay.numpy() * r.sensor_period,
                           atol=8e-6)
        slot = (r.clock.numpy() - 1 - r.delay.numpy()) % r.sensor.shape[1]
        queued = r.sensor.numpy()[np.arange(worlds), slot]
        assert np.array_equal(recorded["proprio_history"].numpy()[:, -1, :6], queued[:, 0, :6])
        assert np.array_equal(recorded["proprio_history"].numpy()[:, -1, 12:18], queued[:, 2, :6])
        assert np.all(np.isfinite(r.obs.numpy()))
        assert np.all(np.isfinite(r.data.qpos.numpy())) and np.all(np.isfinite(r.data.qvel.numpy()))
        assert np.all(r.data.overflow.numpy() == 0)
        assert not any(name in r.get_observation() for name in ("joint_eff_true", "tau_external", "qfrc_constraint"))
        assert "joint_eff" not in r.get_observation(include_effort=False)
        assert all(field.device == r.device for field in recorded.values())

        mask = np.arange(worlds) % 2 == 0
        selection = wp.array(mask, dtype=bool, device=r.device)
        m, d = r.gpu_model, r.data
        robot_body = trial.model.body("link_2").id
        before_mass = m.body_mass.numpy().copy()
        end_tcp = d.site_xpos.numpy()[:, trial.site]
        assert np.max(abs(end_tcp - trial.start)) > .005
        if worlds >= 2:
            assert end_tcp[1, 0] - end_tcp[0, 0] > .01
        if worlds >= 4:
            if randomize:
                assert np.std(end_tcp[0::2, 0]) > 1e-5
            else:
                assert np.allclose(end_tcp[0::2], end_tcp[0], atol=5e-6)
        prior_q, prior_v, prior_obs = d.qpos.numpy(), d.qvel.numpy(), r.obs.numpy()
        r.reset(selection)
        assert np.allclose(d.qpos.numpy()[mask], r.initial.numpy())
        assert np.array_equal(d.qpos.numpy()[~mask], prior_q[~mask])
        assert np.array_equal(d.qvel.numpy()[~mask], prior_v[~mask])
        assert np.array_equal(r.obs.numpy()[~mask], prior_obs[~mask])
        assert not np.any(r.history.read()["valid_mask"].numpy()[mask])
        assert np.all(r.history.read()["valid_mask"].numpy()[~mask])
        assert not np.any(r.history.read(full_try=True)["valid_mask"].numpy()[mask])
        params = {}
        if randomize:
            after_mass = m.body_mass.numpy()
            assert np.array_equal(after_mass[~mask], before_mass[~mask])
            assert np.any(after_mass[mask, robot_body] != before_mass[mask, robot_body])
            nominal_mass = trial.model.body_mass[robot_body]
            scale = after_mass[:, robot_body] / nominal_mass
            assert np.allclose(m.body_inertia.numpy()[:, robot_body],
                               scale[:, None] * trial.model.body_inertia[robot_body])
            assert np.allclose(m.body_subtreemass.numpy()[:, 0], after_mass.sum(axis=1))
            assert np.any(m.dof_frictionloss.numpy() != trial.model.dof_frictionloss)
            assert np.any(m.dof_armature.numpy() != trial.model.dof_armature)
            assert np.any(m.geom_friction.numpy() != trial.model.geom_friction)
            assert np.all(np.abs(r.motors.applied.numpy()[:, :6]) <= np.asarray(r.config["torque_limit_Nm"]))
            q = d.qpos.numpy()[:, trial.pos[:6]]
            assert np.max(abs(q[mask] - prior_q[mask][:, trial.pos[:6]])) > 1e-4
            params = dict(link_2_mass_range_kg=[float(scale.min() * nominal_mass), float(scale.max() * nominal_mass)],
                          observation_delay_ticks=r.delay.numpy().tolist(),
                          pad_sliding_friction_range=[float(m.geom_friction.numpy()[:, r.pad_ids.numpy(), 0].min()),
                                                      float(m.geom_friction.numpy()[:, r.pad_ids.numpy(), 0].max())])
        time.sleep(.24)
    return dict(pass_gate=True, num_envs=worlds, randomization=randomize, elapsed_s=elapsed,
                environment_steps_s=steps * worlds / elapsed, physics_steps_s=steps * worlds * r.substeps / elapsed,
                peak_process_gpu_MiB=max(memory, default=None), gpu_memory_samples=len(memory),
                world0_tcp_end_m=end_tcp[0].tolist(),
                dt_s=r.dt, policy_decimation=r.substeps, numerical_failures=0, **params)


def check_asset():
    cfg = load_config("nominal")
    abc = json.loads((ROOT / "configs/route_b_yam_scene.json").read_text())
    xml, model = build_yam_model()
    assert cfg["base_m"] == abc["arm_bases_m"][1] and cfg["ready_joints_rad"] == abc["ready_arm_joints_rad"][1]
    assert np.allclose(2 * model.geom("left_pad").size, abc["pad_box_size_m"])
    d = initial_state(model, cfg)
    joints, pos, dof = arm_indices(model)
    site = model.site("grasp_site").id
    jacp, jacr = np.empty((3, model.nv)), np.empty((3, model.nv))
    mujoco.mj_jacSite(model, d, jacp, jacr, site)
    ref = d.site_xpos[site].copy()
    for j in range(6):
        d.qpos[pos[j]] += 1e-6
        mujoco.mj_kinematics(model, d)
        assert np.allclose((d.site_xpos[site] - ref) / 1e-6, jacp[:, dof[j]], atol=1e-6)
        d.qpos[pos[j]] -= 1e-6
    sdk = RealYAMInterface()
    command = np.array([*cfg["ready_joints_rad"], .095])
    assert sdk.sdk_joint_position(command)[-1] == 1.0
    mapped = sdk.pack_observation(dict(joint_pos=command[:6], joint_vel=np.zeros(6), joint_eff=np.arange(6),
                                       gripper_pos=[1.0], gripper_vel=[-.2], gripper_eff=[-.5]),
                                  sent_position=command)
    assert mapped.shape == (1, 28)
    assert np.allclose(mapped[0, 24:], [.095, -.019, -.5, .095])
    return dict(fk_tcp_at_ready_m=ref.tolist(), pad_size_m=(2 * model.geom("left_pad").size).tolist(),
                q_limits_rad=model.jnt_range[joints[:6]].tolist(), axis_order=list(ARM_JOINTS),
                link_masses_kg={model.body(i).name: mass for i, mass in enumerate(model.body_mass) if i})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("tracking", "contact", "grasp", "abc", "gpu", "asset"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(yam_warp.__file__).with_name("abc_workcell.py"),
               *sorted(Path(yam_warp.__file__).parent.glob("yam_*.py")),
               *sorted(Path(yam_warp.__file__).parent.glob("yam_*.json"))]
    source_dir = args.output / "source"
    source_dir.mkdir()
    for source in sources:
        (source_dir / source.name).write_bytes(source.read_bytes())
    meta = dict(host=socket.gethostname(), engine=version("mujoco-warp"),
                mujoco=version("mujoco"), warp=version("warp-lang"), args=vars(args) | {"output": str(args.output)},
                assets=asset_fingerprint(), sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
    if args.case == "gpu":
        result = gpu(args.worlds, args.randomize, args.steps)
    elif args.case == "asset":
        result = check_asset()
    elif args.case == "abc":
        result = grasp(args.output, abc=True)
    else:
        result = {"tracking": tracking, "contact": contact, "grasp": grasp}[args.case](args.output)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(not all(value for name, value in result.items() if name == "pass_gate" or name.endswith("_pass")))


if __name__ == "__main__":
    raise SystemExit(main())

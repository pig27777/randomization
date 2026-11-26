from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from policy_v3.data import EpisodeRecord, GroupSplit, OnPolicyBuffer, ReplayBuffer, assert_disjoint
from policy_v3.environment import MacroBagGraspEnv, MotionStop
from policy_v3.evaluation import MODALITY_PROFILES, causal_chain, fixed_try, interaction, matched_modalities
from policy_v3.geometry import InfoMode, PublicContext, VisualObservation, exp_rotation, rotation_6d, rotation_matrix
from policy_v3.history import EFFORT, HistoryPreprocess, encoder_frames, pad_histories
from policy_v3.models import ProprioHistoryEncoder, tanh_log_prob
from policy_v3.training import ContextualPPOEntry, InformationProfile, SupervisedTrainEntry
from policy_v3.workers import CentralLearner, CollectionConfig, SingleMultiGPUWorker, worker_environment

ROOT = Path(__file__).resolve().parents[2]
V0 = VisualObservation([-.03, -.225, .06], np.eye(3), [.03, .03, .04], "fixture_geometry")
V1 = VisualObservation([.01, -.18, .08], exp_rotation([0, 0, .43]), [.04, .035, .02], "fixture_geometry")
G = np.r_[V0.center, rotation_6d(np.eye(3)), .08, .5]


def record(name, initial=None, probe=None, *, invalid=None, unsafe=False, success=False, version=0, action=None):
    return EpisodeRecord(name, initial or name, probe or name, V0, V1,
                         np.zeros(13) if action is None else action, np.zeros((2, 54)), 2,
                         None if invalid == "probe" else G, "v1_allowed", success, invalid is None, invalid,
                         "failure" if invalid else None, {}, version, [.04, .08], np.zeros((2, 3)),
                         ["start", "hover"], {"unsafe": unsafe},
                         {"probe_ticks": 2, "probe_motion_m": .1, "unsafe": unsafe})


class MinimalBackend:
    hover_rotation = np.eye(3)
    joint_pos = np.zeros(6)
    grasp_width_range = (.06, .095)
    grasp_effort_range = (.20, .70)
    common_sampling_region = (np.zeros(3), np.ones(3) * .01, np.eye(3))

    def __init__(self):
        self.observations = 0
        self.plans = []
        self.executed = []

    def reset(self, *, seed=None):
        return V0, np.zeros(14), dict(initial_state=np.array([seed or 0.]))

    def state_identity(self):
        return np.zeros(8)

    def observe(self):
        self.observations += 1
        return V1

    def grasp_plan(self, g):
        self.plans.append(g.copy())
        return True

    def execute_grasp(self, g):
        self.executed.append(g.copy())
        return dict(task_success=True)

    def close(self):
        pass


class PolicyV3Test(unittest.TestCase):
    def test_frozen_paths(self):
        frozen = ("src/low_order_parcel/lumped_contact_proxy.py", "src/low_order_parcel/unified_payload.py")
        for baseline, paths in (
            ("fix4.1-route-b-base", frozen),
            ("2ef4a394", frozen),
            ("736db9b2", ("src/isaac_yam_parcel/yam_load_port.py",)),
            ("2229e650", ("configs", "src/isaac_yam_parcel/yam_parcel_base_env.py",
                          "src/isaac_yam_parcel/yam_bin_env.py", "src/isaac_yam_parcel/joint_bc_ppo.py",
                          "experience/yam_platform")),
        ):
            subprocess.run(["git", "diff", "--exit-code", baseline, "--", *paths], cwd=ROOT, check=True)

    def test_info_mode_never_reads_v1(self):
        class Forbidden:
            def __getattribute__(self, name):
                raise AssertionError(f"v1 was accessed through {name}")

        for mode in (InfoMode.V0_ONLY, InfoMode.WORLD_COMMON):
            with self.subTest(mode=mode):
                backend = MinimalBackend()
                backend.observe = MagicMock(side_effect=AssertionError("post-Try capture was called"))
                env = MacroBagGraspEnv(backend, np.random.default_rng(19), info_mode=mode)
                env.reset()
                with patch.object(env.probe, "execute", return_value=(env.trace, {
                        "retracted": True, "physics_validity": True, "unsafe": False, "stop_reason": None})):
                    self.assertIsNone(env.execute_probe(np.zeros(13))[2])
                env.v1 = Forbidden()
                candidates = env.generate_grasp_candidates(K=3)
                self.assertEqual(len(backend.plans), 3)
                self.assertEqual(candidates.shape, (3, 11))
                features = env.public_context().features()
                np.testing.assert_array_equal(features[12:24], 0.)
                local = env.public_context().grasp_features(candidates)
                expected = candidates[:, :3] if mode is InfoMode.WORLD_COMMON else candidates[:, :3] - V0.center
                np.testing.assert_allclose(local[:, :3], expected, atol=1e-7)
                backend.observe.assert_not_called()
                with self.assertRaisesRegex(ValueError, "elevate"):
                    env.public_context(InfoMode.V1_ALLOWED)
                row = record("observe")
                row.v1 = Forbidden()
                np.testing.assert_array_equal(row.public_context(mode).features(), features)
        reader = MagicMock(return_value=V1)
        context = PublicContext("v1_allowed", V0, reader, np.zeros(13))
        np.testing.assert_allclose(context.grasp_features(G)[:3], (G[:3] - V1.center) @ V1.rotation, atol=1e-7)
        reader.assert_called_once()

    def test_v0_training_rejects_v1_selected_candidates(self):
        learner = SupervisedTrainEntry(profile=InformationProfile("v0_only"))
        row = record("v1_candidate")
        with self.assertRaisesRegex(ValueError, "shared-world"):
            learner.evaluate([row])
        row.probe_info["candidate_info_mode"] = "world_common"
        self.assertEqual(learner.evaluate([row])["examples"], 1)

    def test_group_split_joins_transitive_initial_and_probe_links(self):
        rows = [record("a", "i1", "p1"), record("b", "i1", "p2"), record("c", "i2", "p2")]
        rows += [record(str(i)) for i in range(12)]
        replay = ReplayBuffer(rows)
        splits = replay.splits(73)
        assert_disjoint(splits)
        owner = next(name for name, group in splits.items() if "a" in [r.episode_id for r in group])
        self.assertTrue({"a", "b", "c"} <= {r.episode_id for r in splits[owner]})
        replay.extend([record("repeated", "i2", "newprobe")] + [record(f"new-{i}") for i in range(4)])
        expanded = replay.splits(19)
        for name in splits:
            self.assertLessEqual({r.episode_id for r in splits[name]}, {r.episode_id for r in expanded[name]})
        self.assertIn("repeated", {r.episode_id for r in expanded[owner]})
        with self.assertRaises(AssertionError):
            assert_disjoint(dict(train=rows[:1], val=rows[1:2]))
        with self.assertRaises(AssertionError):
            assert_disjoint(dict(train=rows[1:2], test=rows[2:3]))
        self.assertEqual(sum(map(len, GroupSplit.partition(rows).values())), len(rows))

    def test_invalid_routing(self):
        good = record("valid", success=True)
        loss = record("valid_failure")
        probe_unsafe = record("bad_probe_try", invalid="probe", unsafe=True)
        probe_solver = record("bad_probe_solver", invalid="probe")
        grasp_solver = record("bad_grasp_solver", invalid="grasp")
        samples = [good, loss, probe_unsafe, probe_solver, grasp_solver]
        self.assertEqual([r.episode_id for r in ReplayBuffer.bce_records(samples)], ["valid", "valid_failure"])
        buffer = OnPolicyBuffer(0)
        for r in samples:
            buffer.append(r, V0.features(), np.zeros(14), np.zeros(13), np.zeros(13), -2.)
        self.assertEqual([x.episode_id for x in buffer.transitions], ["valid", "valid_failure", "bad_probe_try"])
        self.assertLess(buffer.transitions[-1].reward, -1.)
        self.assertAlmostEqual(buffer.physics_invalid_rate, .6)
        self.assertEqual(buffer.excluded_count, 2)
        with self.assertRaisesRegex(ValueError, "Mixed policy versions"):
            buffer.append(record("mixed", version=1), V0.features(), np.zeros(14), np.zeros(13), np.zeros(13), -2.)

    def test_tanh_log_prob_saturation_and_actual_ppo_update(self):
        torch.manual_seed(9)
        ppo = ContextualPPOEntry()
        features = torch.from_numpy(V0.features())[None]
        u = torch.tensor([[-100., 100.] * 6 + [-40.]])
        distribution = ppo.actor.distribution(features)
        action = u.tanh()
        self.assertTrue(torch.equal(action.abs(), torch.ones_like(action)))
        old = tanh_log_prob(distribution, u)
        expected = (distribution.log_prob(u) - (1. - action.square() + 1e-6).log()).sum(-1)
        torch.testing.assert_close(old, expected)
        buffer = OnPolicyBuffer(0)
        for i, x in enumerate((u, .005 * u, -.004 * u)):
            a = x.tanh().numpy()[0]
            log_p = float(tanh_log_prob(distribution, x).detach())
            r = record(str(i), action=a, success=i == 0)
            buffer.append(r, V0.features(), np.zeros(14), x.numpy()[0], a, log_p)
        before = [p.detach().clone() for p in (*ppo.actor.parameters(), *ppo.critic.parameters())]
        stats = ppo.update(buffer, epochs=1)
        self.assertTrue(np.isfinite([stats["actor_loss"], stats["critic_loss"]]).all())
        after = [*ppo.actor.parameters(), *ppo.critic.parameters()]
        actor_count = len(list(ppo.actor.parameters()))
        self.assertTrue(any(not torch.equal(before[i], after[i]) for i in range(actor_count)))
        self.assertTrue(any(not torch.equal(before[i], after[i]) for i in range(actor_count, len(after))))

    def test_padding_raw_and_optional_feedback(self):
        torch.manual_seed(11)
        model = ProprioHistoryEncoder()
        raw = np.random.default_rng(7).normal(size=(3, 54))
        raw[:, 18:24] = raw[:, 12:18] - raw[:, :6]
        h = encoder_frames(raw)
        self.assertEqual(h.shape, (3, 48))
        np.testing.assert_allclose(h[:, 12:18] - h[:, :6], raw[:, 18:24], atol=2e-7)
        training = HistoryPreprocess(effort_dropout=1.)
        dropped = training(h, np.random.default_rng(0))
        np.testing.assert_array_equal(dropped[:, EFFORT], 0.)
        h1, pad1 = pad_histories([h])
        h2 = torch.cat([h1, torch.full((1, 8, 48), float("nan"))], dim=1)
        pad2 = torch.cat([pad1, torch.ones((1, 8), dtype=torch.bool)], dim=1)
        a = torch.zeros((1, 13))
        torch.testing.assert_close(model(h1, pad1, a), model(h2, pad2, a))
        empty, mask = pad_histories([np.empty((0, 48), dtype=np.float32)])
        self.assertTrue(model(empty, mask, a).isfinite().all())
        rot = exp_rotation(np.array([.1, -.3, 1.2]))
        np.testing.assert_allclose(rotation_matrix(rotation_6d(rot)), rot, atol=1e-7)

    def test_one_try_and_stop_without_retract_or_visual(self):
        backend = MinimalBackend()
        backend.kinematics = MagicMock()
        backend.kinematics.pose.return_value = (np.zeros(3), np.eye(3))
        backend.kinematics.plan.return_value = [(np.zeros(6), .08, "probe_first")]
        backend.kinematics.connect.return_value = True
        backend.probe_displacement_m = .03
        backend.max_gap = backend.hover_gap = .08
        backend.hover_joints = np.zeros(6)
        backend.workspace_lower, backend.workspace_upper = np.full(3, -1.), np.ones(3)
        backend.probe_effort_Nm = .2
        backend.stop = MagicMock()
        backend.settle_hover = MagicMock()
        backend.follow = MagicMock(side_effect=MotionStop("contact_force_limit", unsafe=True))
        env = MacroBagGraspEnv(backend, np.random.default_rng(2))
        env.reset()
        _, length, v1, info = env.execute_probe(np.zeros(13))
        self.assertEqual(length, 0)
        self.assertEqual(info["stop_reason"], "contact_force_limit")
        self.assertIsNone(v1)
        self.assertEqual(backend.observations, 0)
        backend.stop.assert_called_once()
        backend.settle_hover.assert_not_called()
        with self.assertRaises(RuntimeError):
            env.execute_probe(np.zeros(13))
        with self.assertRaises(RuntimeError):
            env.execute_grasp(G)

        env.reset()
        backend.probe_cutoff_N = 2.
        backend.joint_pos = np.full(6, .1)
        backend.last_command = np.r_[backend.joint_pos, .06]
        contact_point = np.array([.01, -.01, .02])
        backend.kinematics.pose.side_effect = [(np.zeros(3), np.eye(3)), (np.zeros(3), np.eye(3)),
                                               (contact_point, np.eye(3))]
        backend.kinematics.plan.side_effect = [
            [(np.zeros(6), .08, "probe_approach"), (backend.joint_pos, .06, "probe_first")],
            [(backend.hover_joints, .08, "probe_retract")],
        ]
        backend.follow = MagicMock(side_effect=[None, 2.08, None])
        backend.at_hover = MagicMock(return_value=True)
        action = np.zeros(13)
        action[8], action[11] = -.5, -.5
        _, _, v1, info = env.execute_probe(action)
        self.assertEqual(backend.follow.call_args_list[1].kwargs["contact_cutoff_N"], 2.)
        q, gap, recovery = backend.kinematics.plan.call_args.args
        np.testing.assert_array_equal(q, backend.joint_pos)
        self.assertEqual(gap, .06)
        self.assertEqual(recovery[0][3], "probe_unload")
        np.testing.assert_allclose(recovery[0][0], contact_point + [0., 0., .003])
        self.assertEqual(recovery[1][3], "probe_second")
        self.assertGreaterEqual(recovery[1][0][2], recovery[0][0][2])
        self.assertEqual(info["probe_cutoff_trigger_N"], 2.08)
        self.assertTrue(info["retracted"])
        self.assertIs(v1, V1)

        from policy_v3.yam_backend import YamBackend

        physical = YamBackend.__new__(YamBackend)
        physical.scene = SimpleNamespace(is_flex=True)
        physical.drive = SimpleNamespace(tick=MagicMock(side_effect=MotionStop("contact_force_limit", unsafe=True)))
        physical.flex_core_counts = MagicMock(return_value=(0, 1))
        with self.assertRaises(MotionStop) as stop:
            physical.tick(np.zeros(7), .2, "probe")
        self.assertEqual(str(stop.exception), "contact_force_limit")
        self.assertFalse(stop.exception.physics_valid)

    def test_flex_and_rigid_share_the_table_without_duplicate_film_contacts(self):
        from yam_platform.yam_model import build_yam_model
        from policy_v3.yam_scene import FlexScene

        scene = FlexScene()
        _, model = build_yam_model(scene.config, add_scene=scene.add_scene, compile_model=scene.compile_model)
        table, plane = model.geom("table"), model.geom("table_flex_support")
        payload = model.geom(scene.payload_geom)
        film_type, film_affinity = model.flex_contype[0], model.flex_conaffinity[0]

        def permits(geom, contact_type, contact_affinity):
            return bool((geom.contype[0] & contact_affinity) or (geom.conaffinity[0] & contact_type))

        self.assertFalse(permits(table, film_type, film_affinity))
        self.assertTrue(permits(plane, film_type, film_affinity))
        self.assertTrue(permits(table, payload.contype[0], payload.conaffinity[0]))
        self.assertFalse(permits(plane, payload.contype[0], payload.conaffinity[0]))
        np.testing.assert_array_equal(plane.solref, scene.parcel.m.geom("table").solref)
        for name in ("left_pad", "right_pad"):
            pad = model.geom(name)
            self.assertTrue(permits(pad, film_type, film_affinity))
            self.assertFalse(permits(pad, payload.contype[0], payload.conaffinity[0]))
        import mujoco
        from policy_v3.yam_backend import YamBackend
        from policy_v3.yam_scene import CORNERS

        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        backend = YamBackend.__new__(YamBackend)
        backend.payload_id = model.body(scene.payload_body).id
        backend.drive = SimpleNamespace(data=data)
        center, rotation = backend.payload_box_pose()
        corners = (CORNERS * scene.half) @ rotation.T + center
        np.testing.assert_allclose(np.ptp(corners, axis=0), 2. * scene.half)
        self.assertFalse(np.allclose(rotation, data.geom_xmat[payload.id].reshape(3, 3)))

    def test_motor_effort_limit_uses_native_calibration(self):
        from policy_v3.yam_backend import YamBackend

        backend = YamBackend(7, scripted=True)
        backend.reset()
        limit = .31
        grasp = backend.scripted_grasp()
        grasp[-1] = limit
        result = backend.execute_grasp(grasp)
        self.assertTrue(result["task_success"])
        self.assertAlmostEqual(result["peak_gripper_motor_effort_Nm"], limit, places=6)
        self.assertAlmostEqual(backend.effort_cap_Nm, 50 * .096 / 6.57, places=8)

    def test_evaluation_freezes_g_a_in_world_and_matches_four_modalities(self):
        def make(seed, mode):
            e = MacroBagGraspEnv(MinimalBackend(), np.random.default_rng(seed), info_mode=mode)
            e.probe.execute = lambda _a, _v0: (e.trace, dict(retracted=True, physics_validity=True, unsafe=False,
                                                           stop_reason=None))
            return e

        action = np.zeros(13)
        a_selector = lambda c, h, gs: gs[0]
        d_selector = lambda c, h, gs: gs[1]
        results = causal_chain(make, 51, a_selector, a_selector, d_selector, fixed_try(action))
        np.testing.assert_array_equal(results["A"]["grasp_world"], results["B"]["grasp_world"])
        self.assertFalse(np.array_equal(results["A"]["grasp_world"], results["C"]["grasp_world"]))
        cells = matched_modalities(make, 51, action, dict.fromkeys(("J00", "J01", "J10", "J11"), a_selector))
        self.assertEqual(set(cells), {"J00", "J01", "J10", "J11"})
        self.assertFalse(MODALITY_PROFILES["without_effort"].effort)
        self.assertAlmostEqual(interaction({"J00": .1, "J10": .3, "J01": .2, "J11": .5}), .1)

    def test_c4_updates_then_publishes_and_gpu_environment_isolated(self):
        learner = CentralLearner()
        learner.replay.extend(record(str(i), success=i % 2 == 0) for i in range(12))
        events = []

        def collect(*args, **kwargs):
            self.assertEqual(learner.version, 0)
            events.append("statistics from all workers")
            args[0].mkdir()
            return {}, OnPolicyBuffer(0)

        with tempfile.TemporaryDirectory() as tmp, patch.object(learner, "_collect", side_effect=collect):
            with patch.object(learner.ppo, "update", side_effect=lambda *a, **kw: events.append("ppo") or {"examples": 0}):
                result = learner.train_round(Path(tmp) / "round", CollectionConfig(), 4, alternate=True)
            checkpoint = torch.load(Path(tmp) / "round/checkpoint.pt", weights_only=True)
        self.assertEqual(events, ["statistics from all workers", "ppo"])
        self.assertIn("bce", result)
        self.assertEqual(checkpoint["version"], 1)
        self.assertTrue(checkpoint["split_assignments"])
        first, second = worker_environment("0"), worker_environment("1")
        for i, env in enumerate((first, second)):
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], str(i))
            self.assertEqual(env["WARP_CACHE_DIR"], f"/tmp/warp_cache_gpu_{i}")
            self.assertEqual(env["WARP_CACHE_PATH"], env["WARP_CACHE_DIR"])
            self.assertEqual(env["OMP_NUM_THREADS"], "2")
            self.assertEqual(env["OPENBLAS_NUM_THREADS"], "2")
        with self.assertRaises(ValueError):
            SingleMultiGPUWorker(("0", "00"))


if __name__ == "__main__":
    unittest.main()

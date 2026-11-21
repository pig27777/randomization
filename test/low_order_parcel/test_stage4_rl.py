"""Stage 4 macro-step, asymmetric-gradient and C10 protocol gates."""

import os
import unittest
from unittest.mock import patch

import torch

from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.low_order_parcel.envs.handover_place_env import HandoverPlaceEnv
from src.low_order_parcel.envs.pick_to_air_env import PickToAirEnv
from src.low_order_parcel.rl.asymmetric_ppo import AsymmetricPPO, MacroAction, Observation


DEVICE = os.environ.get("STAGE4_TEST_DEVICE", "cpu")


class PPOTests(unittest.TestCase):
    def test_masked_log_prob_and_entropy_have_exact_zero_inactive_gradients(self):
        for env_type in (PickToAirEnv, HandoverPlaceEnv):
            with self.subTest(env=env_type.__name__):
                torch.manual_seed(31)
                model = AsymmetricPPO(env_type.static_size, env_type.seq_size, env_type.priv_size, env_type.decisions)
                obs = Observation(torch.randn(3, env_type.static_size), torch.randn(3, 28, env_type.seq_size),
                                  torch.arange(28)[None] < torch.tensor([10, 24, 2])[:, None],
                                  torch.randn(3, env_type.priv_size), torch.tensor([1, -1, 1]),
                                  torch.tensor([1, -1, 1]))
                latent = torch.randn(3, model.action_size)
                sample = model.evaluate_actions(obs, torch.zeros(3, dtype=torch.long), latent)
                self.assertEqual(sample.log_prob[1].item(), 0.)
                self.assertEqual(sample.entropy[1].item(), 0.)
                self.assertTrue((sample.action.continuous[1] == 0.).all())
                self.assertTrue((sample.action.continuous[:, 2:] == 0.).all())
                self.assertEqual(sample.action.choice.tolist(), [1, 0, 1])
                for loss in (sample.log_prob, sample.entropy):
                    model.zero_grad(set_to_none=True)
                    loss.sum().backward(retain_graph=True)
                    for group in (model.means, model.log_stds, model.categorical):
                        for key, head in group.items():
                            params = head.parameters() if isinstance(head, torch.nn.Module) else (head,)
                            for parameter in params:
                                self.assertIsNotNone(parameter.grad)
                                if key != "1_1":
                                    self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0, key)
                    self.assertGreater(model.log_stds["1_1"].grad.abs().sum().item(), 0.)
                model.zero_grad(set_to_none=True)
                loss = model.loss(obs, sample, torch.ones(3), torch.tensor([.4, -.3, .8]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))

    def test_unselected_continuous_head_with_an_active_categorical_branch(self):
        torch.manual_seed(23)
        model = AsymmetricPPO(15, 12, 86, PickToAirEnv.decisions)
        obs = Observation(torch.zeros(2, 15), torch.zeros(2, 28, 12), torch.ones(2, 28, dtype=torch.bool),
                          torch.zeros(2, 86), torch.zeros(2, dtype=torch.long),
                          torch.full((2,), -1, dtype=torch.long))
        sample = model.evaluate_actions(obs, torch.zeros(2, dtype=torch.long), torch.ones(2, 4))
        for objective in (sample.log_prob, sample.entropy):
            model.zero_grad(set_to_none=True)
            objective.sum().backward(retain_graph=True)
            for parameter in (*model.means["0_1"].parameters(), model.log_stds["0_1"]):
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)
            self.assertGreater(model.log_stds["0_0"].grad.abs().sum().item(), 0.)

    def test_recurrent_padding_and_privileged_state_do_not_reach_actor(self):
        torch.manual_seed(17)
        model = AsymmetricPPO(15, 12, 86, PickToAirEnv.decisions)
        seq = torch.randn(3, 28, 12)
        mask = torch.zeros(3, 28, dtype=torch.bool)
        mask[0, :10] = True
        mask[1, ::3] = True
        obs = Observation(torch.randn(3, 15), seq, mask, torch.randn(3, 86),
                          torch.tensor([0, 1, 0]), torch.tensor([0, -1, -1]))
        modified = obs._replace(seq=torch.where(mask[..., None], seq, -500.), priv=obs.priv + 50.)
        a, b = model.act(obs, deterministic=True), model.act(modified, deterministic=True)
        torch.testing.assert_close(a.action.continuous, b.action.continuous, rtol=0., atol=0.)
        torch.testing.assert_close(a.log_prob, b.log_prob, rtol=0., atol=0.)
        torch.testing.assert_close(a.action.choice, b.action.choice, rtol=0., atol=0.)
        self.assertGreater((a.value - b.value).abs().min().item(), 0.)


class MacroTests(unittest.TestCase):
    def check_observation(self, env, obs):
        b = env.batch
        self.assertEqual(obs.static.shape, (b, env.static_size))
        self.assertEqual(obs.seq.shape, (b, 28, env.seq_size))
        self.assertEqual(obs.valid_mask.shape, (b, 28))
        self.assertEqual(obs.priv.shape, (b, env.priv_size))
        self.assertEqual(obs.valid_mask.dtype, torch.bool)
        for channel in (obs.static, obs.seq, obs.priv):
            self.assertEqual(channel.dtype, torch.float32)
            self.assertTrue(torch.isfinite(channel).all().item())
        if isinstance(env, PickToAirEnv):
            ratios = obs.seq[..., 3]
            self.assertTrue(((ratios >= -.5) & (ratios <= 2.)).all().item())

    def test_pick_macro_history_and_retry(self):
        dr = DomainRandomizer(DEVICE, seed=17)
        env = PickToAirEnv(dr.sample(2))
        obs = env.reset(forced_mode=torch.arange(2, device=DEVICE))
        self.check_observation(env, obs)
        self.assertEqual(obs.valid_mask.sum(dim=1).tolist(), [10, 10])
        a = MacroAction(torch.arange(1, -1, -1, device=DEVICE), torch.zeros((2, 4), device=DEVICE))
        attempt = env.step(a)
        self.check_observation(env, attempt.obs)
        self.assertEqual(attempt.info["mode"].tolist(), [0, 1])
        self.assertEqual(attempt.obs.valid_mask.sum(dim=1).tolist(), [28, 28])
        self.assertEqual(attempt.obs.phase.tolist(), [1, 1])
        cached = env.observation()
        with patch.object(env.sim, "get_privileged_state", return_value=torch.full_like(cached.priv[:, :54], 123.)):
            observed = env.observation()
            torch.testing.assert_close(cached.static, observed.static, atol=0., rtol=0.)
            torch.testing.assert_close(cached.seq, observed.seq, atol=0., rtol=0.)
            self.assertTrue((observed.priv[:, :54] == 123.).all())
        sample_force = env.tare.clone()
        sample_force[:, 0, 2] += env.vector([-4., 4.])
        sensed = env.sensed._replace(force=sample_force)
        with patch.object(env, "sensed", sensed), patch.object(env, "load_reference", env.zeros() + 1.):
            env._sequence(1, env.flags(True), env.zeros() + 2.5)
        torch.testing.assert_close(env.observation().seq[:, -1, 3], env.vector([-.5, 2.]), rtol=0., atol=0.)
        retry = env.step(a._replace(choice=torch.full_like(a.choice, 2)))
        self.check_observation(env, retry.obs)
        self.assertFalse(retry.done.any())
        self.assertEqual(retry.obs.phase.tolist(), [0, 0])
        self.assertEqual(retry.obs.valid_mask.sum(dim=1).tolist(), [10, 10])

    def test_c10_four_consecutive_delayed_frames_after_physical_capture(self):
        dr = DomainRandomizer(DEVICE, seed=347)
        env = HandoverPlaceEnv(CornerSuite(dr).select(torch.tensor([9], device=DEVICE)))
        obs = env.reset(random_presentation=False)
        self.check_observation(env, obs)
        self.assertEqual(obs.valid_mask.sum().item(), 10)
        self.assertTrue(env.rgb.wrist_vis_valid.item())
        # Fixed interception command from the Stage 4 entry check. The test
        # uses the same contact, sensors and macro execution as policy rollout.
        params = env.vector([[0., -.1999771, -.3637438, .0178338, -.3161058, -.3171434]])
        step = env.step(MacroAction(torch.zeros(1, device=DEVICE, dtype=torch.long), params))
        self.check_observation(env, step.obs)
        self.assertEqual(step.obs.valid_mask.sum().item(), 28)
        self.assertEqual(step.obs.phase.item(), 1)
        self.assertFalse(env.rgb.wrist_vis_valid.item())
        self.assertFalse(step.info["dropped"].item())
        self.assertFalse(step.info["torn"].item())
        self.assertEqual(env.episode.yam.delay_steps.flatten().tolist(), [25, 25])
        self.assertEqual(env.episode.yam.coulomb_N.flatten().tolist(), [2., 2.])
        self.assertBetween(env.tug_target_N, 5.5, 6.5)
        self.assertBetween(env.tug_hold_s, .100, .150)
        good = ((env.tug_signal > 3.5).all(dim=-1) & env.tug_capture).flatten()
        starts = torch.where(good, torch.arange(1, len(good) + 1, device=DEVICE), 0)
        last_bad = torch.where(good, 0, torch.arange(1, len(good) + 1, device=DEVICE)).cummax(0).values
        frames = torch.where(good, starts - last_bad, 0).max().item()
        mean_hold = env.tug_truth[:, 11:17].mean(dim=1).flatten().tolist()
        print(f"C10 [{DEVICE}] {frames} consecutive frames ({frames * 20} ms), "
              f"hold mean {mean_hold} N, peak {env.tug_truth.amax().item():.3f} N", flush=True)
        self.assertGreaterEqual(frames, 4)
        for force in mean_hold:
            self.assertBetween(force, 5.5, 6.5)
        self.assertLess(env.tug_truth.amax().item(), 20.)

    def assertBetween(self, value, lo, hi):
        self.assertGreaterEqual(value, lo)
        self.assertLessEqual(value, hi)

    @unittest.skipUnless(DEVICE == "cuda", "4096-way check uses the RTX 5090")
    def test_4096_full_width_gpu_macro_batches(self):
        n = 4096
        dr = DomainRandomizer(DEVICE, seed=53)
        indices = torch.arange(n, device=DEVICE)
        action = MacroAction(indices % 2, torch.zeros((n, 6), device=DEVICE))
        for env_type in (PickToAirEnv, HandoverPlaceEnv):
            with self.subTest(env=env_type.__name__):
                env = env_type(dr.sample(n, curriculum_alpha=1.))
                self.check_observation(env, env.reset())
                for choices in (action.choice, indices % 3, action.choice):
                    result = env.step(action._replace(choice=choices))
                    self.check_observation(env, result.obs)
                    self.assertTrue(torch.isfinite(result.reward).all().item())
                    self.assertTrue(torch.isfinite(env.state.payload.pos).all().item())
                    self.assertTrue(torch.isfinite(env.state.payload.vel).all().item())
                    self.assertTrue(torch.isfinite(env.state.contacts_A).all().item())
                    self.assertTrue(torch.isfinite(env.state.contacts_B).all().item())
                print(f"{env_type.__name__}: {n} rows, {env.done.sum().item()} terminal, "
                      f"{env.success.sum().item()} successful", flush=True)


if __name__ == "__main__":
    unittest.main()

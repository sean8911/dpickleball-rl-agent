"""Tests for reward terms and the sim env's reward wiring."""

import unittest

import numpy as np

from agent import config, rewards
from agent.envs.sim_env import NetHugger, NOOP, SimPickleballEnv
from agent.sim import physics as P


class TestContactAnneal(unittest.TestCase):
    def test_anneals_to_zero_by_level(self):
        self.assertEqual(rewards.contact_bonus_scale(1), 1.0)
        self.assertEqual(
            rewards.contact_bonus_scale(config.CONTACT_ANNEAL_LEVEL),
            0.0,
        )
        self.assertEqual(rewards.contact_bonus_scale(5), 0.0)


class TestStepReward(unittest.TestCase):
    def test_return_pays_full(self):
        r = rewards.step_reward(
            counted_return=True, failed=False,
            first_contact=False, level=3,
        )
        self.assertEqual(r, config.RETURN_REWARD)

    def test_fail_is_negative(self):
        r = rewards.step_reward(
            counted_return=False, failed=True,
            first_contact=False, level=3,
        )
        self.assertEqual(r, config.FAIL_REWARD)

    def test_no_event_zero(self):
        r = rewards.step_reward(
            counted_return=False, failed=False,
            first_contact=False, level=5,
        )
        self.assertEqual(r, 0.0)

    def test_contact_bonus_only_early(self):
        early = rewards.step_reward(
            counted_return=False, failed=False,
            first_contact=True, level=1,
        )
        late = rewards.step_reward(
            counted_return=False, failed=False,
            first_contact=True, level=5,
        )
        self.assertGreater(early, 0.0)
        self.assertEqual(late, 0.0)


class TestNetHugger(unittest.TestCase):
    def test_camps_at_net(self):
        # from its back-court home the hugger should pin to the net
        rng = np.random.default_rng(0)
        sim = P.PickleballSim(rng=rng)
        sim.reset(P.LEFT)
        hugger = NetHugger(P.RIGHT, rng)
        for _ in range(200):
            sim.step(NOOP, hugger.act(sim))
        self.assertLess(sim.right.x, config.RIGHT_PADDLE_X[0] + 1.0)

    def test_tracks_ball_y(self):
        rng = np.random.default_rng(0)
        sim = P.PickleballSim(rng=rng)
        hugger = NetHugger(P.RIGHT, rng)
        sim.right.y = 30.0
        sim.ball = P.BallState(x=100.0, y=60.0)
        act = hugger.act(sim)
        self.assertEqual(act[0], 2)  # ball below -> move +y (down)
        self.assertEqual(act[2], 0)  # never rotates


class TestAdvanceReward(unittest.TestCase):
    def _camper_env(self) -> SimPickleballEnv:
        return SimPickleballEnv(
            level=config.MAX_LEVEL, seed=0, competitive=True,
            scripted_opponent="net_hugger", depth_shaping=True,
            fixed_side=P.LEFT,
        )

    def test_progress_rises_toward_opponent(self):
        # LEFT agent: deeper right = higher progress
        env = self._camper_env()
        env.reset(seed=0)
        env._sim.ball = P.BallState(x=config.SCORE_X_LEFT + 5.0, y=50.0)
        near = env._ball_progress()
        env._sim.ball = P.BallState(x=config.SCORE_X_RIGHT - 5.0, y=50.0)
        far = env._ball_progress()
        self.assertGreater(far, near)

    def test_progress_is_bounded(self):
        env = self._camper_env()
        env.reset(seed=0)
        env._sim.ball = P.BallState(x=config.SCORE_X_RIGHT + 10.0, y=50.0)
        self.assertLessEqual(env._ball_progress(), 1.0)
        env._sim.ball = P.BallState(x=config.SCORE_X_LEFT - 10.0, y=50.0)
        self.assertGreaterEqual(env._ball_progress(), 0.0)


class TestEnvContract(unittest.TestCase):
    def test_spaces(self):
        env = SimPickleballEnv(level=1, seed=0)
        self.assertEqual(
            env.observation_space.shape, (config.FEATURE_DIM,)
        )
        self.assertEqual(list(env.action_space.nvec), [3, 3, 3])

    def test_reset_returns_valid_obs(self):
        env = SimPickleballEnv(level=1, seed=0)
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, (config.FEATURE_DIM,))
        self.assertTrue(np.all(obs >= -1.0) and np.all(obs <= 1.0))

    def test_episode_terminates(self):
        env = SimPickleballEnv(level=1, seed=1)
        env.reset(seed=1)
        done = False
        for _ in range(config.MAX_EPISODE_STEPS + 5):
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                done = True
                break
        self.assertTrue(done)

    def test_both_sides_reachable(self):
        sides = set()
        env = SimPickleballEnv(level=1, seed=3)
        for s in range(40):
            env.reset(seed=s)
            sides.add(env.agent_side)
        self.assertSetEqual(sides, {P.LEFT, P.RIGHT})

    def test_fixed_side_respected(self):
        env = SimPickleballEnv(level=4, seed=0, fixed_side=P.RIGHT)
        for s in range(10):
            env.reset(seed=s)
            self.assertEqual(env.agent_side, P.RIGHT)

    def test_level_5_randomizes_physics(self):
        env = SimPickleballEnv(level=5, seed=0)
        env.reset(seed=0)
        # DR must stay physical: drag/restitution clamped <=1.0
        for _ in range(30):
            env.reset()
            self.assertLessEqual(env._sim.params.drag, 1.0)
            self.assertLessEqual(env._sim.params.wall_restitution, 1.0)
            self.assertGreaterEqual(env._sim.params.drag, 0.99)


if __name__ == "__main__":
    unittest.main()

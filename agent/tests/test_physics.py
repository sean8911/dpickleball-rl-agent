"""Tests for the calibrated 2D sim dynamics."""

import unittest

import numpy as np

from agent import config
from agent.sim import physics as P

NOOP = np.zeros(3, dtype=np.int32)


class TestPaddleMotion(unittest.TestCase):
    def test_vertical_speed_matches_calibration(self):
        sim = P.PickleballSim()
        y0 = sim.left.y
        sim.step(np.array([2, 0, 0]), NOOP)  # down
        self.assertAlmostEqual(
            sim.left.y - y0, config.PADDLE_V_SPEED, places=4
        )

    def test_paddle_clamped_to_bounds(self):
        sim = P.PickleballSim()
        for _ in range(200):
            sim.step(np.array([1, 0, 0]), NOOP)  # hold up
        self.assertGreaterEqual(sim.left.y, config.PADDLE_Y_RANGE[0])

    def test_horizontal_respects_side_bounds(self):
        sim = P.PickleballSim()
        for _ in range(300):
            sim.step(np.array([0, 1, 0]), NOOP)  # left paddle +x
        self.assertLessEqual(sim.left.x, config.LEFT_PADDLE_X[1])


class TestBallDynamics(unittest.TestCase):
    def test_drag_decays_speed(self):
        sim = P.PickleballSim()
        sim.ball = P.BallState(80.0, 50.0, vx=2.0, vy=0.0)
        sim.step(NOOP, NOOP)
        self.assertAlmostEqual(sim.ball.vx, 2.0 * config.BALL_DRAG,
                               places=4)

    def test_no_gravity(self):
        sim = P.PickleballSim()
        sim.ball = P.BallState(80.0, 50.0, vx=1.0, vy=0.0)
        sim.step(NOOP, NOOP)
        self.assertAlmostEqual(sim.ball.vy, 0.0, places=6)

    def test_top_wall_reflects_downward(self):
        sim = P.PickleballSim()
        sim.ball = P.BallState(80.0, config.WALL_Y_TOP + 3.5,
                               vx=0.0, vy=-2.0)
        res = sim.step(NOOP, NOOP)
        self.assertTrue(res.wall_bounce)
        self.assertGreater(sim.ball.vy, 0.0)

    def test_scoring_lines(self):
        sim = P.PickleballSim()
        sim.ball = P.BallState(config.SCORE_X_RIGHT - 1.0, 50.0,
                               vx=3.0, vy=0.0)
        res = sim.step(NOOP, NOOP)
        self.assertEqual(res.scored_side, P.LEFT)


class TestHits(unittest.TestCase):
    def test_left_paddle_returns_ball_rightward(self):
        sim = P.PickleballSim()
        sim.left.y = 50.0
        sim.ball = P.BallState(sim.left.x + 1.0, 50.0,
                               vx=-1.5, vy=0.0)
        res = sim.step(NOOP, NOOP)
        self.assertEqual(res.hit_side, P.LEFT)
        self.assertGreater(sim.ball.vx, 0.0)

    def test_self_start_still_ball_is_struck(self):
        # a near-still ball at the paddle must be launchable
        sim = P.PickleballSim()
        sim.left.y = 50.0
        sim.ball = P.BallState(sim.left.x + 1.0, 50.0,
                               vx=0.0, vy=0.0)
        res = sim.step(np.array([0, 1, 0]), NOOP)  # drive into ball
        self.assertEqual(res.hit_side, P.LEFT)
        self.assertGreater(sim.ball.vx, 0.0)

    def test_launched_ball_not_recaptured(self):
        # ball already moving fast toward opponent is not re-hit
        sim = P.PickleballSim()
        sim.left.y = 50.0
        sim.ball = P.BallState(sim.left.x + 1.0, 50.0,
                               vx=2.0, vy=0.0)
        res = sim.step(NOOP, NOOP)
        self.assertIsNone(res.hit_side)

    def test_serve_directed_at_opponent(self):
        sim = P.PickleballSim()
        sim.reset(P.RIGHT, serve_speed=1.5, serve_angle=0.0)
        self.assertLess(sim.ball.vx, 0.0)  # right serves leftward

    def test_invalid_serve_side_raises(self):
        sim = P.PickleballSim()
        with self.assertRaises(ValueError):
            sim.reset("up")


def _drive_hit(enable_slow_impulse, ball_speed, paddle_vx):
    """Apply one LEFT-paddle hit and return outgoing ball speed.

    The ball sits on the paddle face moving slowly toward it;
    paddle_vx>0 is a forward drive toward the right (opponent).
    """
    params = P.PhysicsParams(enable_slow_impulse=enable_slow_impulse)
    paddle = P.PaddleState(x=50.0, y=50.0)
    paddle.vx = paddle_vx
    paddle.vy = 0.0
    ball = P.BallState(x=50.0, y=50.0, vx=ball_speed, vy=0.0)
    P._apply_hit(ball, paddle, P.LEFT, params)
    return float(np.hypot(ball.vx, ball.vy))


class TestSlowImpulse(unittest.TestCase):
    def test_slow_forward_hit_is_boosted(self):
        speed = _drive_hit(True, ball_speed=0.2, paddle_vx=0.70)
        self.assertGreaterEqual(speed, 1.30)

    def test_slow_forward_hit_beats_baseline(self):
        base = _drive_hit(False, ball_speed=0.2, paddle_vx=0.70)
        boosted = _drive_hit(True, ball_speed=0.2, paddle_vx=0.70)
        self.assertGreater(boosted, base)

    def test_fast_ball_not_boosted(self):
        base = _drive_hit(False, ball_speed=2.0, paddle_vx=0.70)
        same = _drive_hit(True, ball_speed=2.0, paddle_vx=0.70)
        self.assertAlmostEqual(base, same, places=6)

    def test_no_forward_drive_not_boosted(self):
        base = _drive_hit(False, ball_speed=0.2, paddle_vx=0.0)
        same = _drive_hit(True, ball_speed=0.2, paddle_vx=0.0)
        self.assertAlmostEqual(base, same, places=6)

    def test_impulse_capped(self):
        speed = _drive_hit(True, ball_speed=0.2, paddle_vx=5.0)
        self.assertLessEqual(
            speed, P.PhysicsParams().slow_impulse_cap + 1e-6
        )


class TestParamsOverride(unittest.TestCase):
    def test_domain_randomization_override(self):
        params = P.PhysicsParams(drag=0.95, v_speed=1.0)
        sim = P.PickleballSim(params=params)
        y0 = sim.left.y
        sim.step(np.array([2, 0, 0]), NOOP)
        self.assertAlmostEqual(sim.left.y - y0, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

"""Single-agent gymnasium env over the calibrated 2D sim.

Trains one shared side-aware policy: the controlled side alternates
every episode (anti-asymmetry). Episodes are 50/50 receive (a ball
is served at the agent) and self-start (a near-still ball at the
agent's home that must be pushed out), the latter removing any
net-camping incentive. A return counts only when the ball clears
past mid toward the opponent. Levels 4-5 add a scripted returner;
level 5 randomizes physics per episode for sim-to-real robustness.
"""

import dataclasses
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from agent import config, rewards
from agent.curriculum import Level, get_level
from agent.features import pack_features
from agent.sim import physics as P

NOOP = np.zeros(3, dtype=np.int32)


def randomize_physics(
    rng: np.random.Generator,
    frac: float = config.DR_SPEED_FRAC,
) -> P.PhysicsParams:
    """Sample physics params jittered around the calibration.

    drag and restitution are sampled in tight absolute ranges
    clamped <= 1.0 (a multiplier > 1 would accelerate the ball or
    add wall energy); speeds and hit coefs take a proportional
    jitter. Used only at level 5 for sim-to-real robustness.
    """
    def jit(value: float) -> float:
        return float(value * (1.0 + rng.uniform(-frac, frac)))

    return P.PhysicsParams(
        drag=float(rng.uniform(*config.DR_DRAG_RANGE)),
        wall_restitution=float(rng.uniform(*config.DR_RESTITUTION_RANGE)),
        v_speed=jit(config.PADDLE_V_SPEED),
        h_speed=jit(config.PADDLE_H_SPEED),
        hit_incoming_coef=jit(config.HIT_INCOMING_COEF),
        hit_paddle_coef=jit(config.HIT_PADDLE_COEF),
    )


class ScriptedReturner:
    """Dumb ballistic opponent: track ball y, strike when close.

    Deliberately imperfect (randomized rotation) so it is a ball
    source for rally practice, not a benchmark to overfit. Acts
    only when the ball is on its own half.
    """

    def __init__(self, side: str, rng: np.random.Generator) -> None:
        self.side = side
        self.rng = rng

    def act(self, sim: P.PickleballSim) -> np.ndarray:
        paddle = sim.paddle(self.side)
        ball = sim.ball
        on_my_half = (
            ball.x >= config.COURT_MID_X if self.side == P.RIGHT
            else ball.x < config.COURT_MID_X
        )
        if not on_my_half:
            return NOOP
        dy = ball.y - paddle.y
        if dy > 1.0:
            vert = 2
        elif dy < -1.0:
            vert = 1
        else:
            vert = 0
        reach = config.PADDLE_WIDTH / 2.0 + config.BALL_RADIUS + 6.0
        if abs(ball.x - paddle.x) < reach:
            horiz = 2 if self.side == P.RIGHT else 1  # strike to net
        else:
            horiz = 1 if self.side == P.RIGHT else 2  # hold home
        rot = (
            int(self.rng.integers(0, 3))
            if self.rng.random() < 0.15 else 0
        )
        return np.array([vert, horiz, rot], dtype=np.int32)


class NetHugger:
    """Net-camping opponent: pin to the net and track the ball's y.

    Models opponent1's blocker: sits at the front of its half
    (the x-bound clamps it there) and matches the ball's y at the
    real paddle speed, so only fast or sharply-angled shots reach
    the open back court behind it.

    lag is a reaction delay in steps: it tracks where the ball WAS
    lag steps ago, so a direction change beats it. lag=0 is a perfect
    tracker (unbeatable -> no learning signal); larger lag is more
    beatable. Curriculum ramps lag high -> low toward opp1's level.
    """

    def __init__(
        self, side: str, rng: np.random.Generator, lag: int = 0
    ) -> None:
        self.side = side
        self.rng = rng
        self.lag = int(lag)
        self._y_hist: list = []

    def act(self, sim: P.PickleballSim) -> np.ndarray:
        paddle = sim.paddle(self.side)
        self._y_hist.append(sim.ball.y)
        target_y = self._y_hist[max(0, len(self._y_hist) - 1 - self.lag)]
        dy = target_y - paddle.y
        if dy > 1.0:
            vert = 2
        elif dy < -1.0:
            vert = 1
        else:
            vert = 0
        # always push toward the net; the x-bound pins it there
        horiz = 2 if self.side == P.RIGHT else 1
        return np.array([vert, horiz, 0], dtype=np.int32)


class SimPickleballEnv(gym.Env):
    """Curriculum receive/rally task for one controlled side."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        level: int = 1,
        seed: Optional[int] = None,
        fixed_side: Optional[str] = None,
        opponent_paths: Optional[list] = None,
        competitive: bool = False,
        scripted_opponent: Optional[str] = None,
        depth_shaping: bool = False,
        hugger_lag: int = 0,
        enable_rotation: bool = False,
        enable_slow_impulse: bool = False,
        scripted_opponent_prob: float = 1.0,
    ) -> None:
        super().__init__()
        if fixed_side is not None and fixed_side not in P.SIDES:
            raise ValueError(f"fixed_side must be in {P.SIDES}")
        if scripted_opponent not in (None, "net_hugger"):
            raise ValueError(
                f"unknown scripted_opponent: {scripted_opponent!r}"
            )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(config.FEATURE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            [config.ACTION_CHOICES] * config.ACTION_BRANCHES
        )
        self._level_idx = level
        self._fixed_side = fixed_side
        self._rng = np.random.default_rng(seed)
        self._sim = P.PickleballSim(rng=self._rng)
        self._returner: Optional[ScriptedReturner] = None
        # self-play snapshots sampled per episode, cached
        self._opponent_paths = list(opponent_paths or [])
        self._opp_cache: dict = {}
        self._opp_model = None
        # competitive reward = game points, non-saturating
        self._competitive = competitive
        # anti-camper: net-hugger opponent + depth bonus
        self._scripted_opponent = scripted_opponent
        self._depth_shaping = depth_shaping
        self._hugger_lag = hugger_lag
        self._enable_rotation = enable_rotation
        self._enable_slow_impulse = enable_slow_impulse
        # camper/pool mix: net-hugger this fraction, else a snapshot
        self._scripted_opponent_prob = scripted_opponent_prob
        self.agent_side = P.LEFT
        self._reset_episode_state()

    def set_level(self, level: int) -> None:
        """Update the curriculum level (called across vec envs)."""
        self._level_idx = int(level)

    def set_opponent_paths(self, paths: list) -> None:
        """Replace the self-play opponent pool (called across envs)."""
        self._opponent_paths = list(paths)

    def set_hugger_lag(self, lag: int) -> None:
        """Set net-hugger reaction lag (curriculum, across envs)."""
        self._hugger_lag = int(lag)

    def _sample_opponent(self):
        """Pick and lazily load one pool opponent for this episode."""
        if not self._opponent_paths:
            return None
        path = self._opponent_paths[
            int(self._rng.integers(len(self._opponent_paths)))
        ]
        if path not in self._opp_cache:
            self._opp_cache[path] = torch.jit.load(path).eval()
        return self._opp_cache[path]

    @property
    def level(self) -> Level:
        return get_level(self._level_idx)

    def _reset_episode_state(self) -> None:
        self._steps = 0
        self._returns = 0
        self._time_own_side = 0
        self._pending_return = False
        self._contacted_incoming = False
        self._points_for = 0
        self._points_against = 0
        self._time_opp_side = 0
        self._prev_progress = 0.0

    def _opponent_side(self) -> str:
        return P.RIGHT if self.agent_side == P.LEFT else P.LEFT

    def _ball_on_agent_side(self) -> bool:
        if self.agent_side == P.LEFT:
            return self._sim.ball.x < config.COURT_MID_X
        return self._sim.ball.x >= config.COURT_MID_X

    def _ball_cleared_to_opponent(self) -> bool:
        if self.agent_side == P.LEFT:
            return (
                self._sim.ball.x
                > config.COURT_MID_X + config.RETURN_CLEAR_MARGIN_PX
            )
        return (
            self._sim.ball.x
            < config.COURT_MID_X - config.RETURN_CLEAR_MARGIN_PX
        )

    def _ball_progress(self) -> float:
        """Ball progress toward the opponent line: 0 ours -> 1 opp."""
        span = config.SCORE_X_RIGHT - config.SCORE_X_LEFT
        if self.agent_side == P.LEFT:
            frac = (self._sim.ball.x - config.SCORE_X_LEFT) / span
        else:
            frac = (config.SCORE_X_RIGHT - self._sim.ball.x) / span
        return float(min(1.0, max(0.0, frac)))

    def _serve_toward(self, target_side: str, level: Level) -> None:
        """Serve from the far side so the ball travels at target."""
        speed = float(self._rng.uniform(*level.speed_range))
        hi = level.angle_range[1]
        angle = float(self._rng.uniform(-hi, hi))
        server = P.RIGHT if target_side == P.LEFT else P.LEFT
        self._sim.reset(server, speed, angle)

    def _serve_at_agent(self, level: Level) -> None:
        self._serve_toward(self.agent_side, level)

    def _competitive_serve(self, level: Level) -> None:
        """Serve toward a random side so both players receive."""
        target = P.LEFT if self._rng.random() < 0.5 else P.RIGHT
        self._serve_toward(target, level)

    def _self_start(self) -> None:
        # near-still ball at the agent's home; must be pushed out
        self._sim.reset(self._opponent_side(), 0.9, 0.0)
        paddle = self._sim.paddle(self.agent_side)
        offset = 3.0 if self.agent_side == P.LEFT else -3.0
        self._sim.ball = P.BallState(
            x=paddle.x + offset, y=paddle.y, vx=0.0, vy=0.0
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        level = self.level
        params = (
            randomize_physics(self._rng)
            if level.randomize_physics else P.PhysicsParams()
        )
        if self._enable_rotation:
            # inject rotation only for the open-loop experiment
            params = dataclasses.replace(
                params, rot_speed=config.ROTATION_SPEED
            )
        if self._enable_slow_impulse:
            # real Unity mechanic, off by default like rotation
            params = dataclasses.replace(
                params, enable_slow_impulse=True
            )
        self._sim = P.PickleballSim(params=params, rng=self._rng)
        if self._fixed_side is not None:
            self.agent_side = self._fixed_side
        else:
            self.agent_side = (
                P.LEFT if self._rng.random() < 0.5 else P.RIGHT
            )
        use_camper = self._scripted_opponent == "net_hugger" and (
            not self._opponent_paths
            or self._rng.random() < self._scripted_opponent_prob
        )
        if use_camper:
            # net-hugger overrides both the pool and the returner
            self._returner = NetHugger(
                self._opponent_side(), self._rng, lag=self._hugger_lag
            )
            self._opp_model = None
        else:
            self._returner = (
                ScriptedReturner(self._opponent_side(), self._rng)
                if level.has_returner else None
            )
            self._opp_model = self._sample_opponent()
        self._reset_episode_state()
        if self._competitive:
            # real game: always serve a random side, no self-start
            self._competitive_serve(level)
        elif self._rng.random() < config.SELF_START_PROB:
            self._self_start()
        else:
            self._serve_at_agent(level)
        self._prev_progress = self._ball_progress()
        return self._observe(), {}

    def _features_for(self, side: str) -> np.ndarray:
        """Build the 16-dim observation from one side's view.

        time_own_side and success_count are left at 0: the extractor
        cannot measure them from a single frame, so feeding them in
        sim would be privileged info absent at deploy. The paddle
        angle is forced vertical (pi/2) to match the extractor (which
        cannot read tilt) -- so even with rotation enabled the policy
        never observes its angle (open-loop) and parity holds.
        """
        ball = self._sim.ball
        paddle = self._sim.paddle(side)
        opp = self._sim.paddle(P.RIGHT if side == P.LEFT else P.LEFT)
        side_sign = 1.0 if side == P.RIGHT else -1.0
        return pack_features(
            ball_xy=(ball.x, ball.y),
            ball_vxy=(ball.vx, ball.vy),
            paddle_xy=(paddle.x, paddle.y),
            paddle_angle_rad=float(np.pi / 2.0),
            side_sign=side_sign,
            time_own_side_norm=0.0,
            success_count_norm=0.0,
            opponent_y=opp.y,
        )

    def _observe(self) -> np.ndarray:
        return self._features_for(self.agent_side)

    def _opponent_action(self) -> np.ndarray:
        """Opponent action: pool policy if set, else returner/noop."""
        if self._opp_model is not None:
            feats = self._features_for(self._opponent_side())
            tensor = torch.from_numpy(feats).float().unsqueeze(0)
            with torch.no_grad():
                return self._opp_model(tensor)[0].numpy().astype(
                    np.int32
                )
        if self._returner is not None:
            return self._returner.act(self._sim)
        return NOOP

    def _step_sim(self, action: np.ndarray) -> P.StepResult:
        agent_act = np.asarray(action, dtype=np.int32)
        opp_act = self._opponent_action()
        if self.agent_side == P.LEFT:
            return self._sim.step(agent_act, opp_act)
        return self._sim.step(opp_act, agent_act)

    def _competitive_step(self, result, level):
        """Points-based self-play step: +1 agent scores, -1 opp.

        Multi-point episode: re-serve at the agent after each point
        and play to the step cap. The reward never saturates while
        the pool opponent stays competitive.
        """
        # symmetric stall: the side the ball stalls on concedes
        bx = self._sim.ball.x
        # vs a camper, faithful >5s stall rule, no deadzone
        dead = (
            0.0 if self._scripted_opponent
            else config.COMPETITIVE_STALL_DEADZONE_PX
        )
        if abs(bx - config.COURT_MID_X) <= dead:
            # midcourt rally: neither side is stalling
            self._time_own_side = 0
            self._time_opp_side = 0
        elif self._ball_on_agent_side():
            self._time_own_side += 1
            self._time_opp_side = 0
        else:
            self._time_opp_side += 1
            self._time_own_side = 0
        agent_stall = self._time_own_side >= config.OWN_SIDE_TIMEOUT_STEPS
        opp_stall = self._time_opp_side >= config.OWN_SIDE_TIMEOUT_STEPS
        scored_agent = result.scored_side == self.agent_side or opp_stall
        scored_opp = (
            result.scored_side == self._opponent_side() or agent_stall
        )
        reward = 0.0
        cause = ""
        if result.scored_side == self.agent_side:
            cause = "agent_score"
        elif opp_stall:
            cause = "opp_stall"
        elif result.scored_side == self._opponent_side():
            cause = "opp_score"
        elif agent_stall:
            cause = "agent_stall"
        if scored_agent:
            reward = 1.0
            self._points_for += 1
        elif scored_opp:
            reward = -1.0
            self._points_against += 1
        if scored_agent or scored_opp:
            self._competitive_serve(level)
            self._time_own_side = 0
            self._time_opp_side = 0
            self._prev_progress = self._ball_progress()
        elif self._depth_shaping:
            progress = self._ball_progress()
            reward += config.BALL_ADVANCE_COEF * (
                progress - self._prev_progress
            )
            self._prev_progress = progress
        truncated = self._steps >= config.MAX_EPISODE_STEPS
        info = {
            "agent_side": self.agent_side,
            "points_for": self._points_for,
            "points_against": self._points_against,
            "cause": cause,
        }
        return self._observe(), reward, False, truncated, info

    def step(self, action):
        level = self.level
        self._steps += 1
        result = self._step_sim(action)
        if self._competitive:
            return self._competitive_step(result, level)

        agent_hit = result.hit_side == self.agent_side
        first_contact = agent_hit and not self._contacted_incoming
        if agent_hit:
            self._contacted_incoming = True
            self._pending_return = True

        counted_return = False
        if self._pending_return and self._ball_cleared_to_opponent():
            counted_return = True
            self._returns += 1
            self._pending_return = False

        if self._ball_on_agent_side():
            self._time_own_side += 1
        else:
            self._time_own_side = 0
            self._contacted_incoming = False  # reset for next ball

        # opponent conceding is a receive success; re-serve
        if result.scored_side == self.agent_side:
            self._serve_at_agent(level)

        own_timeout = (
            self._time_own_side >= config.OWN_SIDE_TIMEOUT_STEPS
        )
        scored_on = result.scored_side == self._opponent_side()
        failed = scored_on or own_timeout
        reached_target = self._returns >= config.TRIAL_RETURN_TARGET
        truncated = self._steps >= config.MAX_EPISODE_STEPS

        reward = rewards.step_reward(
            counted_return=counted_return,
            failed=failed,
            first_contact=first_contact,
            level=level.index,
        )
        terminated = failed or reached_target
        info = {
            "returns": self._returns,
            "agent_side": self.agent_side,
            "success": reached_target,
        }
        return self._observe(), reward, terminated, truncated, info

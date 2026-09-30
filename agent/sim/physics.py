"""Calibrated 2D pickleball dynamics.

Stateful simulator stepped at 30 FPS in image pixels. The ball
rebounds off the top/bottom (y) walls and is returned by two
rotatable paddles; a point is scored when the ball passes a
left/right scoring line. Physics constants come from config
(Phase 1 calibration) via PhysicsParams, which tests and
domain-randomization may override.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from agent import config

LEFT = "left"
RIGHT = "right"
SIDES = (LEFT, RIGHT)

# faster than this = already launched, not re-captured
AWAY_VX_TOL = 0.3


@dataclass(frozen=True)
class PhysicsParams:
    """Tunable physics constants (defaults = Phase 1 calibration)."""

    drag: float = config.BALL_DRAG
    ball_radius: float = config.BALL_RADIUS
    wall_y_top: float = config.WALL_Y_TOP
    wall_y_bot: float = config.WALL_Y_BOT
    wall_restitution: float = config.WALL_RESTITUTION
    score_x_left: float = config.SCORE_X_LEFT
    score_x_right: float = config.SCORE_X_RIGHT
    left_paddle_x: Tuple[float, float] = config.LEFT_PADDLE_X
    right_paddle_x: Tuple[float, float] = config.RIGHT_PADDLE_X
    paddle_y_range: Tuple[float, float] = config.PADDLE_Y_RANGE
    v_speed: float = config.PADDLE_V_SPEED
    h_speed: float = config.PADDLE_H_SPEED
    rot_speed: float = config.PADDLE_ROT_SPEED
    angle_range: Tuple[float, float] = config.PADDLE_ANGLE_RANGE
    paddle_len: float = config.PADDLE_LEN
    paddle_width: float = config.PADDLE_WIDTH
    hit_incoming_coef: float = config.HIT_INCOMING_COEF
    hit_paddle_coef: float = config.HIT_PADDLE_COEF
    hit_speed_min: float = config.HIT_SPEED_MIN
    hit_speed_max: float = config.HIT_SPEED_MAX
    enable_slow_impulse: bool = False
    slow_impulse_threshold: float = config.SLOW_IMPULSE_THRESHOLD
    slow_impulse_floor: float = config.SLOW_IMPULSE_FLOOR
    slow_impulse_gain: float = config.SLOW_IMPULSE_GAIN
    slow_impulse_cap: float = config.SLOW_IMPULSE_CAP


@dataclass
class PaddleState:
    """Mutable paddle pose and last-step velocity, per side."""

    x: float
    y: float
    angle: float = math.pi / 2.0
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class BallState:
    """Mutable ball position and velocity."""

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class StepResult:
    """What happened in one sim step."""

    scored_side: Optional[str] = None
    hit_side: Optional[str] = None
    wall_bounce: bool = False


def _x_bounds(params: PhysicsParams, side: str) -> Tuple[float, float]:
    return (
        params.left_paddle_x if side == LEFT
        else params.right_paddle_x
    )


def _apply_action(
    paddle: PaddleState,
    action: np.ndarray,
    params: PhysicsParams,
    side: str,
) -> None:
    """Move a paddle by one held action step, clamped to bounds.

    Action branches use the measured global mapping: vertical
    1=up/2=down, horizontal 1=+x/2=-x, rotation 1=ccw/2=cw.
    """
    prev_x, prev_y = paddle.x, paddle.y
    if action[0] == 1:
        paddle.y -= params.v_speed
    elif action[0] == 2:
        paddle.y += params.v_speed
    if action[1] == 1:
        paddle.x += params.h_speed
    elif action[1] == 2:
        paddle.x -= params.h_speed
    if action[2] == 1:
        paddle.angle -= params.rot_speed
    elif action[2] == 2:
        paddle.angle += params.rot_speed
    x_lo, x_hi = _x_bounds(params, side)
    y_lo, y_hi = params.paddle_y_range
    a_lo, a_hi = params.angle_range
    paddle.x = float(np.clip(paddle.x, x_lo, x_hi))
    paddle.y = float(np.clip(paddle.y, y_lo, y_hi))
    paddle.angle = float(np.clip(paddle.angle, a_lo, a_hi))
    paddle.vx = paddle.x - prev_x
    paddle.vy = paddle.y - prev_y


def _reflect_walls(ball: BallState, params: PhysicsParams) -> bool:
    """Reflect the ball off top/bottom walls; return True if hit."""
    r = params.ball_radius
    top = params.wall_y_top + r
    bot = params.wall_y_bot - r
    bounced = False
    if ball.y < top:
        ball.y = top + (top - ball.y)
        ball.vy = abs(ball.vy) * params.wall_restitution
        bounced = True
    elif ball.y > bot:
        ball.y = bot - (ball.y - bot)
        ball.vy = -abs(ball.vy) * params.wall_restitution
        bounced = True
    return bounced


def _paddle_contact(
    ball: BallState, paddle: PaddleState, side: str,
    params: PhysicsParams,
) -> bool:
    """True if the ball overlaps the paddle face this step.

    The paddle is a capture band in x (its width plus ball radius)
    and its length in y; the rotated face only shapes the outgoing
    angle, not capture.
    """
    half_len = params.paddle_len / 2.0
    reach_x = params.paddle_width / 2.0 + params.ball_radius
    if abs(ball.x - paddle.x) > reach_x:
        return False
    if abs(ball.y - paddle.y) > half_len + params.ball_radius:
        return False
    # capture an incoming/near-still ball, skip one launched fast
    return (
        ball.vx < AWAY_VX_TOL if side == LEFT
        else ball.vx > -AWAY_VX_TOL
    )


def _apply_hit(
    ball: BallState, paddle: PaddleState, side: str,
    params: PhysicsParams,
) -> None:
    """Set outgoing velocity from incoming speed, paddle motion,
    and paddle tilt; eject the ball clear of the paddle face."""
    s_in = math.hypot(ball.vx, ball.vy)
    paddle_motion = math.hypot(paddle.vx, paddle.vy)
    speed = (
        params.hit_incoming_coef * s_in
        + params.hit_paddle_coef * paddle_motion
        + params.hit_speed_min
    )
    speed = float(
        np.clip(speed, params.hit_speed_min, params.hit_speed_max)
    )
    # slow ball driven forward launches hard (real Unity mechanic)
    if (
        params.enable_slow_impulse
        and s_in < params.slow_impulse_threshold
    ):
        forward = paddle.vx if side == LEFT else -paddle.vx
        if forward > 0.0:
            push_quality = min(1.0, forward / params.h_speed)
            boost = (
                params.slow_impulse_floor
                + params.slow_impulse_gain * push_quality
            )
            # push-through regime tops out at the cap (opp1 model)
            speed = min(max(speed, boost), params.slow_impulse_cap)
    # tilt from vertical deflects in y; paddle vy adds a slice
    tilt = paddle.angle - math.pi / 2.0
    vy = math.sin(tilt) * speed + paddle.vy
    vy = float(np.clip(vy, -speed * 0.9, speed * 0.9))
    vx_mag = math.sqrt(max(speed * speed - vy * vy, 1e-6))
    ball.vx = vx_mag if side == LEFT else -vx_mag
    ball.vy = vy
    reach_x = params.paddle_width / 2.0 + params.ball_radius
    ball.x = (
        paddle.x + reach_x if side == LEFT else paddle.x - reach_x
    )


class PickleballSim:
    """Stateful 2D pickleball simulator for one rally.

    Coordinates and velocities are in image pixels and px/step.
    Call reset(serve_side) then step(action_left, action_right);
    read state via ball/left/right.
    """

    def __init__(
        self,
        params: Optional[PhysicsParams] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.params = params or PhysicsParams()
        self.rng = rng or np.random.default_rng()
        self.ball = BallState(config.COURT_MID_X, 50.0)
        self.left = PaddleState(*self._home(LEFT))
        self.right = PaddleState(*self._home(RIGHT))

    def _home(self, side: str) -> Tuple[float, float]:
        x_lo, x_hi = _x_bounds(self.params, side)
        mid_y = sum(self.params.paddle_y_range) / 2.0
        x = x_lo if side == LEFT else x_hi  # back of each half
        return (x, mid_y)

    def paddle(self, side: str) -> PaddleState:
        return self.left if side == LEFT else self.right

    def reset(
        self,
        serve_side: str,
        serve_speed: Optional[float] = None,
        serve_angle: Optional[float] = None,
    ) -> None:
        """Place paddles home and serve toward the opponent."""
        if serve_side not in SIDES:
            raise ValueError(f"serve_side must be in {SIDES}")
        self.left = PaddleState(*self._home(LEFT))
        self.right = PaddleState(*self._home(RIGHT))
        server = self.paddle(serve_side)
        if serve_speed is None:
            serve_speed = float(
                self.rng.uniform(*config.SERVE_SPEED_RANGE)
            )
        if serve_angle is None:
            serve_angle = float(self.rng.uniform(-0.6, 0.6))
        toward = 1.0 if serve_side == LEFT else -1.0
        self.ball = BallState(
            x=server.x + toward * (self.params.paddle_width + 2.0),
            y=server.y,
            vx=toward * serve_speed * math.cos(serve_angle),
            vy=serve_speed * math.sin(serve_angle),
        )

    def step(
        self, action_left: np.ndarray, action_right: np.ndarray
    ) -> StepResult:
        """Advance one frame; return the events that occurred."""
        _apply_action(self.left, action_left, self.params, LEFT)
        _apply_action(self.right, action_right, self.params, RIGHT)

        self.ball.x += self.ball.vx
        self.ball.y += self.ball.vy
        self.ball.vx *= self.params.drag
        self.ball.vy *= self.params.drag

        result = StepResult()
        result.wall_bounce = _reflect_walls(self.ball, self.params)

        for side in SIDES:
            paddle = self.paddle(side)
            if _paddle_contact(self.ball, paddle, side, self.params):
                _apply_hit(self.ball, paddle, side, self.params)
                result.hit_side = side
                break

        if self.ball.x <= self.params.score_x_left:
            result.scored_side = RIGHT  # right scores at left line
        elif self.ball.x >= self.params.score_x_right:
            result.scored_side = LEFT
        return result

"""Frozen 16-dim feature-vector contract.

The ONLY place the feature layout is defined. Checkpoints store the
dimension in metadata and validate it against config.FEATURE_DIM on
load; sim, extractor, and deployment all pack through here.
"""

import math
from typing import Optional, Tuple

import numpy as np

from agent import config

FEATURE_NAMES: Tuple[str, ...] = (
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    "paddle_x",
    "paddle_y",
    "paddle_sin",
    "paddle_cos",
    "rel_x",
    "rel_y",
    "time_own_side",
    "success_count",
    "ball_visible",
    "paddle_visible",
    "side_sign",
    "opponent_y",
)

assert len(FEATURE_NAMES) == config.FEATURE_DIM


def _norm_x(x: float, img_w: int) -> float:
    return (x / max(1, img_w - 1)) * 2.0 - 1.0


def _norm_y(y: float, img_h: int) -> float:
    return (y / max(1, img_h - 1)) * 2.0 - 1.0


def pack_features(
    *,
    ball_xy: Tuple[float, float],
    ball_vxy: Tuple[float, float],
    paddle_xy: Tuple[float, float],
    paddle_angle_rad: float,
    side_sign: float,
    ball_visible: float = 1.0,
    paddle_visible: float = 1.0,
    time_own_side_norm: float = 0.0,
    success_count_norm: float = 0.0,
    opponent_y: Optional[float] = None,
    img_w: int = config.IMG_W,
    img_h: int = config.IMG_H,
) -> np.ndarray:
    """Pack raw pixel-space state into the 16-dim feature vector.

    All positions are in image pixels; velocity is px/step. Output
    values are normalized to [-1, 1] (flags/timers to [0, 1]).

    Raises:
        ValueError: if side_sign is not +1.0 or -1.0.
    """
    if side_sign not in (1.0, -1.0):
        raise ValueError(f"side_sign must be +/-1.0, got {side_sign}")
    bx, by = ball_xy
    px, py = paddle_xy
    opp_y = img_h * 0.5 if opponent_y is None else float(opponent_y)
    vel_n = config.BALL_VEL_NORM
    features = np.array(
        [
            _norm_x(bx, img_w),
            _norm_y(by, img_h),
            float(np.clip(ball_vxy[0] / vel_n, -1.0, 1.0)),
            float(np.clip(ball_vxy[1] / vel_n, -1.0, 1.0)),
            _norm_x(px, img_w),
            _norm_y(py, img_h),
            math.sin(paddle_angle_rad),
            math.cos(paddle_angle_rad),
            float(np.clip((bx - px) / img_w, -1.0, 1.0)),
            float(np.clip((by - py) / img_h, -1.0, 1.0)),
            float(np.clip(time_own_side_norm, 0.0, 1.0)),
            float(np.clip(success_count_norm, 0.0, 1.0)),
            float(ball_visible),
            float(paddle_visible),
            float(side_sign),
            float(np.clip(_norm_y(opp_y, img_h), -1.0, 1.0)),
        ],
        dtype=np.float32,
    )
    if features.shape != (config.FEATURE_DIM,):
        raise ValueError(
            f"packed {features.shape}, expected ({config.FEATURE_DIM},)"
        )
    return features

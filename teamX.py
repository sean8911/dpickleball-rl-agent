"""Competition submission: left paddle.

OpenCV extractor, 16-dim features, and TorchScript policy in one file;
loads checkpoints/best_1_policy.pt.
"""

import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch

MODEL_PATH = Path(__file__).resolve().parent / "checkpoints" / \
    "best_1_policy.pt"

IMG_H, IMG_W = 84, 168
FEATURE_DIM = 16

BALL_HSV_LO, BALL_HSV_HI = (20, 80, 80), (45, 255, 255)
RACKET_ORANGE_LO, RACKET_ORANGE_HI = (5, 45, 45), (30, 255, 255)
# red wraps hue 0/179, so two ranges
RACKET_RED1_LO, RACKET_RED1_HI = (0, 45, 45), (6, 255, 255)
RACKET_RED2_LO, RACKET_RED2_HI = (170, 45, 45), (179, 255, 255)

ROI_X1, ROI_X2 = 8, 164
ROI_Y1, ROI_Y2 = 20, 80
BALL_MIN_AREA, BALL_MAX_AREA = 3, 160
PADDLE_MIN_AREA = 15

BALL_VEL_NORM = 8.0
MAX_BALL_DELTA_PX = 6.0
PADDLE_FALLBACK_X_RIGHT, PADDLE_FALLBACK_X_LEFT = 0.82, 0.18
PADDLE_FALLBACK_Y = 0.55

# hold vertical within this many px of the ball (stops the hop)
Y_DEADBAND_PX = 3.0
_WARMUP_CALLS = 3
# horizontal mirror: right<->left, none unchanged
_MIRROR_ACTION = {0: 0, 1: 2, 2: 1}
# paddle pinned vertical
_PADDLE_SIN, _PADDLE_COS = math.sin(math.pi / 2), math.cos(math.pi / 2)


def _norm(v: float, hi: int) -> float:
    return (v / max(1, hi - 1)) * 2.0 - 1.0


def _pack_features(
    ball_xy: Tuple[float, float],
    ball_vxy: Tuple[float, float],
    paddle_xy: Tuple[float, float],
    side_sign: float,
    ball_visible: float,
    paddle_visible: float,
    opponent_y: Optional[float],
) -> np.ndarray:
    """Pack pixel-space state into the frozen 16-dim feature vector."""
    bx, by = ball_xy
    px, py = paddle_xy
    opp_y = IMG_H * 0.5 if opponent_y is None else float(opponent_y)
    return np.array(
        [
            _norm(bx, IMG_W),
            _norm(by, IMG_H),
            float(np.clip(ball_vxy[0] / BALL_VEL_NORM, -1.0, 1.0)),
            float(np.clip(ball_vxy[1] / BALL_VEL_NORM, -1.0, 1.0)),
            _norm(px, IMG_W),
            _norm(py, IMG_H),
            _PADDLE_SIN,
            _PADDLE_COS,
            float(np.clip((bx - px) / IMG_W, -1.0, 1.0)),
            float(np.clip((by - py) / IMG_H, -1.0, 1.0)),
            0.0,
            0.0,
            float(ball_visible),
            float(paddle_visible),
            float(side_sign),
            float(np.clip(_norm(opp_y, IMG_H), -1.0, 1.0)),
        ],
        dtype=np.float32,
    )


def _largest_blob(
    mask: np.ndarray, min_area: float, max_area: float = 1e9
) -> Optional[Tuple[float, float]]:
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        return (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))
    return None


class _Extractor:
    """Side-aware OpenCV frame-to-feature extractor."""

    def __init__(self, side: str) -> None:
        self.side = side
        self.prev_ball_xy: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self.prev_ball_xy = None

    def _rgb(self, img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            scale = 255.0 if arr.max() <= 1.0 else 1.0
            arr = (arr * scale).clip(0, 255).astype(np.uint8)
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    def _masks(self, img: np.ndarray):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        roi = np.zeros(img.shape[:2], dtype=np.uint8)
        roi[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = 255
        ball = cv2.bitwise_and(
            cv2.inRange(hsv, np.array(BALL_HSV_LO),
                        np.array(BALL_HSV_HI)),
            roi,
        )
        ball = cv2.morphologyEx(
            ball, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )
        racket = cv2.bitwise_or(
            cv2.inRange(hsv, np.array(RACKET_ORANGE_LO),
                        np.array(RACKET_ORANGE_HI)),
            cv2.bitwise_or(
                cv2.inRange(hsv, np.array(RACKET_RED1_LO),
                            np.array(RACKET_RED1_HI)),
                cv2.inRange(hsv, np.array(RACKET_RED2_LO),
                            np.array(RACKET_RED2_HI)),
            ),
        )
        racket = cv2.morphologyEx(
            cv2.bitwise_and(racket, roi), cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8),
        )
        return ball, racket

    def features_from_image(self, img: np.ndarray) -> np.ndarray:
        """Detect on one frame and pack the 16-dim feature vector."""
        rgb = self._rgb(img)
        ball_mask, racket = self._masks(rgb)
        ball_xy = _largest_blob(
            ball_mask, BALL_MIN_AREA, BALL_MAX_AREA
        )
        controlled, opponent = racket.copy(), racket.copy()
        half = IMG_W // 2
        if self.side == "right":
            controlled[:, :half] = 0
            opponent[:, half:] = 0
        else:
            controlled[:, half:] = 0
            opponent[:, :half] = 0
        paddle_xy = _largest_blob(controlled, PADDLE_MIN_AREA)
        opponent_xy = _largest_blob(opponent, PADDLE_MIN_AREA)

        if ball_xy is None:
            bx, by = 0.5 * IMG_W, 0.5 * IMG_H
            ball_visible = 0.0
        else:
            bx, by = ball_xy
            ball_visible = 1.0
        # velocity from frame delta; prev kept across short dropouts
        if self.prev_ball_xy is None or ball_xy is None:
            bvx, bvy = 0.0, 0.0
        else:
            bvx = bx - self.prev_ball_xy[0]
            bvy = by - self.prev_ball_xy[1]
            # discard occlusion-recovery jumps, not real motion
            if math.hypot(bvx, bvy) > MAX_BALL_DELTA_PX:
                bvx, bvy = 0.0, 0.0
        if ball_xy is not None:
            self.prev_ball_xy = (bx, by)

        if paddle_xy is None:
            frac = (PADDLE_FALLBACK_X_RIGHT if self.side == "right"
                    else PADDLE_FALLBACK_X_LEFT)
            px, py = IMG_W * frac, IMG_H * PADDLE_FALLBACK_Y
            paddle_visible = 0.0
        else:
            px, py = paddle_xy
            paddle_visible = 1.0

        side_sign = 1.0 if self.side == "right" else -1.0
        opp_y = opponent_xy[1] if opponent_xy is not None else None
        return _pack_features(
            (bx, by), (bvx, bvy), (px, py), side_sign,
            ball_visible, paddle_visible, opp_y,
        )


def _mirror_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    is_chw = arr.shape[0] == 3 and arr.shape[-1] != 3
    return np.ascontiguousarray(np.flip(arr, axis=2 if is_chw else 1))


def _extract_frame(observation: Any) -> np.ndarray:
    """Pull and copy the frame from any observation form.

    Copied because Competition.py may reuse the buffer mid-call.
    """
    if isinstance(observation, dict):
        first = next(iter(observation.values()))
        if isinstance(first, dict) and "observation" in first:
            frame = first["observation"][0]
        elif "observation" in observation:
            frame = observation["observation"][0]
        else:
            raise ValueError("unrecognized observation dict structure")
    else:
        frame = observation
    return np.array(frame, copy=True)


class _Policy:
    """Side-aware policy under the 10ms step budget.

    The right paddle mirrors into the left frame and mirrors the action
    back; a deadband holds vertical when aligned.
    """

    def __init__(self, side: str, mirror_obs: bool) -> None:
        torch.set_num_threads(1)  # two policies share the CPU
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"model not found: {MODEL_PATH}")
        self.mirror_obs = mirror_obs
        ext_side = side
        if mirror_obs:
            ext_side = "left" if side == "right" else "right"
        self.extractor = _Extractor(ext_side)
        self.model = torch.jit.load(str(MODEL_PATH)).eval()
        self._warmup()

    def _warmup(self) -> None:
        dummy = torch.zeros((1, FEATURE_DIM), dtype=torch.float32)
        with torch.no_grad():
            for _ in range(_WARMUP_CALLS):
                self.model(dummy)

    def reset(self) -> None:
        """Clear velocity history between matches/points."""
        self.extractor.reset()

    def policy(self, observation: Any, reward: float) -> List[int]:
        """Return [vertical, horizontal, rotation] for this frame."""
        frame = _extract_frame(observation)
        if self.mirror_obs:
            frame = _mirror_frame(frame)
        feats = self.extractor.features_from_image(frame)
        with torch.no_grad():
            action = self.model(
                torch.from_numpy(feats).float().unsqueeze(0)
            )[0].numpy().astype(int)
        vertical = int(action[0])
        # feature 9 = rel_y, mirror-invariant; hold when aligned
        if feats[12] == 1.0 and \
                abs(float(feats[9])) * IMG_H < Y_DEADBAND_PX:
            vertical = 0
        horizontal = int(action[1])
        if self.mirror_obs:
            horizontal = _MIRROR_ACTION[horizontal]
        return [vertical, horizontal, 0]


class TeamX(_Policy):
    """Competition entry point: left paddle."""

    def __init__(self) -> None:
        super().__init__(side="left", mirror_obs=False)

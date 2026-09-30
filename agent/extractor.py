"""OpenCV detection of ball/paddles on the shared visual frame.

Port of the proven Wall_sim_v39 extractor: HSV segmentation (yellow
ball, orange/red paddles) inside a court ROI, side-aware paddle
split, frame-delta ball velocity. Thresholds live in config.py.
Part of the standalone submission bundle - imports only config and
features.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from agent import config
from agent.features import pack_features

SIDES = ("left", "right")


@dataclass(frozen=True)
class DetectionResult:
    """Pixel-space detections for one frame (None = not found)."""

    ball_xy: Optional[Tuple[float, float]]
    paddle_xy: Optional[Tuple[float, float]]
    paddle_angle_rad: float
    opponent_xy: Optional[Tuple[float, float]]
    debug: Dict[str, Any]


def _as_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Accept (C,H,W) or (H,W,C), float [0,1] or uint8; return HWC."""
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"expected 3-dim image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _largest_blob_center(
    mask: np.ndarray,
    min_area: float,
    max_area: float = 1e9,
) -> Optional[Tuple[float, float]]:
    """Return the centroid of the largest in-range blob, if any."""
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


def _largest_oriented_blob(
    mask: np.ndarray,
    min_area: float,
) -> Tuple[Optional[Tuple[float, float]], float]:
    """Return (centroid, angle_rad) of the largest blob, if any."""
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        angle_rad = 0.0
        if len(c) >= 5:
            angle_deg = cv2.minAreaRect(c)[-1]
            # minAreaRect angle is ambiguous past 45 degrees
            if angle_deg < -45:
                angle_deg += 90
            angle_rad = math.radians(float(angle_deg))
        return (cx, cy), angle_rad
    return None, 0.0


class OpenCVStateExtractor:
    """Side-aware frame-to-feature extractor.

    Side is encoded into the packed feature vector; change it via
    set_side (resets velocity history), not by mutating fields.
    """

    def __init__(self, side: str = "right") -> None:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}: {side!r}")
        self.side = side
        self.prev_ball_xy: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        """Clear frame-delta velocity history (episode reset)."""
        self.prev_ball_xy = None

    def set_side(self, side: str) -> None:
        """Switch controlled side and clear history."""
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}: {side!r}")
        self.side = side
        self.reset()

    @property
    def side_sign(self) -> float:
        return 1.0 if self.side == "right" else -1.0

    def _roi_mask(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[config.ROI_Y1:config.ROI_Y2,
             config.ROI_X1:config.ROI_X2] = 255
        return mask

    def _racket_mask(
        self, hsv: np.ndarray, roi_mask: np.ndarray
    ) -> np.ndarray:
        orange = cv2.inRange(
            hsv,
            np.array(config.RACKET_ORANGE_LO),
            np.array(config.RACKET_ORANGE_HI),
        )
        red1 = cv2.inRange(
            hsv,
            np.array(config.RACKET_RED1_LO),
            np.array(config.RACKET_RED1_HI),
        )
        red2 = cv2.inRange(
            hsv,
            np.array(config.RACKET_RED2_LO),
            np.array(config.RACKET_RED2_HI),
        )
        mask = cv2.bitwise_or(orange, cv2.bitwise_or(red1, red2))
        mask = cv2.bitwise_and(mask, roi_mask)
        return cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )

    def detect(self, img: np.ndarray) -> DetectionResult:
        """Detect ball, controlled paddle, and opponent paddle."""
        img_rgb = _as_uint8_rgb(img)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        roi = self._roi_mask(img_rgb.shape[0], img_rgb.shape[1])

        ball_mask = cv2.inRange(
            hsv, np.array(config.BALL_HSV_LO),
            np.array(config.BALL_HSV_HI),
        )
        ball_mask = cv2.bitwise_and(ball_mask, roi)
        ball_mask = cv2.morphologyEx(
            ball_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )
        ball_xy = _largest_blob_center(
            ball_mask, config.BALL_MIN_AREA, config.BALL_MAX_AREA
        )

        racket = self._racket_mask(hsv, roi)
        controlled = racket.copy()
        opponent = racket.copy()
        half = config.IMG_W // 2
        if self.side == "right":
            controlled[:, :half] = 0
            opponent[:, half:] = 0
        else:
            controlled[:, half:] = 0
            opponent[:, :half] = 0

        paddle_xy, paddle_angle = _largest_oriented_blob(
            controlled, config.PADDLE_MIN_AREA
        )
        opponent_xy, _ = _largest_oriented_blob(
            opponent, config.PADDLE_MIN_AREA
        )
        return DetectionResult(
            ball_xy=ball_xy,
            paddle_xy=paddle_xy,
            paddle_angle_rad=paddle_angle,
            opponent_xy=opponent_xy,
            debug={
                "ball_found": ball_xy is not None,
                "paddle_found": paddle_xy is not None,
                "opponent_found": opponent_xy is not None,
                "side": self.side,
            },
        )

    def features_from_image(
        self,
        img: np.ndarray,
        time_own_side_norm: float = 0.0,
        success_count_norm: float = 0.0,
    ) -> np.ndarray:
        """Detect on one frame and pack the 16-dim feature vector."""
        det = self.detect(img)
        if det.ball_xy is None:
            bx, by = 0.5 * config.IMG_W, 0.5 * config.IMG_H
            ball_visible = 0.0
        else:
            bx, by = det.ball_xy
            ball_visible = 1.0
        # velocity from frame delta; prev kept across short dropouts
        if self.prev_ball_xy is None or det.ball_xy is None:
            bvx, bvy = 0.0, 0.0
        else:
            bvx = bx - self.prev_ball_xy[0]
            bvy = by - self.prev_ball_xy[1]
            # discard occlusion-recovery jumps, not real motion
            if math.hypot(bvx, bvy) > config.MAX_BALL_DELTA_PX:
                bvx, bvy = 0.0, 0.0
        if det.ball_xy is not None:
            self.prev_ball_xy = (bx, by)

        if det.paddle_xy is None:
            fallback_x = (
                config.PADDLE_FALLBACK_X_RIGHT
                if self.side == "right"
                else config.PADDLE_FALLBACK_X_LEFT
            )
            px = config.IMG_W * fallback_x
            py = config.IMG_H * config.PADDLE_FALLBACK_Y
            paddle_visible = 0.0
        else:
            px, py = det.paddle_xy
            paddle_visible = 1.0
        # paddle angle pinned vertical; minAreaRect angle degenerate
        pang = math.pi / 2.0

        opp_y = (
            det.opponent_xy[1] if det.opponent_xy is not None else None
        )
        return pack_features(
            ball_xy=(bx, by),
            ball_vxy=(bvx, bvy),
            paddle_xy=(px, py),
            paddle_angle_rad=pang,
            side_sign=self.side_sign,
            ball_visible=ball_visible,
            paddle_visible=paddle_visible,
            time_own_side_norm=time_own_side_norm,
            success_count_norm=success_count_norm,
            opponent_y=opp_y,
        )

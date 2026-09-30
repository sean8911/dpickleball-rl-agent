"""Object tracks and event segmentation for system ID.

Runs the extractor over every recorded frame once and caches the
result as tracks.npz in the run dir; all fitting works from tracks,
never raw frames. Positions are NaN where the object was not
detected.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

from agent import config
from agent.extractor import OpenCVStateExtractor

TRACKS_FILE = "tracks.npz"
TRACKS_VERSION = 1

EVENT_WALL = "wall_bounce"
EVENT_HIT = "paddle_hit"
EVENT_TURN = "turn"


@dataclass(frozen=True)
class BallEvent:
    """A detected change of ball motion at frame `index`."""

    index: int
    kind: str
    v_in: Tuple[float, float]
    v_out: Tuple[float, float]


def _iter_chunks(run_dir: Path) -> Iterator[Dict[str, np.ndarray]]:
    chunks = sorted(run_dir.glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"no chunk_*.npz in {run_dir}")
    for chunk in chunks:
        yield dict(np.load(chunk))


def build_tracks(run_dir: Path, force: bool = False) -> dict:
    """Detect all objects per frame; cache and return the tracks."""
    cache = run_dir / TRACKS_FILE
    if cache.is_file() and not force:
        data = dict(np.load(cache))
        if int(data.get("version", -1)) == TRACKS_VERSION:
            return data
    left_ext = OpenCVStateExtractor(side="left")
    right_ext = OpenCVStateExtractor(side="right")
    ball: List[Tuple[float, float]] = []
    left: List[Tuple[float, float]] = []
    right: List[Tuple[float, float]] = []
    left_ang: List[float] = []
    right_ang: List[float] = []
    actions: List[np.ndarray] = []
    rewards: List[np.ndarray] = []
    nan2 = (np.nan, np.nan)
    for chunk in _iter_chunks(run_dir):
        for frame in chunk["frames"]:
            det_l = left_ext.detect(frame)
            det_r = right_ext.detect(frame)
            # det_l/det_r share the ball; use left's detection
            ball.append(det_l.ball_xy or nan2)
            left.append(det_l.paddle_xy or nan2)
            right.append(det_r.paddle_xy or nan2)
            left_ang.append(det_l.paddle_angle_rad)
            right_ang.append(det_r.paddle_angle_rad)
        actions.append(chunk["actions"])
        rewards.append(chunk["rewards"])
    data = {
        "version": np.int64(TRACKS_VERSION),
        "ball": np.asarray(ball, dtype=np.float32),
        "left": np.asarray(left, dtype=np.float32),
        "right": np.asarray(right, dtype=np.float32),
        "left_angle": np.asarray(left_ang, dtype=np.float32),
        "right_angle": np.asarray(right_ang, dtype=np.float32),
        "actions": np.concatenate(actions),
        "rewards": np.concatenate(rewards),
    }
    np.savez_compressed(cache, **data)
    return data


def visible_runs(ball: np.ndarray) -> List[Tuple[int, int]]:
    """Return [start, end) index pairs of contiguous detections."""
    visible = ~np.isnan(ball[:, 0])
    runs: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(visible):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(visible)))
    return runs


def _is_event(v_prev: np.ndarray, v_cur: np.ndarray) -> bool:
    s_prev = float(np.hypot(*v_prev))
    s_cur = float(np.hypot(*v_cur))
    if abs(s_cur - s_prev) > config.TRACK_EVENT_SPEED_DELTA:
        return True
    if min(s_prev, s_cur) < config.TRACK_MIN_EVENT_SPEED:
        return False
    cos = float(np.dot(v_prev, v_cur)) / (s_prev * s_cur)
    return float(np.arccos(np.clip(cos, -1.0, 1.0))) > (
        config.TRACK_EVENT_ANGLE_DELTA
    )


def _classify(
    index: int,
    v_in: np.ndarray,
    v_out: np.ndarray,
    ball_xy: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    y_lo: float,
    y_hi: float,
) -> str:
    vy_flip = (
        v_in[1] * v_out[1] < 0
        and min(abs(v_in[1]), abs(v_out[1])) > 0.2
    )
    near_wall = (
        ball_xy[1] - y_lo < config.WALL_PROX_PX
        or y_hi - ball_xy[1] < config.WALL_PROX_PX
    )
    if vy_flip and near_wall:
        return EVENT_WALL
    for paddle in (left[index], right[index]):
        if not np.isnan(paddle[0]):
            if np.hypot(*(ball_xy - paddle)) < config.CONTACT_RADIUS_PX:
                return EVENT_HIT
    return EVENT_TURN


def find_events(
    ball: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> List[BallEvent]:
    """Detect and classify motion-change events on the ball track."""
    finite_y = ball[~np.isnan(ball[:, 1]), 1]
    if len(finite_y) == 0:
        return []
    # effective wall lines from observed extremes
    y_lo = float(np.quantile(finite_y, 0.001))
    y_hi = float(np.quantile(finite_y, 0.999))
    events: List[BallEvent] = []
    for start, end in visible_runs(ball):
        if end - start < 3:
            continue
        v = np.diff(ball[start:end], axis=0)
        for t in range(1, len(v)):
            if not _is_event(v[t - 1], v[t]):
                continue
            idx = start + t
            kind = _classify(
                idx, v[t - 1], v[t], ball[idx], left, right,
                y_lo, y_hi,
            )
            events.append(
                BallEvent(
                    index=idx,
                    kind=kind,
                    v_in=(float(v[t - 1][0]), float(v[t - 1][1])),
                    v_out=(float(v[t][0]), float(v[t][1])),
                )
            )
    return events


def flight_segments(
    ball: np.ndarray,
    events: List[BallEvent],
) -> List[Tuple[int, int]]:
    """Return [start, end) spans of event-free visible flight."""
    cut = {e.index for e in events}
    segments: List[Tuple[int, int]] = []
    for start, end in visible_runs(ball):
        seg_start = start
        for i in range(start, end):
            if i in cut:
                if i - seg_start >= config.MIN_FLIGHT_FRAMES:
                    segments.append((seg_start, i))
                seg_start = i + 1
        if end - seg_start >= config.MIN_FLIGHT_FRAMES:
            segments.append((seg_start, end))
    return segments

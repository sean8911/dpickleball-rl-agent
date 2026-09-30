"""Measure the Unity slow-ball impulse from a recording.

Reuses the calibrate tracks/event pipeline: detect paddle-hit events,
then report outgoing vs incoming ball speed binned by whether the
incoming ball was slow and whether the paddle was driving forward.
A slow + forward-driven bin with a much higher outgoing speed is the
push-through signature; the max outgoing speed estimates the cap.
This is a confidence check on the seeded SLOW_IMPULSE_* params, not a
gate -- if hits are too sparse to read, keep opp1's params unchanged.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from agent import config
from agent.calibrate.tracks import (
    EVENT_HIT, build_tracks, find_events,
)


def _forward_at(
    paddle: np.ndarray, index: int, side_is_left: bool
) -> float:
    """Paddle x-velocity toward the opponent at a hit index.

    Forward is +x for the left paddle, -x for the right. Returns 0 if
    either frame is undetected (NaN).
    """
    if index < 1:
        return 0.0
    cur, prev = paddle[index], paddle[index - 1]
    if np.isnan(cur[0]) or np.isnan(prev[0]):
        return 0.0
    dvx = float(cur[0] - prev[0])
    return dvx if side_is_left else -dvx


def _nearest_side_is_left(
    ball_xy: np.ndarray, left: np.ndarray, right: np.ndarray,
    index: int,
) -> bool:
    """True if the left paddle is the nearer hitter at this index."""
    dl = (
        np.hypot(*(ball_xy - left[index]))
        if not np.isnan(left[index][0]) else np.inf
    )
    dr = (
        np.hypot(*(ball_xy - right[index]))
        if not np.isnan(right[index][0]) else np.inf
    )
    return dl <= dr


def measure(
    run_dir: Path, max_speed: float = config.MAX_BALL_DELTA_PX
) -> List[Tuple[str, int, float, float]]:
    """Return (bin_label, count, median_out, p95_out) per speed bin.

    Events whose incoming or outgoing speed exceeds max_speed are
    dropped as extractor occlusion-recovery spikes (a real ball never
    moves faster than ~HIT_SPEED_MAX); without this the max is
    meaningless. p95 (not raw max) estimates the impulse cap robustly.
    """
    data = build_tracks(run_dir)
    ball, left, right = data["ball"], data["left"], data["right"]
    events = find_events(ball, left, right)
    bins = {
        "slow_forward": [], "slow_still": [],
        "fast_forward": [], "fast_still": [],
    }
    for e in (ev for ev in events if ev.kind == EVENT_HIT):
        s_in = float(np.hypot(*e.v_in))
        s_out = float(np.hypot(*e.v_out))
        if s_in > max_speed or s_out > max_speed:
            continue  # occlusion-recovery jump, not a real hit
        is_left = _nearest_side_is_left(
            ball[e.index], left, right, e.index
        )
        paddle = left if is_left else right
        forward = _forward_at(paddle, e.index, is_left)
        slow = s_in < config.SLOW_IMPULSE_THRESHOLD
        driving = forward > 0.05
        key = (
            ("slow" if slow else "fast")
            + ("_forward" if driving else "_still")
        )
        bins[key].append(s_out)
    rows: List[Tuple[str, int, float, float]] = []
    for label, outs in bins.items():
        if outs:
            rows.append(
                (label, len(outs), float(np.median(outs)),
                 float(np.percentile(outs, 95)))
            )
        else:
            rows.append((label, 0, float("nan"), float("nan")))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the Unity slow-ball impulse signature."
    )
    parser.add_argument(
        "--run-dir", default=str(config.RECORDINGS_DIR),
        help="recording dir with chunk_*.npz frames",
    )
    parser.add_argument(
        "--max-speed", type=float, default=config.MAX_BALL_DELTA_PX,
        help="drop hits above this speed as occlusion spikes",
    )
    args = parser.parse_args()
    if not Path(args.run_dir).is_dir():
        raise FileNotFoundError(f"not a dir: {args.run_dir}")
    return args


def main() -> None:
    args = parse_args()
    rows = measure(Path(args.run_dir), args.max_speed)
    print(f"{'bin':<14}{'n':>6}{'median_out':>12}{'p95_out':>10}")
    for label, n, med, p95 in rows:
        print(f"{label:<14}{n:>6}{med:>12.2f}{p95:>10.2f}")
    print(
        "\nsignature: slow_forward median_out should exceed "
        "slow_still and approach the SLOW_IMPULSE_CAP if the "
        "mechanic is present (p95 estimates the cap)."
    )


if __name__ == "__main__":
    main()

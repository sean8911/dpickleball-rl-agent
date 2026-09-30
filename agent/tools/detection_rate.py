"""Measure extractor detection rates over a recorded run dir.

Phase 0 gate: ball + both paddles must be detected on >99% of
on-court frames. Ball misses can be legitimate (out of play), so
the report separates raw rates from longest miss streaks.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from agent.extractor import OpenCVStateExtractor


def scan_run(run_dir: Path, max_frames: int) -> dict:
    """Run the extractor over chunk frames; return rate stats."""
    chunks = sorted(run_dir.glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"no chunk_*.npz in {run_dir}")
    extractor = OpenCVStateExtractor(side="right")
    counts = {"frames": 0, "ball": 0, "paddle": 0, "opponent": 0}
    miss_streak = 0
    longest_miss = 0
    latencies: list[float] = []
    for chunk in chunks:
        frames = np.load(chunk)["frames"]
        for frame in frames:
            t0 = time.perf_counter()
            det = extractor.detect(frame)
            latencies.append(time.perf_counter() - t0)
            counts["frames"] += 1
            counts["ball"] += det.ball_xy is not None
            counts["paddle"] += det.paddle_xy is not None
            counts["opponent"] += det.opponent_xy is not None
            if det.ball_xy is None:
                miss_streak += 1
                longest_miss = max(longest_miss, miss_streak)
            else:
                miss_streak = 0
            if counts["frames"] >= max_frames:
                break
        if counts["frames"] >= max_frames:
            break
    lat = np.array(latencies)
    n = counts["frames"]
    return {
        "frames": n,
        "ball_rate": counts["ball"] / n,
        "paddle_rate": counts["paddle"] / n,
        "opponent_rate": counts["opponent"] / n,
        "longest_ball_miss_streak": longest_miss,
        "detect_ms_mean": float(lat.mean() * 1e3),
        "detect_ms_p99": float(np.percentile(lat, 99) * 1e3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extractor detection-rate report."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=100_000)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    stats = scan_run(run_dir, args.max_frames)
    print(f"[det] frames          : {stats['frames']}")
    print(f"[det] ball rate       : {stats['ball_rate']:.4f}")
    print(f"[det] paddle rate     : {stats['paddle_rate']:.4f}")
    print(f"[det] opponent rate   : {stats['opponent_rate']:.4f}")
    print(
        "[det] longest ball miss streak: "
        f"{stats['longest_ball_miss_streak']}"
    )
    print(
        f"[det] detect ms mean/p99: "
        f"{stats['detect_ms_mean']:.2f}/{stats['detect_ms_p99']:.2f}"
    )


if __name__ == "__main__":
    main()

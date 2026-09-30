"""Measure TeamX.policy latency against the 10ms competition budget.

Replays recorded frames through the deployed submission policy and
reports p50/p99 per-call latency. Gate: p99 < 5ms (half the budget),
since two policies run per step on the competition machine.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from teamX import TeamX  # noqa: E402

BUDGET_MS = 10.0
GATE_P99_MS = 5.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TeamX.policy latency check."
    )
    parser.add_argument("--run-dir", required=True,
                        help="recording dir with chunk_*.npz frames")
    parser.add_argument("--max-frames", type=int, default=5000)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    chunks = sorted(run_dir.glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"no chunk_*.npz in {run_dir}")

    team = TeamX()
    latencies = []
    for chunk in chunks:
        for frame in np.load(chunk)["frames"]:
            t0 = time.perf_counter()
            team.policy(frame, 0.0)
            latencies.append((time.perf_counter() - t0) * 1e3)
            if len(latencies) >= args.max_frames:
                break
        if len(latencies) >= args.max_frames:
            break
    lat = np.array(latencies)
    p50 = float(np.percentile(lat, 50))
    p99 = float(np.percentile(lat, 99))
    status = "PASS" if p99 < GATE_P99_MS else "FAIL"
    print(
        f"[timing] calls={len(lat)} p50={p50:.3f}ms "
        f"p99={p99:.3f}ms max={lat.max():.3f}ms "
        f"budget={BUDGET_MS}ms -> {status} (gate p99<{GATE_P99_MS}ms)"
    )


if __name__ == "__main__":
    main()

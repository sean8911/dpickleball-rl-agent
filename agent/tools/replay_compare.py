"""Validate the calibrated sim against real Unity trajectories.

Phase 1 gate. Seeds PickleballSim from measured ball segments and
compares its free-flight and wall-bounce behavior to the recording.
Free-flight mostly checks drag/geometry (drag was fit here, so it
is a consistency check); the wall-bounce report is the real test of
the assumed restitution and wall positions.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from agent import config
from agent.calibrate import tracks as T
from agent.sim import physics as P

NOOP = np.zeros(3, dtype=np.int32)
# keep free-flight tests away from paddle-capture x bands
MIDCOURT_X = (45.0, 125.0)
COMPARE_FRAMES = 30
# below this vy a bounce is jitter, not a real reflection
WALL_MIN_VY = 0.8
WALL_MIN_EVENTS = 10  # fewer than this -> result is data-limited


def _seed_velocity(seg: np.ndarray) -> np.ndarray:
    return np.diff(seg[:3], axis=0).mean(axis=0)


def _sim_free_flight(
    start_xy: np.ndarray, v0: np.ndarray, n: int
) -> np.ndarray:
    """Roll the sim ball forward n steps with no paddle action."""
    sim = P.PickleballSim()
    # park paddles at corners so they never capture the ball
    sim.left.x, sim.left.y = config.LEFT_PADDLE_X[0], 24.0
    sim.right.x, sim.right.y = config.RIGHT_PADDLE_X[1], 24.0
    sim.ball = P.BallState(
        float(start_xy[0]), float(start_xy[1]),
        float(v0[0]), float(v0[1]),
    )
    out = [(sim.ball.x, sim.ball.y)]
    for _ in range(n):
        sim.step(NOOP, NOOP)
        out.append((sim.ball.x, sim.ball.y))
    return np.asarray(out)


def free_flight_report(
    ball: np.ndarray, segments: List[Tuple[int, int]]
) -> dict:
    """Mean/p90 sim-vs-real position error over flight segments."""
    errors: List[float] = []
    used = 0
    for start, end in segments:
        seg = ball[start:end]
        if len(seg) < 8:
            continue
        xs = seg[:, 0]
        if xs.min() < MIDCOURT_X[0] or xs.max() > MIDCOURT_X[1]:
            continue
        n = min(COMPARE_FRAMES, len(seg) - 1)
        sim = _sim_free_flight(seg[0], _seed_velocity(seg), n)
        err = np.hypot(*(sim[: n + 1] - seg[: n + 1]).T)
        errors.extend(err.tolist())
        used += 1
    arr = np.array(errors) if errors else np.array([np.nan])
    return {
        "segments_used": used,
        "mean_err": float(np.nanmean(arr)),
        "p90_err": float(np.nanpercentile(arr, 90)),
        "max_err": float(np.nanmax(arr)),
    }


def wall_bounce_report(
    ball: np.ndarray, events: List[T.BallEvent]
) -> dict:
    """Compare sim reflection to real post-bounce direction/angle."""
    angle_errs: List[float] = []
    used = 0
    for e in events:
        if e.kind != T.EVENT_WALL:
            continue
        # need clear vy both sides of the bounce, else jitter
        if abs(e.v_in[1]) < WALL_MIN_VY or abs(e.v_out[1]) < WALL_MIN_VY:
            continue
        v_in = np.array(e.v_in)
        start_xy = ball[e.index] - v_in  # one frame pre-bounce
        sim = _sim_free_flight(start_xy, v_in, 2)
        sim_v_out = sim[2] - sim[1]
        if np.hypot(*sim_v_out) < 1e-3:
            continue
        real_out = np.array(e.v_out)
        cos = float(
            np.dot(sim_v_out, real_out)
            / (np.hypot(*sim_v_out) * np.hypot(*real_out))
        )
        angle_errs.append(
            float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        )
        used += 1
    if not angle_errs:
        return {"events_used": 0}
    arr = np.array(angle_errs)
    return {
        "events_used": used,
        "mean_angle_err_deg": float(arr.mean()),
        "p90_angle_err_deg": float(np.percentile(arr, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sim-vs-Unity replay comparison (Phase 1 gate)."
    )
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    d = T.build_tracks(run_dir)
    events = T.find_events(d["ball"], d["left"], d["right"])
    segments = T.flight_segments(d["ball"], events)
    ff = free_flight_report(d["ball"], segments)
    wb = wall_bounce_report(d["ball"], events)

    print("[replay] free flight (gate: mean < 2.0 px)")
    print(
        f"  segments={ff['segments_used']} "
        f"mean={ff['mean_err']:.3f} p90={ff['p90_err']:.3f} "
        f"max={ff['max_err']:.3f}"
    )
    print("[replay] wall bounce (gate: mean angle < 5 deg)")
    if wb["events_used"]:
        note = (
            "  [DATA-LIMITED: too few clean bounces to trust]"
            if wb["events_used"] < WALL_MIN_EVENTS else ""
        )
        print(
            f"  events={wb['events_used']} "
            f"mean={wb['mean_angle_err_deg']:.2f} "
            f"p90={wb['p90_angle_err_deg']:.2f}{note}"
        )
    else:
        print("  no clean wall-bounce events found")


if __name__ == "__main__":
    main()

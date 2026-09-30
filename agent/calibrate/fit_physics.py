"""Fit sim physics parameters from recorded Unity rollouts.

System identification over cached tracks (calibrate/tracks.py):
court geometry, ball drag/gravity, wall restitution, paddle
kinematics per action branch, and the serve model. Prints a
config.py snippet plus per-fit residuals that size the
domain-randomization ranges used in curriculum level 5.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from agent import config
from agent.calibrate import tracks as T

BRANCH_NAMES = ("vertical", "horizontal", "rotation")
BRANCH_VALUE_NAMES = (
    ("none", "up", "down"),
    ("none", "right", "left"),
    ("none", "ccw", "cw"),
)


def _finite(arr: np.ndarray) -> np.ndarray:
    return arr[~np.isnan(arr[:, 0])]


def fit_geometry(d: dict) -> dict:
    """Court and paddle travel bounds from observed extremes."""
    ball = _finite(d["ball"])
    left = _finite(d["left"])
    right = _finite(d["right"])
    return {
        "ball_x": (float(ball[:, 0].min()), float(ball[:, 0].max())),
        "ball_y": (float(ball[:, 1].min()), float(ball[:, 1].max())),
        "left_x": (float(left[:, 0].min()), float(left[:, 0].max())),
        "right_x": (float(right[:, 0].min()),
                    float(right[:, 0].max())),
        "paddle_y": (
            float(min(left[:, 1].min(), right[:, 1].min())),
            float(max(left[:, 1].max(), right[:, 1].max())),
        ),
    }


def fit_ball_dynamics(
    ball: np.ndarray, segments: List[Tuple[int, int]]
) -> dict:
    """Per-step speed-decay ratio and vertical acceleration."""
    ratios: List[float] = []
    vy_accel: List[float] = []
    speeds: List[float] = []
    for start, end in segments:
        v = np.diff(ball[start:end], axis=0)
        sp = np.hypot(v[:, 0], v[:, 1])
        if len(sp) < 4 or sp.min() < config.TRACK_MIN_EVENT_SPEED:
            continue
        ratios.extend((sp[1:] / sp[:-1]).tolist())
        vy_accel.extend(np.diff(v[:, 1]).tolist())
        speeds.extend(sp.tolist())
    r = np.array(ratios)
    r = r[(r > 0.5) & (r < 1.5)]  # drop tracking-jitter outliers
    sp = np.array(speeds)
    return {
        "drag_mean": float(r.mean()),
        "drag_median": float(np.median(r)),
        "drag_std": float(r.std()),
        "vy_accel_median": float(np.median(vy_accel)),
        "speed_median": float(np.median(sp)),
        "speed_p95": float(np.percentile(sp, 95)),
        "speed_max": float(sp.max()),
        "n_samples": int(len(r)),
    }


def fit_wall_restitution(events: List[T.BallEvent]) -> dict:
    """Restitution |vy_out|/|vy_in| at top/bottom wall bounces."""
    rest: List[float] = []
    for e in events:
        if e.kind != T.EVENT_WALL:
            continue
        if abs(e.v_in[1]) < 0.2:
            continue
        rest.append(abs(e.v_out[1]) / abs(e.v_in[1]))
    rest_arr = np.array(rest)
    if len(rest_arr) == 0:
        return {"n": 0}
    return {
        "restitution_mean": float(rest_arr.mean()),
        "restitution_std": float(rest_arr.std()),
        "n": int(len(rest_arr)),
    }


def _branch_speed(
    pos: np.ndarray,
    actions_side: np.ndarray,
    branch: int,
    axis: int,
    lo: float,
    hi: float,
) -> dict:
    """Mean signed displacement per branch value, bound-filtered.

    Excludes steps where the paddle sits within a margin of its
    travel bound on `axis`, so clamped steps do not depress the
    measured free-travel speed.
    """
    deltas = np.diff(pos, axis=0)
    act = actions_side[1:]
    coord = pos[:-1, axis]
    others = [b for b in range(config.ACTION_BRANCHES) if b != branch]
    pure = (act[:, others] == 0).all(axis=1)
    free = (coord > lo + config.PADDLE_BOUND_MARGIN_PX) & (
        coord < hi - config.PADDLE_BOUND_MARGIN_PX
    )
    valid = ~np.isnan(deltas).any(axis=1) & pure & free
    out = {}
    for value in range(config.ACTION_CHOICES):
        rows = valid & (act[:, branch] == value)
        out[value] = (
            float(deltas[rows, axis].mean()) if rows.sum() >= 10
            else None
        )
    return out


def _branch_angle_speed(
    angle: np.ndarray, actions_side: np.ndarray
) -> dict:
    """Mean angle delta (rad) per rotation-branch value.

    Angle from minAreaRect wraps at +/-pi/2, so unwrap the small
    per-step difference before averaging.
    """
    dang = np.diff(angle)
    dang = (dang + np.pi / 2) % np.pi - np.pi / 2
    act = actions_side[1:]
    others = (act[:, 0] == 0) & (act[:, 1] == 0)
    valid = ~np.isnan(dang) & others
    out = {}
    for value in range(config.ACTION_CHOICES):
        rows = valid & (act[:, 2] == value)
        out[value] = (
            float(dang[rows].mean()) if rows.sum() >= 10 else None
        )
    return out


def fit_paddle_kinematics(d: dict, geom: dict) -> dict:
    """Per-side vertical/horizontal px-per-step and rotation."""
    actions = d["actions"]
    result = {}
    sides = (
        ("left", d["left"], d["left_angle"], actions[:, 0],
         geom["left_x"]),
        ("right", d["right"], d["right_angle"], actions[:, 1],
         geom["right_x"]),
    )
    for name, pos, ang, acts, x_bounds in sides:
        vert = _branch_speed(
            pos, acts, branch=0, axis=1,
            lo=geom["paddle_y"][0], hi=geom["paddle_y"][1],
        )
        horiz = _branch_speed(
            pos, acts, branch=1, axis=0,
            lo=x_bounds[0], hi=x_bounds[1],
        )
        rot = _branch_angle_speed(ang, acts)
        result[name] = {
            "vertical": vert,
            "horizontal": horiz,
            "rotation": rot,
        }
    return result


def fit_serves(d: dict) -> dict:
    """Ball speed at the first flight frames after each point gap."""
    ball = d["ball"]
    speeds: List[float] = []
    for start, end in T.visible_runs(ball):
        if end - start < config.MIN_FLIGHT_FRAMES:
            continue
        v = np.diff(ball[start:start + 4], axis=0)
        sp = float(np.hypot(v[:, 0], v[:, 1]).mean())
        if sp < config.TRACK_MIN_EVENT_SPEED:
            continue
        speeds.append(sp)
    sp = np.array(speeds)
    return {
        "serve_speed_median": float(np.median(sp)),
        "serve_speed_p10": float(np.percentile(sp, 10)),
        "serve_speed_p90": float(np.percentile(sp, 90)),
        "n": int(len(sp)),
    }


def _fmt_branch(name: str, res: dict, unit: str) -> str:
    parts = []
    names = BRANCH_VALUE_NAMES[BRANCH_NAMES.index(name)]
    for value in range(config.ACTION_CHOICES):
        v = res[value]
        parts.append(
            f"{names[value]}="
            + ("n/a" if v is None else f"{v:+.3f}{unit}")
        )
    return "  " + name.ljust(11) + " ".join(parts)


def report(run_dir: Path, force: bool) -> None:
    """Run every fit and print results plus residual ranges."""
    d = T.build_tracks(run_dir, force=force)
    geom = fit_geometry(d)
    events = T.find_events(d["ball"], d["left"], d["right"])
    segments = T.flight_segments(d["ball"], events)
    ball_dyn = fit_ball_dynamics(d["ball"], segments)
    wall = fit_wall_restitution(events)
    kin = fit_paddle_kinematics(d, geom)
    serves = fit_serves(d)

    print("\n[fit] geometry (image px)")
    for k, (lo, hi) in geom.items():
        print(f"  {k:<10} [{lo:.1f}, {hi:.1f}]")

    print(f"\n[fit] ball dynamics  (n={ball_dyn['n_samples']})")
    print(
        f"  drag/step  mean={ball_dyn['drag_mean']:.4f} "
        f"median={ball_dyn['drag_median']:.4f} "
        f"std={ball_dyn['drag_std']:.4f}"
    )
    print(
        f"  vy accel   median={ball_dyn['vy_accel_median']:+.4f} "
        f"(>0 would mean gravity)"
    )
    print(
        f"  speed px/s median={ball_dyn['speed_median']:.2f} "
        f"p95={ball_dyn['speed_p95']:.2f} "
        f"max={ball_dyn['speed_max']:.2f}"
    )

    if wall.get("n", 0):
        print(
            f"\n[fit] wall bounce  (n={wall['n']})\n"
            f"  restitution mean={wall['restitution_mean']:.3f} "
            f"std={wall['restitution_std']:.3f}"
        )
    else:
        print("\n[fit] wall bounce  (no samples)")

    print("\n[fit] paddle kinematics (bound-filtered)")
    for side in ("left", "right"):
        print(f" {side.upper()}")
        print(_fmt_branch("vertical", kin[side]["vertical"], "px"))
        print(_fmt_branch("horizontal", kin[side]["horizontal"],
                          "px"))
        print(_fmt_branch("rotation", kin[side]["rotation"], "rad"))

    print(f"\n[fit] serves  (n={serves['n']})")
    print(
        f"  speed px/step median={serves['serve_speed_median']:.2f} "
        f"p10={serves['serve_speed_p10']:.2f} "
        f"p90={serves['serve_speed_p90']:.2f}"
    )

    print(
        f"\n[fit] events: {len(events)} total, "
        f"flight segments: {len(segments)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="System ID from recorded Unity rollouts."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--force", action="store_true",
                        help="rebuild the tracks cache")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    report(run_dir, args.force)


if __name__ == "__main__":
    main()

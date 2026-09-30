"""Sanity-check the sim env with a feature-only heuristic agent.

If a competent hand-coded agent cannot rack up returns, the env or
physics are broken and PPO will not learn either. The agent reads
only the 16-dim observation (same contract as deployment), so it
doubles as the Phase 4 scripted baseline opponent.
"""

import argparse

import numpy as np

from agent import config
from agent.envs.sim_env import SimPickleballEnv


def heuristic_action(obs: np.ndarray) -> np.ndarray:
    """Track the ball in y and strike forward when it is close.

    Feature indices (see features.FEATURE_NAMES): 1 ball_y,
    5 paddle_y, 8 rel_x = (ball-paddle)/W, 14 side_sign.
    """
    ball_y, paddle_y = obs[1], obs[5]
    rel_x, side_sign = obs[8], obs[14]
    dy = ball_y - paddle_y
    if dy > 0.02:
        vert = 2  # ball below -> down
    elif dy < -0.02:
        vert = 1
    else:
        vert = 0
    # forward = toward net: left (sign<0) -> +x (1); right -> -x (2)
    forward = 1 if side_sign < 0 else 2
    retreat = 2 if side_sign < 0 else 1
    # strike when the ball is close in x, else return home
    horiz = forward if abs(rel_x) < 0.10 else retreat
    return np.array([vert, horiz, 0], dtype=np.int32)


def run(level: int, episodes: int, seed: int) -> dict:
    """Return success rate and mean returns over episodes."""
    env = SimPickleballEnv(level=level, seed=seed)
    successes = 0
    total_returns = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        info = {"success": False, "returns": 0}
        while not done:
            obs, _, term, trunc, info = env.step(heuristic_action(obs))
            done = term or trunc
        successes += int(info["success"])
        total_returns += info["returns"]
    return {
        "success_rate": successes / episodes,
        "mean_returns": total_returns / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heuristic-agent sanity check on the sim env."
    )
    parser.add_argument("--levels", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5])
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for level in args.levels:
        if level < 1 or level > config.MAX_LEVEL:
            raise ValueError(f"level out of range: {level}")
        stats = run(level, args.episodes, args.seed)
        print(
            f"[sanity] level {level}: "
            f"success={stats['success_rate']:.2f} "
            f"mean_returns={stats['mean_returns']:.2f}"
        )


if __name__ == "__main__":
    main()

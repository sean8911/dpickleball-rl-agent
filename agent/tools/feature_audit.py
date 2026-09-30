"""Compare sim-trained feature distributions to real Unity ones.

Sim-to-real check: the policy trained on features built from sim
state; at deployment it sees features built by the extractor from
real frames. If a feature's range/mean diverges between the two,
the policy is seeing out-of-distribution input and will misbehave.
Uses a Phase 0 recording (no live Unity needed).
"""

import argparse
from pathlib import Path

import numpy as np

from agent.envs.sim_env import SimPickleballEnv
from agent.extractor import OpenCVStateExtractor
from agent.features import FEATURE_NAMES


def collect_sim(level: int, episodes: int) -> np.ndarray:
    """Gather feature vectors from random sim play."""
    env = SimPickleballEnv(level=level, seed=0)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            rows.append(obs)
            obs, _, term, trunc, _ = env.step(
                env.action_space.sample()
            )
            done = term or trunc
    return np.asarray(rows)


def collect_unity(run_dir: Path, max_frames: int) -> np.ndarray:
    """Gather features from the extractor over recorded frames."""
    chunks = sorted(run_dir.glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"no chunk_*.npz in {run_dir}")
    ext = OpenCVStateExtractor(side="right")
    rows = []
    for chunk in chunks:
        for frame in np.load(chunk)["frames"]:
            rows.append(ext.features_from_image(frame))
            if len(rows) >= max_frames:
                return np.asarray(rows)
    return np.asarray(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sim-vs-Unity feature distribution audit."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=20000)
    parser.add_argument("--level", type=int, default=3)
    args = parser.parse_args()
    sim = collect_sim(args.level, args.episodes)
    unity = collect_unity(Path(args.run_dir), args.max_frames)
    print(f"sim rows={len(sim)}  unity rows={len(unity)}")
    print(f"{'feature':<16}{'sim mean':>10}{'sim rng':>16}"
          f"{'unity mean':>12}{'unity rng':>16}")
    for i, name in enumerate(FEATURE_NAMES):
        s, u = sim[:, i], unity[:, i]
        print(
            f"{name:<16}{s.mean():>10.2f}"
            f"{f'[{s.min():.2f},{s.max():.2f}]':>16}"
            f"{u.mean():>12.2f}"
            f"{f'[{u.min():.2f},{u.max():.2f}]':>16}"
        )


if __name__ == "__main__":
    main()

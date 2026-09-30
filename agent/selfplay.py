"""Phase 4+ entry point: competitive self-play (win-rate reward).

Trains against a pool of frozen snapshots with a non-saturating
points reward; eval is win rate vs a fixed reference, so only a model
that beats the current deliverable is kept.
"""

import argparse
from pathlib import Path
from typing import Callable, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from agent import config
from agent.checkpoints import CheckpointManager, export_torchscript
from agent.envs.sim_env import SimPickleballEnv
from agent.opponents import SnapshotPool
from agent.sim import physics as P


def make_env(seed: int, paths: list) -> Callable[[], SimPickleballEnv]:
    """Thunk: competitive sim env vs the snapshot pool."""
    def _init() -> SimPickleballEnv:
        return SimPickleballEnv(
            level=config.MAX_LEVEL, seed=seed,
            opponent_paths=paths, competitive=True,
        )

    return _init


def win_rate(
    model: PPO, reference_path: str, steps: int, seed: int,
    enable_rotation: bool = False, enable_slow_impulse: bool = False,
) -> Tuple[float, float, float]:
    """Learner vs a fixed reference; return (overall, left, right).

    Win rate = points_for / (points_for + points_against), measured
    per side (the controlled side alternates per episode).
    """
    env = SimPickleballEnv(
        level=config.MAX_LEVEL, seed=seed,
        opponent_paths=[reference_path], competitive=True,
        enable_rotation=enable_rotation,
        enable_slow_impulse=enable_slow_impulse,
    )
    pf = {P.LEFT: 0, P.RIGHT: 0}
    pa = {P.LEFT: 0, P.RIGHT: 0}
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        side = info["agent_side"]
        if reward > 0:
            pf[side] += 1
        elif reward < 0:
            pa[side] += 1
        if term or trunc:
            obs, _ = env.reset()

    def rate(side: str) -> float:
        total = pf[side] + pa[side]
        return pf[side] / total if total else 0.5

    total = sum(pf.values()) + sum(pa.values())
    overall = sum(pf.values()) / total if total else 0.5
    return overall, rate(P.LEFT), rate(P.RIGHT)


def selfplay(args: argparse.Namespace) -> None:
    """Competitive pooled self-play from the warm-start."""
    out_dir = Path(args.out_dir)
    pool = SnapshotPool(out_dir / "pool", max_size=args.pool_size)
    pool.seed(args.reference)  # fixed floor opponent in the pool too
    vec = SubprocVecEnv([
        make_env(args.seed + i, pool.paths())
        for i in range(args.num_envs)
    ])
    model = PPO.load(args.pretrained, env=vec, device=args.device)
    manager = CheckpointManager(out_dir)
    total = 0
    while total < args.total_steps:
        model.learn(args.interval, reset_num_timesteps=False,
                    progress_bar=False)
        total += args.interval
        pool.add(model)
        vec.env_method("set_opponent_paths", pool.paths())
        overall, left, right = win_rate(
            model, args.reference, args.eval_steps, args.seed + 9000
        )
        kept = manager.consider(
            model, score=overall, level=config.MAX_LEVEL,
            overall_rate=overall, left_rate=left, right_rate=right,
            total_steps=total,
        )
        model.save(out_dir / "latest.zip")
        export_torchscript(model, out_dir / "latest_policy.pt")
        print(
            f"[selfplay] steps={total} winrate_vs_ref={overall:.2f} "
            f"L={left:.2f} R={right:.2f} pool={len(pool.paths())} "
            f"kept={kept}"
        )
    manager.promote_best()
    vec.close()
    print("[selfplay] done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Competitive pooled self-play in the sim."
    )
    default_ref = str(
        config.PACKAGE_DIR / "checkpoints_selfplay"
        / "best_1_policy.pt"
    )
    parser.add_argument(
        "--pretrained",
        default=str(
            config.PACKAGE_DIR / "checkpoints_selfplay" / "best_1.zip"
        ),
    )
    parser.add_argument(
        "--reference", default=default_ref,
        help="fixed .pt yardstick (current deliverable / fallback)",
    )
    parser.add_argument("--num-envs", type=int,
                        default=config.NUM_ENVS)
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--interval", type=int, default=250_000)
    parser.add_argument("--eval-steps", type=int, default=6000)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out-dir",
        default=str(config.PACKAGE_DIR / "checkpoints_compete"),
    )
    args = parser.parse_args()
    for path in (args.pretrained, args.reference):
        if not Path(path).is_file():
            raise FileNotFoundError(f"not found: {path}")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    return args


def main() -> None:
    selfplay(parse_args())


if __name__ == "__main__":
    main()

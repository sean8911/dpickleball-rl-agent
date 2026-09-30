"""Camp+angle hybrid: beat the camper AND keep rally skill.

Resumes the rotation model and trains it (rotation + slow-impulse ON)
against a mix of the net-hugger camper and a pinned rallier pool.
Keeps a checkpoint only if it beats the camper, wins vs the rallier,
and retains the sim level-5 receive task.
"""

import argparse
from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from agent import config
from agent.checkpoints import CheckpointManager, export_torchscript
from agent.envs.sim_env import SimPickleballEnv
from agent.opponents import SnapshotPool
from agent.selfplay import win_rate
from agent.train_anticamper import curriculum_lag, score_rate
from agent.train_sim import evaluate as sim_evaluate


def make_env(
    seed: int, paths: list, camper_prob: float
) -> Callable[[], SimPickleballEnv]:
    """Thunk: competitive env mixing the net-hugger and the pool."""
    def _init() -> SimPickleballEnv:
        return SimPickleballEnv(
            level=config.MAX_LEVEL, seed=seed, competitive=True,
            scripted_opponent="net_hugger", depth_shaping=True,
            opponent_paths=paths, scripted_opponent_prob=camper_prob,
            enable_rotation=True, enable_slow_impulse=True,
        )

    return _init


def train(args: argparse.Namespace) -> None:
    """Train the camp+angle hybrid from the rotation warm-start."""
    out_dir = Path(args.out_dir)
    pool = SnapshotPool(out_dir / "pool", max_size=args.pool_size)
    pool.seed(args.reference)  # 800k rallier pinned as the floor
    vec = SubprocVecEnv([
        make_env(args.seed + i, pool.paths(), args.camper_prob)
        for i in range(args.num_envs)
    ])
    model = PPO.load(args.pretrained, env=vec, device=args.device)
    manager = CheckpointManager(out_dir)
    total = 0
    while total < args.total_steps:
        lag = curriculum_lag(
            total, args.total_steps, args.start_lag, args.ramp_frac
        )
        vec.env_method("set_hugger_lag", lag)
        model.learn(args.interval, reset_num_timesteps=False,
                    progress_bar=False)
        total += args.interval
        pool.add(model)
        vec.env_method("set_opponent_paths", pool.paths())
        # the two skills that must coexist, plus receive retention
        tgt, camp_l, camp_r = score_rate(
            model, args.eval_steps, args.seed + 9000, lag=0,
            enable_rotation=True, enable_slow_impulse=True,
        )
        wvr, win_l, win_r = win_rate(
            model, args.reference, args.eval_steps, args.seed + 9000,
            enable_rotation=True, enable_slow_impulse=True,
        )
        sim_overall, _, _ = sim_evaluate(
            model, config.MAX_LEVEL, config.EVAL_TRIALS,
            args.seed + 9000,
        )
        # keep only a model good at BOTH skills and the receive task
        kept = manager.consider(
            model, score=tgt + wvr + sim_overall,
            level=config.MAX_LEVEL, overall_rate=min(tgt, wvr),
            left_rate=win_l, right_rate=win_r, total_steps=total,
        )
        model.save(out_dir / "latest.zip")
        export_torchscript(model, out_dir / "latest_policy.pt")
        print(
            f"[hybrid] steps={total} lag={lag} camp_tgt0={tgt:.2f} "
            f"(L={camp_l:.2f} R={camp_r:.2f}) winVs800k={wvr:.2f} "
            f"(L={win_l:.2f} R={win_r:.2f}) simL5={sim_overall:.2f} "
            f"kept={kept}"
        )
    manager.promote_best()
    vec.close()
    print("[hybrid] done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Camp+angle hybrid: beat campers and ralliers."
    )
    parser.add_argument(
        "--pretrained",
        default=str(
            config.PACKAGE_DIR / "checkpoints_rot" / "best_1.zip"
        ),
        help="warm-start (default: the rotation model)",
    )
    parser.add_argument(
        "--reference",
        default=str(
            config.PACKAGE_DIR / "checkpoints_selfplay"
            / "best_1_policy.pt"
        ),
        help="rallier yardstick to not lose to (the 800k)",
    )
    parser.add_argument("--num-envs", type=int,
                        default=config.NUM_ENVS)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--interval", type=int, default=100_000)
    parser.add_argument(
        "--camper-prob", type=float, default=0.5,
        help="fraction of episodes vs the net-hugger (else pool)",
    )
    parser.add_argument(
        "--start-lag", type=int, default=25,
        help="net-hugger reaction lag at the start (beatable)",
    )
    parser.add_argument(
        "--ramp-frac", type=float, default=0.7,
        help="fraction of training over which lag ramps to 0",
    )
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out-dir",
        default=str(config.PACKAGE_DIR / "checkpoints_hybrid"),
    )
    args = parser.parse_args()
    for path in (args.pretrained, args.reference):
        if not Path(path).is_file():
            raise FileNotFoundError(f"not found: {path}")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    if not 0.0 <= args.camper_prob <= 1.0:
        raise ValueError("--camper-prob must be in [0, 1]")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

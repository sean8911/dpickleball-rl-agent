"""Phase 2 entry point: PPO curriculum pretraining in the sim.

Trains one shared side-aware policy across the 5-level curriculum
with parallel envs (SubprocVecEnv). After each train interval it
evaluates per side, offers the model to the top-K checkpoint
manager (with the anti-asymmetry floor), and advances the level
when the gate passes overall and on both sides.
"""

import argparse
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from agent import config
from agent.checkpoints import CheckpointManager
from agent.curriculum import passes_gate
from agent.envs.sim_env import SimPickleballEnv
from agent.sim import physics as P


def make_env(level: int, seed: int) -> Callable[[], SimPickleballEnv]:
    """Return a thunk constructing one env (picklable for subproc)."""
    def _init() -> SimPickleballEnv:
        return SimPickleballEnv(level=level, seed=seed)

    return _init


def evaluate(
    model: PPO, level: int, trials: int, seed: int
) -> Tuple[float, float, float]:
    """Deterministic per-side success rates over eval trials.

    Returns (overall_rate, left_rate, right_rate).
    """
    per_side = {P.LEFT: [], P.RIGHT: []}
    for side in (P.LEFT, P.RIGHT):
        env = SimPickleballEnv(
            level=level, seed=seed, fixed_side=side
        )
        for t in range(trials // 2):
            obs, _ = env.reset(seed=seed + t)
            done = False
            info = {"success": False}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            per_side[side].append(int(info["success"]))
    left = float(np.mean(per_side[P.LEFT]))
    right = float(np.mean(per_side[P.RIGHT]))
    overall = float(np.mean(per_side[P.LEFT] + per_side[P.RIGHT]))
    return overall, left, right


def build_model(vec: SubprocVecEnv, seed: int, device: str) -> PPO:
    """Construct a PPO model with the configured hyperparameters."""
    return PPO(
        "MlpPolicy",
        vec,
        learning_rate=config.PPO_LR,
        n_steps=config.PPO_N_STEPS,
        batch_size=config.PPO_BATCH,
        n_epochs=config.PPO_EPOCHS,
        gamma=config.PPO_GAMMA,
        gae_lambda=config.PPO_GAE_LAMBDA,
        clip_range=config.PPO_CLIP,
        ent_coef=config.PPO_ENT_COEF,
        vf_coef=config.PPO_VF_COEF,
        max_grad_norm=config.PPO_MAX_GRAD_NORM,
        policy_kwargs={"net_arch": list(config.PPO_NET_ARCH)},
        seed=seed,
        device=device,
        verbose=0,
    )


def train(args: argparse.Namespace) -> None:
    """Run the curriculum training loop to MAX_LEVEL or step cap."""
    level = args.start_level
    vec = SubprocVecEnv([
        make_env(level, args.seed + i) for i in range(args.num_envs)
    ])
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume zip not found: {resume_path}")
        model = PPO.load(str(resume_path), env=vec, device=args.device)
        print(f"[train] resumed from {resume_path} at level {level}")
    else:
        model = build_model(vec, args.seed, args.device)
    manager = CheckpointManager(Path(args.out_dir))
    total = 0
    while total < args.total_steps and level <= config.MAX_LEVEL:
        model.learn(
            args.train_interval, reset_num_timesteps=False,
            progress_bar=False,
        )
        total += args.train_interval
        overall, left, right = evaluate(
            model, level, config.EVAL_TRIALS, args.seed + 9000
        )
        kept = manager.consider(
            model, score=level + overall, level=level,
            overall_rate=overall, left_rate=left,
            right_rate=right, total_steps=total,
        )
        print(
            f"[train] steps={total} level={level} "
            f"overall={overall:.2f} L={left:.2f} R={right:.2f} "
            f"kept={kept}"
        )
        # keep refining at the final level until the step cap
        if passes_gate(overall, left, right) and level < config.MAX_LEVEL:
            level += 1
            vec.env_method("set_level", level)
            print(f"[train] -> advancing to level {level}")
    manager.promote_best()
    vec.close()
    print("[train] done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO sim curriculum pretraining."
    )
    parser.add_argument("--num-envs", type=int,
                        default=config.NUM_ENVS)
    parser.add_argument("--total-steps", type=int,
                        default=config.MAX_TOTAL_STEPS)
    parser.add_argument("--train-interval", type=int,
                        default=config.TRAIN_INTERVAL)
    parser.add_argument("--start-level", type=int, default=1)
    parser.add_argument("--resume", default=None,
                        help="SB3 zip to resume from")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir", default=str(config.CHECKPOINT_DIR)
    )
    args = parser.parse_args()
    if not 1 <= args.start_level <= config.MAX_LEVEL:
        raise ValueError("--start-level out of range")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

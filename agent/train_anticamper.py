"""Anti-camper fine-tune: learn to score past a net-hugger in sim.

Resumes the self-play deliverable and trains it against a scripted
net-hugger (opponent1's blocker) with a dense advance reward to
bootstrap the losing-start gradient. Validate any candidate vs
opponent1 in Unity before adopting.
"""

import argparse
from pathlib import Path
from typing import Callable, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from agent import config
from agent.checkpoints import CheckpointManager, export_torchscript
from agent.envs.sim_env import SimPickleballEnv
from agent.sim import physics as P
from agent.train_sim import evaluate as sim_evaluate


def make_env(
    seed: int, enable_rotation: bool, enable_slow_impulse: bool
) -> Callable[[], SimPickleballEnv]:
    """Thunk: competitive sim env vs the scripted net-hugger."""
    def _init() -> SimPickleballEnv:
        return SimPickleballEnv(
            level=config.MAX_LEVEL, seed=seed, competitive=True,
            scripted_opponent="net_hugger", depth_shaping=True,
            enable_rotation=enable_rotation,
            enable_slow_impulse=enable_slow_impulse,
        )

    return _init


def score_rate(
    model: PPO, steps: int, seed: int, lag: int = 0,
    enable_rotation: bool = False, enable_slow_impulse: bool = False,
) -> Tuple[float, float, float]:
    """Score rate past the net-hugger; return (overall, left, right).

    Eval runs with depth shaping OFF so the reward is pure points
    (+1 we score past the camper, -1 it scores on us). lag=0 is the
    perfect camper (opp1-like difficulty); a higher lag is the
    easier curriculum opponent.
    """
    env = SimPickleballEnv(
        level=config.MAX_LEVEL, seed=seed, competitive=True,
        scripted_opponent="net_hugger", depth_shaping=False,
        hugger_lag=lag, enable_rotation=enable_rotation,
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
        return pf[side] / total if total else 0.0

    total = sum(pf.values()) + sum(pa.values())
    overall = sum(pf.values()) / total if total else 0.0
    return overall, rate(P.LEFT), rate(P.RIGHT)


def curriculum_lag(
    total: int, total_steps: int, start_lag: int, ramp_frac: float
) -> int:
    """Net-hugger reaction lag for the current step.

    Ramps start_lag -> 0 linearly over the first ramp_frac of
    training, then holds at 0 (the perfect, opp1-like camper) so the
    agent first learns the exploit on a beatable opponent, then
    sharpens it at full difficulty.
    """
    if ramp_frac <= 0.0:
        return 0
    frac = total / (ramp_frac * total_steps)
    return max(0, round(start_lag * (1.0 - frac)))


def train(args: argparse.Namespace) -> None:
    """Fine-tune the warm-start against the net-hugger curriculum."""
    out_dir = Path(args.out_dir)
    vec = SubprocVecEnv([
        make_env(args.seed + i, args.enable_rotation,
                 args.enable_slow_impulse)
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
        # checkpoint on target + sim retention (keep the receive task)
        cur, _, _ = score_rate(
            model, args.eval_steps, args.seed + 9000, lag=lag,
            enable_rotation=args.enable_rotation,
            enable_slow_impulse=args.enable_slow_impulse,
        )
        tgt, left, right = score_rate(
            model, args.eval_steps, args.seed + 9000, lag=0,
            enable_rotation=args.enable_rotation,
            enable_slow_impulse=args.enable_slow_impulse,
        )
        sim_overall, _, _ = sim_evaluate(
            model, config.MAX_LEVEL, config.EVAL_TRIALS,
            args.seed + 9000,
        )
        kept = manager.consider(
            model, score=tgt + sim_overall, level=config.MAX_LEVEL,
            overall_rate=tgt, left_rate=left, right_rate=right,
            total_steps=total,
        )
        model.save(out_dir / "latest.zip")
        export_torchscript(model, out_dir / "latest_policy.pt")
        print(
            f"[anticamper] steps={total} lag={lag} cur={cur:.2f} "
            f"tgt0={tgt:.2f} L={left:.2f} R={right:.2f} "
            f"simL5={sim_overall:.2f} kept={kept}"
        )
    manager.promote_best()
    vec.close()
    print("[anticamper] done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune to score past a net-hugger in the sim."
    )
    parser.add_argument(
        "--pretrained",
        default=str(
            config.PACKAGE_DIR / "checkpoints_selfplay" / "best_1.zip"
        ),
    )
    parser.add_argument("--num-envs", type=int,
                        default=config.NUM_ENVS)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--interval", type=int, default=100_000)
    parser.add_argument(
        "--start-lag", type=int, default=25,
        help="net-hugger reaction lag at the start (beatable)",
    )
    parser.add_argument(
        "--ramp-frac", type=float, default=0.7,
        help="fraction of training over which lag ramps to 0",
    )
    parser.add_argument(
        "--enable-rotation", action="store_true",
        help="open-loop rotation experiment: angled shots vs camper",
    )
    parser.add_argument(
        "--enable-slow-impulse", action="store_true",
        help="slow-ball push-through: hard-drive a slow ball",
    )
    parser.add_argument("--eval-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out-dir",
        default=str(config.PACKAGE_DIR / "checkpoints_anticamper"),
    )
    args = parser.parse_args()
    if not Path(args.pretrained).is_file():
        raise FileNotFoundError(f"not found: {args.pretrained}")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

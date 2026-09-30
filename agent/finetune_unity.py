"""Phase 3 entry point: fine-tune the sim-pretrained policy in Unity.

Resumes the Phase 2 checkpoint and continues PPO against the live
Unity build, with the frozen pretrained policy as the opponent.
Learning rate and entropy are lowered for gentle adaptation; the
rollout size is inherited from the checkpoint. Eval reports points
won vs the frozen opponent per side and checks that sim level-5
skill is retained (catastrophic-forgetting guard).
"""

import argparse
from pathlib import Path
from typing import Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agent import config
from agent.checkpoints import (
    CheckpointManager,
    export_torchscript,
)
from agent.envs.unity_env import LEFT, RIGHT, UnityPickleballEnv
from agent.train_sim import evaluate as sim_evaluate

FINETUNE_LR = 5e-5
FINETUNE_ENT = 0.005


def eval_vs_opponent(
    model: PPO, env: UnityPickleballEnv, steps: int
) -> Tuple[float, float, float]:
    """Run the learner deterministically; net points per side.

    The env opponent is the frozen pretrained policy, so summing
    own-minus-opp reward gives the point margin. Returns
    (overall_winrate, left_winrate, right_winrate) where winrate
    is points_for / (points_for + points_against).
    """
    won = {LEFT: 0, RIGHT: 0}
    lost = {LEFT: 0, RIGHT: 0}
    obs, _ = env.reset()
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        side = info["agent_side"]
        if reward > 0:
            won[side] += 1
        elif reward < 0:
            lost[side] += 1
        if terminated or truncated:
            obs, _ = env.reset()

    def winrate(side: str) -> float:
        total = won[side] + lost[side]
        return won[side] / total if total else 0.0

    total_for = won[LEFT] + won[RIGHT]
    total_all = total_for + lost[LEFT] + lost[RIGHT]
    overall = total_for / total_all if total_all else 0.0
    return overall, winrate(LEFT), winrate(RIGHT)


def apply_finetune_hyperparams(
    model: PPO, lr: float, ent_coef: float
) -> None:
    """Lower lr/entropy on a resumed model for gentle adaptation."""
    model.learning_rate = lr
    model.lr_schedule = lambda _progress: lr
    model.ent_coef = ent_coef


def finetune(args: argparse.Namespace) -> None:
    """Resume the pretrained policy and fine-tune against Unity."""
    opponent = args.opponent or (
        str(Path(args.pretrained).with_name("best_1_policy.pt"))
    )

    def make() -> UnityPickleballEnv:
        return UnityPickleballEnv(
            env_path=args.env_path,
            opponent_model_path=opponent,
            episode_steps=args.episode_steps,
            seed=args.seed,
            worker_id=args.worker_id,
            base_port=args.base_port,
            graphics=not args.no_graphics,
        )

    vec = DummyVecEnv([make])
    model = PPO.load(args.pretrained, env=vec, device=args.device)
    apply_finetune_hyperparams(model, args.lr, args.ent_coef)
    manager = CheckpointManager(Path(args.out_dir))
    eval_env = vec.envs[0]
    total = 0
    while total < args.total_steps:
        model.learn(
            args.interval, reset_num_timesteps=False,
            progress_bar=False,
        )
        total += args.interval
        overall, left, right = eval_vs_opponent(
            model, eval_env, args.eval_steps
        )
        sim_overall, _, _ = sim_evaluate(
            model, config.MAX_LEVEL, config.EVAL_TRIALS,
            args.seed + 9000,
        )
        kept = manager.consider(
            model, score=overall + sim_overall,
            level=config.MAX_LEVEL, overall_rate=overall,
            left_rate=left, right_rate=right, total_steps=total,
        )
        # unconditional latest so weights survive an ineligible eval
        out_dir = Path(args.out_dir)
        model.save(out_dir / "latest.zip")
        export_torchscript(model, out_dir / "latest_policy.pt")
        print(
            f"[finetune] steps={total} vs_frozen overall={overall:.2f}"
            f" L={left:.2f} R={right:.2f} | simL5={sim_overall:.2f}"
            f" kept={kept}"
        )
    manager.promote_best()
    vec.close()
    print("[finetune] done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the pretrained policy in Unity."
    )
    parser.add_argument("--env-path", required=True)
    parser.add_argument(
        "--pretrained",
        default=str(
            config.PACKAGE_DIR / "checkpoints_l5" / "best_1.zip"
        ),
        help="Phase 2 SB3 zip to resume",
    )
    parser.add_argument(
        "--opponent", default=None,
        help="frozen TorchScript opponent (default: pretrained .pt)",
    )
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--interval", type=int, default=50_000)
    parser.add_argument("--episode-steps", type=int, default=256)
    parser.add_argument("--eval-steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=FINETUNE_LR)
    parser.add_argument("--ent-coef", type=float, default=FINETUNE_ENT)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=5005)
    parser.add_argument("--no-graphics", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=str(config.PACKAGE_DIR / "checkpoints_unity"),
    )
    args = parser.parse_args()
    if not Path(args.pretrained).is_file():
        raise FileNotFoundError(
            f"pretrained zip not found: {args.pretrained}"
        )
    return args


def main() -> None:
    finetune(parse_args())


if __name__ == "__main__":
    main()

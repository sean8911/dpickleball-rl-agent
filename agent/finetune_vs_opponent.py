"""Fine-tune our agent in Unity against another team's agent.

Unlike self-play (competent clones never score on each other), a
DIFFERENT opponent has exploitable weaknesses, so there is a real,
non-saturating win/loss signal. The opponent submission class is
loaded from its folder and reads the raw frame itself; our agent
plays the opposite side (fixed). Reward = own - opponent point.
Cross-training against other teams' checkpoints is done only with
explicit authorization.
"""

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agent import config
from agent.checkpoints import CheckpointManager, export_torchscript
from agent.envs.unity_env import LEFT, RIGHT, UnityPickleballEnv
from agent.finetune_unity import (
    apply_finetune_hyperparams,
    eval_vs_opponent,
    FINETUNE_ENT,
    FINETUNE_LR,
)

# the opponent submission file is teamX.py (left) / teamY.py (right)
_MODULE = {"TeamX": "teamX", "TeamY": "teamY"}


def load_opponent(opponent_dir: str, class_name: str):
    """Import the opponent team's submission class from its folder."""
    opp_dir = Path(opponent_dir).resolve()
    if not opp_dir.is_dir():
        raise FileNotFoundError(f"opponent dir not found: {opp_dir}")
    if str(opp_dir) not in sys.path:
        sys.path.insert(0, str(opp_dir))
    module = __import__(_MODULE[class_name])
    return getattr(module, class_name)()


def finetune(args: argparse.Namespace) -> None:
    """Resume our agent and fine-tune it against the opponent."""
    opponent = load_opponent(args.opponent_dir, args.opponent_class)
    # opponent TeamX is left-side; we play the opposite side
    our_side = RIGHT if args.opponent_class == "TeamX" else LEFT

    def make() -> UnityPickleballEnv:
        return UnityPickleballEnv(
            env_path=args.env_path,
            opponent_model_path="",
            external_opponent=opponent,
            fixed_side=our_side,
            episode_steps=args.episode_steps,
            seed=args.seed,
            worker_id=args.worker_id,
            base_port=args.base_port,
            graphics=not args.no_graphics,
            timeout_wait=args.timeout_wait,
        )

    vec = DummyVecEnv([make])
    model = PPO.load(args.pretrained, env=vec, device=args.device)
    apply_finetune_hyperparams(model, args.lr, args.ent_coef)
    manager = CheckpointManager(Path(args.out_dir))
    eval_env = vec.envs[0]
    total = 0
    while total < args.total_steps:
        model.learn(args.interval, reset_num_timesteps=False,
                    progress_bar=False)
        total += args.interval
        overall, left, right = eval_vs_opponent(
            model, eval_env, args.eval_steps
        )
        kept = manager.consider(
            model, score=overall, level=config.MAX_LEVEL,
            overall_rate=overall, left_rate=left, right_rate=right,
            total_steps=total,
        )
        out_dir = Path(args.out_dir)
        model.save(out_dir / "latest.zip")
        export_torchscript(model, out_dir / "latest_policy.pt")
        print(
            f"[vs_opp] steps={total} winrate={overall:.2f} "
            f"(our_side={our_side}) kept={kept}"
        )
    manager.promote_best()
    vec.close()
    print("[vs_opp] done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune vs another team's agent in Unity."
    )
    parser.add_argument("--env-path", required=True)
    parser.add_argument("--opponent-dir", required=True)
    parser.add_argument("--opponent-class", default="TeamX",
                        choices=("TeamX", "TeamY"))
    parser.add_argument(
        "--pretrained",
        default=str(
            config.PACKAGE_DIR / "checkpoints_selfplay" / "best_1.zip"
        ),
    )
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--interval", type=int, default=50_000)
    parser.add_argument("--episode-steps", type=int, default=256)
    parser.add_argument("--eval-steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=FINETUNE_LR)
    parser.add_argument("--ent-coef", type=float, default=FINETUNE_ENT)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=5005)
    parser.add_argument("--timeout-wait", type=int, default=220)
    parser.add_argument("--no-graphics", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=str(config.PACKAGE_DIR / "checkpoints_vs_opp"),
    )
    args = parser.parse_args()
    if not Path(args.pretrained).is_file():
        raise FileNotFoundError(
            f"pretrained not found: {args.pretrained}"
        )
    return args


def main() -> None:
    finetune(parse_args())


if __name__ == "__main__":
    main()

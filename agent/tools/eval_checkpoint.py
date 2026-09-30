"""Evaluate a trained SB3 checkpoint across all curriculum levels.

Reports per-side success per level, including the slow-ball (level
1) retention slice the plan requires after top-level refinement
(catastrophic-forgetting check).
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from agent import config
from agent.train_sim import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-level eval of a trained checkpoint."
    )
    parser.add_argument("--model", required=True, help="SB3 .zip")
    parser.add_argument("--trials", type=int,
                        default=config.EVAL_TRIALS)
    parser.add_argument("--seed", type=int,
                        default=config.SEED + 9000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    model = PPO.load(str(model_path), device=args.device)
    for level in range(1, config.MAX_LEVEL + 1):
        overall, left, right = evaluate(
            model, level, args.trials, args.seed
        )
        print(
            f"[eval] level {level}: overall={overall:.2f} "
            f"L={left:.2f} R={right:.2f} "
            f"gap={abs(left - right):.2f}"
        )


if __name__ == "__main__":
    main()

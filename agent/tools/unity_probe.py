"""Measure absolute rally competence of a policy in live Unity.

The win-vs-frozen metric is blind to mutual flailing (both paddles
broken still split points ~50/50). This probe reports points scored
and the mean steps between points: long gaps = real rallies, short
gaps = serve-and-miss flailing. Use it to confirm sim->Unity
transfer before committing to a long fine-tune.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from agent.envs.unity_env import UnityPickleballEnv


def probe(
    model_path: str, env: UnityPickleballEnv, steps: int
) -> dict:
    """Run the TorchScript policy; report points and rally gaps."""
    policy = torch.jit.load(model_path).eval()
    obs, _ = env.reset()
    points = 0
    gaps = []
    since = 0
    for _ in range(steps):
        tensor = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            action = policy(tensor)[0].numpy().astype(np.int32)
        obs, reward, term, trunc, _ = env.step(action)
        since += 1
        if reward != 0:
            points += 1
            gaps.append(since)
            since = 0
        if term or trunc:
            obs, _ = env.reset()
    return {
        "steps": steps,
        "points": points,
        "mean_gap": float(np.mean(gaps)) if gaps else float(steps),
        "max_gap": int(np.max(gaps)) if gaps else steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unity rally-competence probe."
    )
    parser.add_argument("--env-path", required=True)
    parser.add_argument(
        "--model", required=True, help="TorchScript .pt"
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=5005)
    parser.add_argument("--timeout-wait", type=int, default=220)
    args = parser.parse_args()
    if not Path(args.model).is_file():
        raise FileNotFoundError(f"model not found: {args.model}")
    env = UnityPickleballEnv(
        env_path=args.env_path,
        opponent_model_path=args.model,
        episode_steps=10_000,  # avoid truncation during the probe
        worker_id=args.worker_id,
        base_port=args.base_port,
        graphics=True,
        timeout_wait=args.timeout_wait,
    )
    try:
        stats = probe(args.model, env, args.steps)
    finally:
        env.close()
    print(
        f"[probe] steps={stats['steps']} points={stats['points']} "
        f"mean_gap={stats['mean_gap']:.1f} max_gap={stats['max_gap']}"
    )
    print(
        "[probe] interpretation: mean_gap >> ~40 means rallies are "
        "sustained; ~20-40 means serve-and-miss flailing"
    )


if __name__ == "__main__":
    main()

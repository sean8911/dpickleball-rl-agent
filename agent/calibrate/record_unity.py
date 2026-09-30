"""Record dPickleBall Unity rollouts to npz chunks for calibration.

Phase 0 entry point. Verifies the env interface (--inspect) and
saves (frames, actions, rewards) chunks consumed by fit_physics.py.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.custom_side_channel import (
    CustomDataChannel,
    StringSideChannel,
)
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

from agent import config

# single-branch probes so per-branch displacement is measurable
SWEEP_ACTIONS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (2, 0, 0),
    (0, 1, 0),
    (0, 2, 0),
    (0, 0, 1),
    (0, 0, 2),
)

ACTION_MODES = ("random", "square", "sweep", "chase")

# official scripted square pattern from dPickleBallEnv/test_paral.py
SQUARE_ACTIONS: tuple[tuple[int, int, int], ...] = (
    (1, 0, 0),
    (0, 1, 1),
    (2, 0, 2),
    (0, 2, 0),
)

# deadband so the chaser does not jitter when roughly aligned
CHASE_Y_DEADBAND_PX = 1.5
# x-distance windows for the chaser strike/retreat behavior
CHASE_STRIKE_RANGE_PX = 14.0
CHASE_HOME_RANGE_PX = 30.0


class ActionSource:
    """Produce per-step actions for both paddles in a given mode.

    Open-loop modes ignore the frame; the chase mode reads it via
    the extractor to drive both paddles toward the ball, producing
    fast sustained rallies for wall-bounce and hit calibration.
    """

    def __init__(self, mode: str, seed: int) -> None:
        if mode not in ACTION_MODES:
            raise ValueError(f"unknown action mode: {mode!r}")
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self._held: Optional[np.ndarray] = None
        self._hold_left = 0
        self._left_ext = None
        self._right_ext = None
        if mode == "chase":
            from agent.extractor import OpenCVStateExtractor

            self._left_ext = OpenCVStateExtractor(side="left")
            self._right_ext = OpenCVStateExtractor(side="right")

    def _random_held(self) -> np.ndarray:
        # hold each random action 5-20 steps so paddles travel
        if self._hold_left <= 0 or self._held is None:
            self._held = self.rng.integers(
                0, config.ACTION_CHOICES,
                size=(2, config.ACTION_BRANCHES),
            ).astype(np.int32)
            self._hold_left = int(self.rng.integers(5, 21))
        self._hold_left -= 1
        return self._held

    def _chase_one(
        self,
        paddle_xy: Optional[Tuple[float, float]],
        ball_xy: Optional[Tuple[float, float]],
        is_right: bool,
    ) -> np.ndarray:
        """Align in y, then strike forward in x to drive the ball.

        Forward = toward the net (left paddle +x, right paddle -x).
        Striking only when the ball is near imparts speed (Unity hit
        speed scales with paddle motion), creating fast angled
        rallies and wall bounces.
        """
        if paddle_xy is None or ball_xy is None:
            return np.zeros(3, dtype=np.int32)
        dy = ball_xy[1] - paddle_xy[1]
        if dy > CHASE_Y_DEADBAND_PX:
            vert = 2  # ball below -> move down
        elif dy < -CHASE_Y_DEADBAND_PX:
            vert = 1  # ball above -> move up
        else:
            vert = 0
        # strike forward when ball is close in x, else hold home line
        dx = abs(ball_xy[0] - paddle_xy[0])
        if dx < CHASE_STRIKE_RANGE_PX:
            horiz = 2 if is_right else 1  # toward the net
        elif dx > CHASE_HOME_RANGE_PX:
            horiz = 1 if is_right else 2  # retreat to home line
        else:
            horiz = 0
        rot = (
            int(self.rng.integers(0, 3))
            if self.rng.random() < 0.2 else 0
        )
        return np.array([vert, horiz, rot], dtype=np.int32)

    def next_pair(
        self, step: int, frame: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return actions for (left, right), shape (2, 3) int32."""
        if self.mode == "random":
            return self._random_held()
        if self.mode == "chase":
            if frame is None:
                return np.zeros((2, 3), dtype=np.int32)
            det_l = self._left_ext.detect(frame)
            det_r = self._right_ext.detect(frame)
            ball = det_l.ball_xy
            return np.stack([
                self._chase_one(det_l.paddle_xy, ball, is_right=False),
                self._chase_one(det_r.paddle_xy, ball, is_right=True),
            ])
        if self.mode == "square":
            i = (step // config.SWEEP_HOLD_STEPS) % len(SQUARE_ACTIONS)
            one = np.array(SQUARE_ACTIONS[i], dtype=np.int32)
        else:
            i = (step // config.SWEEP_HOLD_STEPS) % len(SWEEP_ACTIONS)
            one = np.array(SWEEP_ACTIONS[i], dtype=np.int32)
        return np.stack([one, one])


class ChunkWriter:
    """Buffer steps and flush compressed npz chunks to a run dir."""

    def __init__(self, run_dir: Path, chunk_steps: int) -> None:
        run_dir.mkdir(parents=True, exist_ok=False)
        self.run_dir = run_dir
        self.chunk_steps = chunk_steps
        self.chunks_written = 0
        self.total_steps = 0
        self._frames: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[np.ndarray] = []

    def add(
        self,
        frame: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        """Append one step; flush when the chunk is full."""
        self._frames.append(frame)
        self._actions.append(actions.astype(np.int8))
        self._rewards.append(rewards.astype(np.float32))
        self.total_steps += 1
        if len(self._frames) >= self.chunk_steps:
            self.flush()

    def flush(self) -> None:
        """Write buffered steps as one compressed npz chunk."""
        if not self._frames:
            return
        path = self.run_dir / f"chunk_{self.chunks_written:04d}.npz"
        np.savez_compressed(
            path,
            frames=np.stack(self._frames),
            actions=np.stack(self._actions),
            rewards=np.stack(self._rewards),
        )
        n = len(self._frames)
        self._frames, self._actions, self._rewards = [], [], []
        self.chunks_written += 1
        print(f"[rec] wrote {path.name} ({n} steps)")


def frame_to_uint8(obs: np.ndarray) -> np.ndarray:
    """Convert an ML-Agents (C,H,W) float frame to (H,W,C) uint8."""
    expected = (config.IMG_C, config.IMG_H, config.IMG_W)
    if obs.shape != expected:
        raise ValueError(
            f"unexpected obs shape {obs.shape}, expected {expected}"
        )
    img = np.transpose(obs, (1, 2, 0))
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def inspect_env(observation: dict, agents: list) -> None:
    """Print verified interface facts for the Phase 0 notes."""
    print("[inspect] agents:", agents)
    for i, agent in enumerate(agents):
        entry = observation[agent]
        obs_list = entry["observation"]
        print(f"[inspect] agent[{i}] = {agent}")
        print(f"[inspect]   obs entries: {len(obs_list)}")
        for j, o in enumerate(obs_list):
            arr = np.asarray(o)
            print(
                f"[inspect]   obs[{j}]: shape={arr.shape} "
                f"dtype={arr.dtype} min={arr.min():.3f} "
                f"max={arr.max():.3f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record Unity rollouts for calibration."
    )
    parser.add_argument(
        "--env-path", required=True, help="path to dp.exe"
    )
    parser.add_argument(
        "--mode", default="random", choices=ACTION_MODES
    )
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument(
        "--out", default=None,
        help="output dir (default: agent/data/recordings/<ts>)",
    )
    parser.add_argument(
        "--serve-code", type=int, default=config.DEFAULT_SERVE_CODE
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=5005)
    parser.add_argument("--timeout-wait", type=int, default=60,
                        help="seconds to wait for the Unity handshake")
    parser.add_argument("--graphics", action="store_true")
    parser.add_argument(
        "--inspect", action="store_true",
        help="print interface facts after reset and exit",
    )
    return parser.parse_args()


def launch_env(args: argparse.Namespace) -> UnityParallelEnv:
    """Launch dp.exe and return the wrapped parallel env."""
    env_path = Path(args.env_path)
    if not env_path.is_file():
        raise FileNotFoundError(f"env not found: {env_path}")
    string_channel = StringSideChannel()
    data_channel = CustomDataChannel()
    data_channel.send_data(serve=args.serve_code, p1=0, p2=0)
    unity_env = UnityEnvironment(
        file_name=str(env_path),
        worker_id=args.worker_id,
        base_port=args.base_port,
        side_channels=[string_channel, data_channel],
        no_graphics=not args.graphics,
        timeout_wait=args.timeout_wait,
    )
    return UnityParallelEnv(unity_env)


def record(
    env: UnityParallelEnv,
    writer: ChunkWriter,
    source: ActionSource,
    target_steps: int,
) -> dict:
    """Run the env until target_steps, saving every step."""
    points = [0, 0]
    matches = 0
    observation = env.reset()
    start = time.monotonic()
    step = 0
    last_frame: Optional[np.ndarray] = None
    while writer.total_steps < target_steps:
        if not env.agents:
            observation = env.reset()
            matches += 1
            last_frame = None
            continue
        left, right = env.agents[0], env.agents[1]
        # closed-loop modes act on the most recent frame
        pair = source.next_pair(step, last_frame)
        observation, rewards, dones, _ = env.step(
            {left: pair[0], right: pair[1]}
        )
        # obs only populated on agent 0; both paddles share it
        frame = frame_to_uint8(
            np.asarray(observation[left]["observation"][0])
        )
        last_frame = frame
        rew = np.array([rewards[left], rewards[right]])
        writer.add(frame, pair, rew)
        if rew[0] > 0:
            points[0] += 1
        if rew[1] > 0:
            points[1] += 1
        if dones[left] or dones[right]:
            observation = env.reset()
            matches += 1
        step += 1
        if writer.total_steps % 1000 == 0:
            rate = writer.total_steps / (time.monotonic() - start)
            print(
                f"[rec] {writer.total_steps}/{target_steps} steps "
                f"({rate:.1f}/s) points={points}"
            )
    elapsed = time.monotonic() - start
    return {
        "points": points,
        "matches": matches,
        "steps_per_sec": writer.total_steps / elapsed,
    }


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    env = launch_env(args)
    try:
        if args.inspect:
            observation = env.reset()
            inspect_env(observation, env.agents)
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            Path(args.out) if args.out
            else config.RECORDINGS_DIR / f"{args.mode}_{stamp}"
        )
        writer = ChunkWriter(run_dir, config.REC_CHUNK_STEPS)
        stats = record(env, writer, ActionSource(args.mode, args.seed),
                       args.steps)
        writer.flush()
        meta = {
            "mode": args.mode,
            "serve_code": args.serve_code,
            "total_steps": writer.total_steps,
            "chunks": writer.chunks_written,
            "started_at": stamp,
            "env_path": str(args.env_path),
            **stats,
        }
        meta_path = run_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[rec] done: {writer.total_steps} steps -> {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

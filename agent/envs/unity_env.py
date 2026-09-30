"""Feature-based single-agent wrapper over the live Unity env.

Phase 3 fine-tuning. Drives BOTH paddles every step: the learner
side (alternating per episode) and a frozen TorchScript opponent on
the other side. Observations are the 16-dim feature vector built by
the OpenCV extractor from the shared frame (obs only populated on
agent 0). Reward is own-minus-opponent point. Unity's `done` never
fires (see MEASUREMENTS), so episodes are truncated after a fixed
number of steps; the underlying match plays on continuously.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.custom_side_channel import (
    CustomDataChannel,
    StringSideChannel,
)
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

from agent import config
from agent.extractor import OpenCVStateExtractor

LEFT = "left"
RIGHT = "right"


def _shared_frame(observation: dict, agents: list) -> np.ndarray:
    """Return the (3,84,168) frame; only agent 0 carries it."""
    return np.asarray(observation[agents[0]]["observation"][0])


class UnityPickleballEnv(gym.Env):
    """One Unity instance as a single-agent feature env.

    The controlled side alternates each episode so the shared
    policy keeps both paddles balanced. The opponent paddle is
    driven by a frozen TorchScript actor reading the same frame
    through its own-side extractor.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_path: str,
        opponent_model_path: str,
        episode_steps: int = 256,
        seed: int = 0,
        worker_id: int = 0,
        base_port: int = 5005,
        graphics: bool = True,
        serve_code: int = config.DEFAULT_SERVE_CODE,
        timeout_wait: int = 60,
        external_opponent=None,
        fixed_side: str = None,
    ) -> None:
        super().__init__()
        if not Path(env_path).is_file():
            raise FileNotFoundError(f"env not found: {env_path}")
        # external_opponent.policy(frame, reward) drives the other side
        self._external_opponent = external_opponent
        self._fixed_side = fixed_side
        if external_opponent is None and not Path(
            opponent_model_path
        ).is_file():
            raise FileNotFoundError(
                f"opponent model not found: {opponent_model_path}"
            )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(config.FEATURE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            [config.ACTION_CHOICES] * config.ACTION_BRANCHES
        )
        self.episode_steps = episode_steps
        self._rng = np.random.default_rng(seed)
        self._opponent = (
            None if external_opponent is not None
            else torch.jit.load(opponent_model_path).eval()
        )
        self._left_ext = OpenCVStateExtractor(side=LEFT)
        self._right_ext = OpenCVStateExtractor(side=RIGHT)
        string_channel = StringSideChannel()
        data_channel = CustomDataChannel()
        data_channel.send_data(serve=serve_code, p1=0, p2=0)
        unity = UnityEnvironment(
            file_name=str(env_path),
            worker_id=worker_id,
            base_port=base_port,
            side_channels=[string_channel, data_channel],
            no_graphics=not graphics,
            timeout_wait=timeout_wait,
        )
        self._env = UnityParallelEnv(unity)
        self._obs = self._env.reset()
        self.agent_side = LEFT
        self._steps = 0

    def _ext(self, side: str) -> OpenCVStateExtractor:
        return self._left_ext if side == LEFT else self._right_ext

    def _opponent_side(self) -> str:
        return RIGHT if self.agent_side == LEFT else LEFT

    def _frame(self) -> np.ndarray:
        return _shared_frame(self._obs, self._env.agents)

    def _opponent_action(self, frame: np.ndarray) -> np.ndarray:
        if self._external_opponent is not None:
            # external agent reads the raw frame and extracts itself
            act = self._external_opponent.policy(frame, 0.0)
            return np.asarray(act, dtype=np.int32).reshape(-1)
        opp_ext = self._ext(self._opponent_side())
        feats = opp_ext.features_from_image(frame)
        tensor = torch.from_numpy(feats).float().unsqueeze(0)
        with torch.no_grad():
            action = self._opponent(tensor)[0].numpy()
        return action.astype(np.int32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if not self._env.agents:  # match ended (defensive)
            self._obs = self._env.reset()
        # alternate side each episode unless fixed by an opponent
        if self._fixed_side is not None:
            self.agent_side = self._fixed_side
        else:
            self.agent_side = (
                LEFT if self._rng.random() < 0.5 else RIGHT
            )
        self._left_ext.reset()
        self._right_ext.reset()
        self._steps = 0
        feats = self._ext(self.agent_side).features_from_image(
            self._frame()
        )
        return feats, {}

    def step(self, action):
        agents = self._env.agents
        left, right = agents[0], agents[1]
        frame = self._frame()
        learner_act = np.asarray(action, dtype=np.int32).copy()
        opp_act = self._opponent_action(frame)
        # both paddles vertical so Unity matches the sim
        learner_act[2] = 0
        opp_act[2] = 0
        if self.agent_side == LEFT:
            actions = {left: learner_act, right: opp_act}
        else:
            actions = {left: opp_act, right: learner_act}
        self._obs, rewards, dones, _ = self._env.step(actions)

        learner_agent = left if self.agent_side == LEFT else right
        opp_agent = right if self.agent_side == LEFT else left
        reward = float(rewards[learner_agent] - rewards[opp_agent])

        self._steps += 1
        truncated = self._steps >= self.episode_steps
        terminated = bool(dones[learner_agent])
        next_frame = (
            self._frame() if self._env.agents else frame
        )
        feats = self._ext(self.agent_side).features_from_image(
            next_frame
        )
        info = {"agent_side": self.agent_side}
        return feats, reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

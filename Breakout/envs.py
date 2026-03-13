from __future__ import annotations

from typing import Sequence

import ale_py
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

ALE_BREAKOUT_ENV_ID = "ALE/Breakout-v5"
DEFAULT_OBS_SHAPE = (84, 84)


def register_atari_envs() -> None:
    gym.register_envs(ale_py)


class TorchResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, shape: Sequence[int] = DEFAULT_OBS_SHAPE):
        super().__init__(env)
        self.shape = tuple(shape)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.shape,
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.uint8)
        if observation_array.ndim == 2:
            observation_tensor = torch.from_numpy(observation_array).unsqueeze(0)
        elif observation_array.ndim == 3 and observation_array.shape[-1] == 1:
            observation_tensor = torch.from_numpy(observation_array).permute(2, 0, 1)
        else:
            raise ValueError(
                f"Expected grayscale observation with 2 or 3 dims, got {observation_array.shape}"
            )

        resized = resize(
            observation_tensor,
            list(self.shape),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return resized.squeeze(0).clamp(0, 255).to(torch.uint8).cpu().numpy()


def create_breakout_env(
    *,
    render_mode: str | None = None,
    frameskip: int = 4,
    repeat_action_probability: float = 0.0,
    resize_shape: Sequence[int] = DEFAULT_OBS_SHAPE,
    full_action_space: bool = False,
    seed: int | None = None,
) -> gym.Env:
    register_atari_envs()
    env = gym.make(
        ALE_BREAKOUT_ENV_ID,
        obs_type="grayscale",
        frameskip=frameskip,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        render_mode=render_mode,
    )
    env = TorchResizeObservation(env, resize_shape)
    if seed is not None:
        env.action_space.seed(seed)
    return env

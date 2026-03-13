from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

# Observations are stored as uint8 [0, 255] to minimise disk usage.
# They are scaled to float32 [0.0, 1.0] at batch time.
# Keeping inputs in this range is important for JEPA: the SmoothL1 loss
# ‖ẑ_{t+1} − z̄_{t+1}‖ is sensitive to absolute scale, so unnormalised
# pixel values (0–255) would cause large initial gradients and unstable
# early training before the encoder has learned anything meaningful.
PIXEL_NORM_SCALE: float = 1.0 / 255.0


@dataclass(frozen=True)
class ShardIndex:
    path: Path
    valid_indices: np.ndarray

    @property
    def size(self) -> int:
        return int(self.valid_indices.shape[0])


class BreakoutTransitionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        data_dir: str | Path,
        context_length: int = 4,
        prediction_horizon: int = 1,
    ):
        if context_length < 1:
            raise ValueError("context_length must be at least 1")
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1")

        self.data_dir = Path(data_dir)
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.shards = self._index_shards()
        if not self.shards:
            raise FileNotFoundError(f"No transition shards found in {self.data_dir}")

        shard_sizes = [shard.size for shard in self.shards]
        self.cumulative_sizes = np.cumsum(shard_sizes)

    def _index_shards(self) -> list[ShardIndex]:
        shard_paths = sorted(self.data_dir.glob("transitions_*.npz"))
        shard_indices: list[ShardIndex] = []
        for shard_path in shard_paths:
            with np.load(shard_path) as shard:
                episode_steps = shard["episode_step"]
                episode_ids = shard["episode_id"]
                max_transition_index = episode_steps.shape[0] - self.prediction_horizon
                if max_transition_index < 0:
                    continue

                valid_indices_list: list[int] = []
                for idx in range(max_transition_index + 1):
                    # Need enough history for the context stack.
                    if idx < self.context_length - 1:
                        continue

                    # Need enough future transitions for the horizon.
                    if idx + self.prediction_horizon - 1 >= episode_steps.shape[0]:
                        continue

                    # Context should start at valid local index.
                    if idx - self.context_length + 1 < 0:
                        continue

                    # Keep windows inside one episode only.
                    if episode_ids[idx + self.prediction_horizon - 1] != episode_ids[idx]:
                        continue

                    # Episode step consistency for both history and horizon.
                    if episode_steps[idx] < self.context_length - 1:
                        continue
                    expected_future_step = episode_steps[idx] + self.prediction_horizon - 1
                    if episode_steps[idx + self.prediction_horizon - 1] != expected_future_step:
                        continue

                    valid_indices_list.append(idx)

                valid_indices = np.asarray(valid_indices_list, dtype=np.int64)
            if valid_indices.size > 0:
                shard_indices.append(ShardIndex(path=shard_path, valid_indices=valid_indices))
        return shard_indices

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        shard_id = int(np.searchsorted(self.cumulative_sizes, index, side="right"))
        previous = 0 if shard_id == 0 else int(self.cumulative_sizes[shard_id - 1])
        local_index = int(index - previous)
        shard_index = self.shards[shard_id]
        transition_index = int(shard_index.valid_indices[local_index])
        shard = self._load_shard(shard_index.path)

        context_start = transition_index - self.context_length + 1
        context_frames = shard["obs"][context_start : transition_index + 1]

        horizon_end_index = transition_index + self.prediction_horizon - 1
        horizon_actions = shard["action"][transition_index : horizon_end_index + 1]

        next_frame = shard["next_obs"][transition_index]
        if self.context_length == 1:
            target_frames = next_frame[None, ...]
        else:
            target_frames = np.concatenate((context_frames[1:], next_frame[None, ...]), axis=0)

        # k-step target window ending at t + prediction_horizon.
        horizon_next_frame = shard["next_obs"][horizon_end_index]
        if self.context_length == 1:
            horizon_target_frames = horizon_next_frame[None, ...]
        else:
            prior_start = horizon_end_index - self.context_length + 2
            prior_frames = shard["obs"][prior_start : horizon_end_index + 1]
            horizon_target_frames = np.concatenate((prior_frames, horizon_next_frame[None, ...]), axis=0)

        return {
            "context": torch.from_numpy(context_frames.astype(np.float32) * PIXEL_NORM_SCALE),
            "action": torch.tensor(int(shard["action"][transition_index]), dtype=torch.long),
            "target": torch.from_numpy(target_frames.astype(np.float32) * PIXEL_NORM_SCALE),
            "future_actions": torch.from_numpy(horizon_actions.astype(np.int64)),
            "horizon_target": torch.from_numpy(horizon_target_frames.astype(np.float32) * PIXEL_NORM_SCALE),
            "reward": torch.tensor(float(shard["reward"][transition_index]), dtype=torch.float32),
            "terminated": torch.tensor(bool(shard["terminated"][transition_index]), dtype=torch.bool),
            "truncated": torch.tensor(bool(shard["truncated"][transition_index]), dtype=torch.bool),
            "episode_id": torch.tensor(int(shard["episode_id"][transition_index]), dtype=torch.long),
            "episode_step": torch.tensor(int(shard["episode_step"][transition_index]), dtype=torch.long),
        }

    @staticmethod
    @lru_cache(maxsize=16)
    def _load_shard(path: Path) -> dict[str, np.ndarray]:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

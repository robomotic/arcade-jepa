from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from .envs import ALE_BREAKOUT_ENV_ID, DEFAULT_OBS_SHAPE, create_breakout_env
except ImportError:
    from envs import ALE_BREAKOUT_ENV_ID, DEFAULT_OBS_SHAPE, create_breakout_env


@dataclass
class ShardBuffer:
    observations: list[np.ndarray] = field(default_factory=list)
    next_observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    episode_ids: list[int] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def append(
        self,
        observation: np.ndarray,
        next_observation: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        episode_id: int,
        episode_step: int,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.uint8))
        self.next_observations.append(np.asarray(next_observation, dtype=np.uint8))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.episode_ids.append(int(episode_id))
        self.episode_steps.append(int(episode_step))

    def clear(self) -> None:
        self.observations.clear()
        self.next_observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.terminated.clear()
        self.truncated.clear()
        self.episode_ids.clear()
        self.episode_steps.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect random Breakout transitions into compressed NPZ shards.")
    parser.add_argument("--output-dir", type=Path, default=Path("Breakout/data/random"))
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    return parser.parse_args()


def flush_shard(buffer: ShardBuffer, output_dir: Path, shard_index: int) -> Path | None:
    if len(buffer) == 0:
        return None

    output_path = output_dir / f"transitions_{shard_index:05d}.npz"
    np.savez_compressed(
        output_path,
        obs=np.stack(buffer.observations).astype(np.uint8),
        next_obs=np.stack(buffer.next_observations).astype(np.uint8),
        action=np.asarray(buffer.actions, dtype=np.int64),
        reward=np.asarray(buffer.rewards, dtype=np.float32),
        terminated=np.asarray(buffer.terminated, dtype=np.bool_),
        truncated=np.asarray(buffer.truncated, dtype=np.bool_),
        episode_id=np.asarray(buffer.episode_ids, dtype=np.int32),
        episode_step=np.asarray(buffer.episode_steps, dtype=np.int32),
    )
    buffer.clear()
    return output_path


def write_run_summary(
    *,
    output_dir: Path,
    num_steps: int,
    shard_paths: list[Path],
    seed: int,
    frameskip: int,
    repeat_action_probability: float,
) -> None:
    summary_path = output_dir / "run_summary.json"
    summary = {
        "env_id": ALE_BREAKOUT_ENV_ID,
        "observation_shape": list(DEFAULT_OBS_SHAPE),
        "num_steps": num_steps,
        "num_shards": len(shard_paths),
        "seed": seed,
        "frameskip": frameskip,
        "repeat_action_probability": repeat_action_probability,
        "shards": [path.name for path in shard_paths],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = create_breakout_env(
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
        seed=args.seed,
    )

    observation, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    buffer = ShardBuffer()
    shard_paths: list[Path] = []
    episode_id = 0
    episode_step = 0

    for global_step in range(args.num_steps):
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, _ = env.step(action)

        buffer.append(
            observation=observation,
            next_observation=next_observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            episode_id=episode_id,
            episode_step=episode_step,
        )

        if len(buffer) >= args.shard_size:
            shard_path = flush_shard(buffer, args.output_dir, len(shard_paths))
            if shard_path is not None:
                shard_paths.append(shard_path)
                print(f"Wrote {shard_path} ({global_step + 1} / {args.num_steps} transitions)")

        if terminated or truncated:
            observation, _ = env.reset()
            episode_id += 1
            episode_step = 0
        else:
            observation = next_observation
            episode_step += 1

    final_shard = flush_shard(buffer, args.output_dir, len(shard_paths))
    if final_shard is not None:
        shard_paths.append(final_shard)
        print(f"Wrote {final_shard} (final shard)")

    write_run_summary(
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        shard_paths=shard_paths,
        seed=args.seed,
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
    )
    env.close()

    print(f"Collected {args.num_steps} transitions into {len(shard_paths)} shard(s) at {args.output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

try:
    from .envs import ALE_BREAKOUT_ENV_ID, create_breakout_env
except ImportError:
    from envs import ALE_BREAKOUT_ENV_ID, create_breakout_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a random-policy smoke test on ALE Breakout.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = create_breakout_env(seed=args.seed)
    observation, _ = env.reset(seed=args.seed)

    print(f"Environment: {ALE_BREAKOUT_ENV_ID}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Initial Observation Shape: {observation.shape}")

    total_reward = 0.0
    episode_reward = 0.0
    steps = 0
    episode = 0

    for _ in range(args.steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        episode_reward += reward
        steps += 1

        if steps % 100 == 0:
            print(f"Step {steps}, Episode {episode}, Episode Reward: {episode_reward}, Total Reward: {total_reward}")

        if terminated or truncated:
            print(
                f"Episode {episode} finished after {steps} total steps with episode reward {episode_reward}"
            )
            observation, _ = env.reset()
            episode += 1
            episode_reward = 0.0

    print(f"Finished. Final steps: {steps}, Final Total Reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()

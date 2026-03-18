from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import deque
from pathlib import Path

import numpy as np
import torch

try:
    from .envs import create_breakout_env
    from .plan_mppi import PIXEL_NORM_SCALE, MppiPlanner, load_world_model
except ImportError:
    from envs import create_breakout_env
    from plan_mppi import PIXEL_NORM_SCALE, MppiPlanner, load_world_model


NOOP_ACTION = 0
FIRE_ACTION = 1
LEFT_ACTION = 3
RIGHT_ACTION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MPPI gameplay with a Stage-1 JEPA world model.")
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("Breakout/checkpoints/eval_mppi"))

    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--context-length", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=0)

    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--action-switch-penalty", type=float, default=0.02)
    parser.add_argument("--latent-norm-penalty", type=float, default=0.01)
    parser.add_argument("--disable-reuse-best-sequence", action="store_true")

    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    parser.add_argument("--force-fire-on-reset", action="store_true")
    parser.add_argument("--launch-random-moves", type=int, default=10)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _reset_with_launch(env, rng: np.random.Generator, force_fire: bool, launch_random_moves: int) -> np.ndarray:
    observation, _ = env.reset()

    warmup_steps = int(rng.integers(1, 11))
    for _ in range(warmup_steps):
        observation, _reward, terminated, truncated, _ = env.step(NOOP_ACTION)
        if terminated or truncated:
            observation, _ = env.reset()

    for _ in range(max(0, launch_random_moves)):
        action = LEFT_ACTION if rng.random() < 0.5 else RIGHT_ACTION
        observation, _reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()

    if force_fire:
        observation, _reward, terminated, truncated, _ = env.step(FIRE_ACTION)
        if terminated or truncated:
            observation, _ = env.reset()

    return np.asarray(observation, dtype=np.uint8)


def _context_to_tensor(context_frames: deque[np.ndarray], device: str) -> torch.Tensor:
    stacked = np.stack(list(context_frames), axis=0)
    tensor = torch.from_numpy(stacked).to(device=device, dtype=torch.float32)
    return tensor * PIXEL_NORM_SCALE


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    context_length_arg = None if args.context_length <= 0 else args.context_length
    latent_dim_arg = None if args.latent_dim <= 0 else args.latent_dim
    encoder, predictor, context_length, latent_dim = load_world_model(
        args.encoder_checkpoint,
        device=args.device,
        context_length=context_length_arg,
        latent_dim=latent_dim_arg,
    )

    planner = MppiPlanner(
        encoder,
        predictor,
        device=args.device,
        num_samples=args.num_samples,
        horizon=args.horizon,
        gamma=args.gamma,
        temperature=args.temperature,
        action_switch_penalty=args.action_switch_penalty,
        latent_norm_penalty=args.latent_norm_penalty,
        reuse_best_sequence=not args.disable_reuse_best_sequence,
    )

    env = create_breakout_env(
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    episodes_rows: list[dict[str, float | int | str]] = []

    for episode_index in range(1, args.episodes + 1):
        observation = _reset_with_launch(
            env,
            rng,
            force_fire=args.force_fire_on_reset,
            launch_random_moves=args.launch_random_moves,
        )
        context_frames: deque[np.ndarray] = deque([observation.copy() for _ in range(context_length)], maxlen=context_length)

        episode_return = 0.0
        score_sum = 0.0
        score_best = float("-inf")
        steps = 0

        for step in range(args.max_steps):
            context_tensor = _context_to_tensor(context_frames, args.device)
            plan = planner.plan_tensor(context_tensor, step_index=step)

            next_obs, reward, terminated, truncated, _info = env.step(plan.action)
            next_obs = np.asarray(next_obs, dtype=np.uint8)
            context_frames.append(next_obs)

            episode_return += float(reward)
            score_sum += plan.mean_score
            score_best = max(score_best, plan.best_score)
            steps = step + 1

            if terminated or truncated:
                break

        episodes_rows.append(
            {
                "run_id": run_id,
                "episode": episode_index,
                "return": episode_return,
                "steps": steps,
                "mean_plan_score": score_sum / max(steps, 1),
                "best_plan_score": score_best,
            }
        )

    env.close()

    returns = np.asarray([float(r["return"]) for r in episodes_rows], dtype=np.float64)
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))

    episodes_csv = args.output_dir / "eval_episodes.csv"
    write_header = not episodes_csv.exists()
    with episodes_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "episode", "return", "steps", "mean_plan_score", "best_plan_score"],
        )
        if write_header:
            writer.writeheader()
        writer.writerows(episodes_rows)

    runs_csv = args.output_dir / "eval_runs.csv"
    run_row = {
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "encoder_checkpoint": str(args.encoder_checkpoint),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "num_samples": args.num_samples,
        "horizon": args.horizon,
        "gamma": args.gamma,
        "temperature": args.temperature,
        "action_switch_penalty": args.action_switch_penalty,
        "latent_norm_penalty": args.latent_norm_penalty,
        "context_length": context_length,
        "latent_dim": latent_dim,
        "mean_return": mean_return,
        "std_return": std_return,
    }
    write_header = not runs_csv.exists()
    with runs_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(run_row)

    print(
        f"MPPI summary | mean_return={mean_return:.2f} std_return={std_return:.2f} "
        f"episodes={args.episodes} run_id={run_id}"
    )
    print(f"Saved: {runs_csv}")
    print(f"Saved: {episodes_csv}")


if __name__ == "__main__":
    main()
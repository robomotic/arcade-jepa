from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
import torch

try:
    from .envs import create_breakout_env
    from .plan_mppi import MppiPlanner, load_world_model
except ImportError:
    from envs import create_breakout_env
    from plan_mppi import MppiPlanner, load_world_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a curiosity-driven MPPI planner in ALE/Breakout-v5.")
    parser.add_argument("--jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("Breakout/checkpoints/eval_mppi"))
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-fire-on-reset", action="store_true")
    parser.add_argument("--disable-reuse-best-sequence", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def append_csv_row(path: Path, row: dict[str, float | int], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    encoder, predictor = load_world_model(
        checkpoint_path=args.jepa_checkpoint,
        context_length=args.context_length,
        latent_dim=args.latent_dim,
        device=args.device,
    )
    planner = MppiPlanner(
        encoder,
        predictor,
        num_samples=args.num_samples,
        horizon=args.horizon,
        gamma=args.gamma,
        device=args.device,
        seed=args.seed,
        force_fire_on_reset=args.force_fire_on_reset,
        reuse_best_sequence=not args.disable_reuse_best_sequence,
    )

    env = create_breakout_env(seed=args.seed)
    episode_csv = args.output_dir / "eval_episodes.csv"
    summary_csv = args.output_dir / "eval_runs.csv"

    returns: list[float] = []
    try:
        for episode in range(1, args.episodes + 1):
            planner.reset_episode()
            obs, _ = env.reset(seed=args.seed + episode - 1)
            context_frames: deque[np.ndarray] = deque(maxlen=args.context_length)
            for _ in range(args.context_length):
                context_frames.append(np.asarray(obs, dtype=np.uint8))

            total_return = 0.0
            step = 0
            done = False
            while not done and step < args.max_steps:
                step += 1
                action = planner.plan(context_frames, step)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                total_return += float(reward)
                done = bool(terminated or truncated)
                context_frames.append(np.asarray(next_obs, dtype=np.uint8))

            returns.append(total_return)
            append_csv_row(
                episode_csv,
                {"episode": episode, "return": total_return, "steps": step},
                ["episode", "return", "steps"],
            )
            print(f"Episode {episode}: return={total_return:.1f} steps={step}")
    finally:
        env.close()

    mean_return = float(np.mean(returns)) if returns else 0.0
    std_return = float(np.std(returns, ddof=0)) if returns else 0.0
    append_csv_row(
        summary_csv,
        {
            "episodes": args.episodes,
            "mean_return": mean_return,
            "std_return": std_return,
            "num_samples": args.num_samples,
            "horizon": args.horizon,
            "gamma": args.gamma,
        },
        ["episodes", "mean_return", "std_return", "num_samples", "horizon", "gamma"],
    )
    print(
        f"MPPI summary | mean_return={mean_return:.2f} std_return={std_return:.2f} "
        f"| baseline_random=1.10 baseline_stage2=10.85"
    )


if __name__ == "__main__":
    main()
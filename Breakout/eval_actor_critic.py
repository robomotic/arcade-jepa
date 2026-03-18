"""Evaluate a trained Actor-Critic policy (and optionally record gameplay video).

Usage
-----
    # Metrics-only run (20 episodes):
    python Breakout/eval_actor_critic.py \\
        --ac-checkpoint Breakout/checkpoints/ac_ppo/ac_update_003900.pt \\
        --encoder-checkpoint Breakout/checkpoints/stage1_run_20260318/jepa_epoch_008.pt \\
        --episodes 20

    # With gameplay video (1 episode):
    python Breakout/eval_actor_critic.py \\
        --ac-checkpoint Breakout/checkpoints/ac_ppo/ac_update_003900.pt \\
        --encoder-checkpoint Breakout/checkpoints/stage1_run_20260318/jepa_epoch_008.pt \\
        --episodes 1 --record-video
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch

try:
    from .envs import create_breakout_env
    from .models import ActorCriticHead, ConvEncoder
    from .plan_mppi import PIXEL_NORM_SCALE, load_world_model
except ImportError:
    from envs import create_breakout_env
    from models import ActorCriticHead, ConvEncoder
    from plan_mppi import PIXEL_NORM_SCALE, load_world_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Actor-Critic policy on Breakout.")
    p.add_argument("--ac-checkpoint", type=Path, required=True,
                   help="Path to an ac_update_*.pt checkpoint produced by train_actor_critic.py.")
    p.add_argument("--encoder-checkpoint", type=Path, required=True,
                   help="Path to the Stage-1 JEPA encoder checkpoint used during training.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory to write CSV/video (default: same dir as ac-checkpoint).")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--greedy", action="store_true",
                   help="Use argmax (greedy) policy instead of sampling.")
    p.add_argument("--record-video", action="store_true",
                   help="Record an MP4 for every episode (requires moviepy).")
    p.add_argument("--frameskip", type=int, default=4)
    p.add_argument("--repeat-action-probability", type=float, default=0.0)
    p.add_argument("--context-length", type=int, default=0)
    p.add_argument("--latent-dim", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        out = args.output_dir
    else:
        out = args.ac_checkpoint.parent / "eval"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _obs_to_latent(
    context_frames: deque[np.ndarray],
    encoder: ConvEncoder,
    device: str,
) -> torch.Tensor:
    stacked = np.stack(list(context_frames), axis=0)
    t = torch.from_numpy(stacked).to(device=device, dtype=torch.float32).unsqueeze(0)
    return encoder(t * PIXEL_NORM_SCALE)


def main() -> None:
    args = parse_args()
    out_dir = _make_output_dir(args)

    # ----- load frozen encoder -----
    context_length_arg = None if args.context_length <= 0 else args.context_length
    latent_dim_arg = None if args.latent_dim <= 0 else args.latent_dim
    encoder, _predictor, context_length, latent_dim = load_world_model(
        args.encoder_checkpoint,
        device=args.device,
        context_length=context_length_arg,
        latent_dim=latent_dim_arg,
    )

    # ----- load actor-critic head -----
    ac_ckpt = torch.load(args.ac_checkpoint, map_location=args.device, weights_only=False)
    ac_head = ActorCriticHead(
        latent_dim=ac_ckpt.get("latent_dim", latent_dim),
        num_actions=4,
        hidden_dim=ac_ckpt.get("args", {}).get("hidden_dim", 256),
    ).to(args.device)
    ac_head.load_state_dict(ac_ckpt["ac_head"])
    ac_head.eval()

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ----- set up video recording if requested -----
    render_mode = "rgb_array" if args.record_video else None

    episode_rows: list[dict] = []

    for ep_idx in range(1, args.episodes + 1):
        ep_seed = args.seed + ep_idx

        if args.record_video:
            from gymnasium.wrappers import RecordVideo
            video_dir = out_dir / f"videos_{run_id}"
            video_dir.mkdir(parents=True, exist_ok=True)
            env = create_breakout_env(
                render_mode=render_mode,
                frameskip=args.frameskip,
                repeat_action_probability=args.repeat_action_probability,
                seed=ep_seed,
            )
            env = RecordVideo(
                env,
                video_folder=str(video_dir),
                episode_trigger=lambda ep: True,
                name_prefix=f"ac_{run_id}_ep{ep_idx:03d}",
                disable_logger=True,
            )
        else:
            env = create_breakout_env(
                frameskip=args.frameskip,
                repeat_action_probability=args.repeat_action_probability,
                seed=ep_seed,
            )

        raw_obs, _ = env.reset(seed=ep_seed)
        obs = np.asarray(raw_obs, dtype=np.uint8)
        context_frames: deque[np.ndarray] = deque(
            [obs.copy() for _ in range(context_length)], maxlen=context_length
        )

        ep_return = 0.0
        steps = 0

        with torch.no_grad():
            for step in range(args.max_steps):
                latent = _obs_to_latent(context_frames, encoder, args.device)

                if args.greedy:
                    logits, _ = ac_head(latent)
                    action = int(logits.argmax(dim=-1).item())
                else:
                    action_t, _, _, _ = ac_head.get_action_and_value(latent)
                    action = int(action_t.item())

                raw_obs, reward, terminated, truncated, _ = env.step(action)
                obs = np.asarray(raw_obs, dtype=np.uint8)
                context_frames.append(obs)

                ep_return += float(reward)
                steps = step + 1

                if terminated or truncated:
                    break

        env.close()

        video_path = ""
        if args.record_video:
            videos = sorted(video_dir.glob(f"ac_{run_id}_ep{ep_idx:03d}*.mp4"))
            if videos:
                video_path = str(videos[0])

        episode_rows.append({
            "run_id": run_id,
            "episode": ep_idx,
            "return": ep_return,
            "steps": steps,
            "video_path": video_path,
        })
        print(f"ep={ep_idx:3d}  return={ep_return:.1f}  steps={steps}  video={video_path or 'n/a'}")

    returns = [float(r["return"]) for r in episode_rows]
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns))
    se = std_r / math.sqrt(len(returns))
    ci95_low = mean_r - 1.96 * se
    ci95_high = mean_r + 1.96 * se

    # ----- write episodes CSV -----
    episodes_csv = out_dir / "eval_episodes.csv"
    write_header = not episodes_csv.exists()
    with episodes_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run_id", "episode", "return", "steps", "video_path"]
        )
        if write_header:
            writer.writeheader()
        writer.writerows(episode_rows)

    # ----- write run summary CSV -----
    runs_csv = out_dir / "eval_runs.csv"
    run_row = {
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ac_checkpoint": str(args.ac_checkpoint),
        "encoder_checkpoint": str(args.encoder_checkpoint),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "greedy": args.greedy,
        "mean_return": round(mean_r, 4),
        "std_return": round(std_r, 4),
        "ci95_low": round(ci95_low, 4),
        "ci95_high": round(ci95_high, 4),
    }
    write_header = not runs_csv.exists()
    with runs_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(run_row)

    print(
        f"\nSummary | mean_return={mean_r:.2f}  std={std_r:.2f}  "
        f"CI95=[{ci95_low:.2f}, {ci95_high:.2f}]  episodes={args.episodes}"
    )
    print(f"Saved: {episodes_csv}")
    print(f"Saved: {runs_csv}")


if __name__ == "__main__":
    main()

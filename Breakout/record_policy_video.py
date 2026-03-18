from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from .envs import create_breakout_env
    from .models import ConvEncoder, QHead
    from .plan_mppi import MppiPlanner, load_world_model
except ImportError:
    from envs import create_breakout_env
    from models import ConvEncoder, QHead
    from plan_mppi import MppiPlanner, load_world_model


NUM_BREAKOUT_ACTIONS = 4
ACTION_NAMES = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
}
PIXEL_NORM_SCALE = 1.0 / 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record an MP4 of a JEPA policy rollout in Breakout.")
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--q-checkpoint", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--planner", type=str, choices=("q_head", "mppi"), default="q_head")
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--episode-index", type=int, default=1, help="1-based episode to record.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--force-fire-on-reset", action="store_true")
    parser.add_argument("--disable-reuse-best-sequence", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_models(args: argparse.Namespace) -> tuple[ConvEncoder, QHead]:
    if args.q_checkpoint is None:
        raise ValueError("--q-checkpoint is required when --planner q_head")

    enc_ckpt = torch.load(args.encoder_checkpoint, map_location=args.device, weights_only=False)
    encoder = ConvEncoder(input_channels=args.context_length, latent_dim=args.latent_dim).to(args.device)
    encoder.load_state_dict(enc_ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    q_ckpt = torch.load(args.q_checkpoint, map_location=args.device, weights_only=False)
    q_head = QHead(latent_dim=args.latent_dim, num_actions=NUM_BREAKOUT_ACTIONS).to(args.device)
    q_head.load_state_dict(q_ckpt["q_head"])
    q_head.eval()
    for p in q_head.parameters():
        p.requires_grad_(False)

    return encoder, q_head


def build_context_tensor(context_frames: deque[np.ndarray], device: str) -> torch.Tensor:
    stacked = np.stack(list(context_frames), axis=0).astype(np.float32) * PIXEL_NORM_SCALE
    tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)
    return tensor


def choose_action(q_values: torch.Tensor, epsilon: float) -> int:
    greedy_action = int(q_values.argmax(dim=1).item())
    if epsilon <= 0.0:
        return greedy_action
    if torch.rand(1).item() < epsilon:
        return int(torch.randint(0, NUM_BREAKOUT_ACTIONS, (1,)).item())
    return greedy_action


def overlay_frame(
    frame_rgb: np.ndarray,
    *,
    episode: int,
    step: int,
    total_return: float,
    action: int,
    q_values: np.ndarray | None,
    planner_name: str,
    best_score: float | None = None,
    mean_score: float | None = None,
) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    panel = frame_bgr.copy()

    cv2.rectangle(panel, (8, 8), (308, 136), (0, 0, 0), thickness=-1)
    cv2.addWeighted(panel, 0.45, frame_bgr, 0.55, 0.0, frame_bgr)

    text_lines = [
        f"Episode: {episode}",
        f"Step: {step}",
        f"Return: {total_return:.1f}",
        f"Planner: {planner_name}",
        f"Action: {ACTION_NAMES.get(action, str(action))}",
    ]
    if best_score is not None:
        text_lines.append(f"Best score: {best_score:.3f}")
    if mean_score is not None:
        text_lines.append(f"Mean score: {mean_score:.3f}")
    if q_values is not None:
        text_lines.extend(
            [
                "Q-values:",
                f"  NOOP={q_values[0]:.3f}",
                f"  FIRE={q_values[1]:.3f}",
                f"  RIGHT={q_values[2]:.3f}",
                f"  LEFT={q_values[3]:.3f}",
            ]
        )
    y = 28
    for line in text_lines:
        cv2.putText(frame_bgr, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 12

    return frame_bgr


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.planner == "q_head":
        encoder, q_head = load_models(args)
        planner = None
    else:
        encoder, predictor = load_world_model(
            checkpoint_path=args.encoder_checkpoint,
            context_length=args.context_length,
            latent_dim=args.latent_dim,
            device=args.device,
        )
        q_head = None
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
    env = create_breakout_env(render_mode="rgb_array", seed=args.seed)

    writer = None
    try:
        for episode in range(1, args.episodes + 1):
            if planner is not None:
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
                best_score = None
                mean_score = None
                q_values_np = None
                if planner is None:
                    context = build_context_tensor(context_frames, args.device)
                    with torch.no_grad():
                        latent = encoder(context)
                        q_values = q_head(latent)
                    action = choose_action(q_values, args.epsilon)
                    q_values_np = q_values.squeeze(0).detach().cpu().numpy()
                else:
                    plan = planner.plan_debug(context_frames, step)
                    action = plan.action
                    best_score = plan.best_score
                    mean_score = plan.mean_score

                next_obs, reward, terminated, truncated, _ = env.step(action)
                total_return += float(reward)
                done = bool(terminated or truncated)
                context_frames.append(np.asarray(next_obs, dtype=np.uint8))

                if episode == args.episode_index:
                    frame_rgb = np.asarray(env.render(), dtype=np.uint8)
                    frame_bgr = overlay_frame(
                        frame_rgb,
                        episode=episode,
                        step=step,
                        total_return=total_return,
                        action=action,
                        q_values=q_values_np,
                        planner_name=args.planner,
                        best_score=best_score,
                        mean_score=mean_score,
                    )
                    if writer is None:
                        h, w = frame_bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(args.output_path), fourcc, float(args.fps), (w, h))
                    writer.write(frame_bgr)

            print(f"Episode {episode}: return={total_return:.1f} steps={step}")
            if episode == args.episode_index:
                print(f"Recorded episode {episode} to {args.output_path}")
    finally:
        env.close()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
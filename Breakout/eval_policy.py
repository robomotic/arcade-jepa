"""Evaluate a trained Q-Head in the ALE/Breakout-v5 environment.

Loads a frozen JEPA encoder together with a trained Q-Head (from either Stage
1.5 or Stage 2) and runs N greedy episodes in the real environment, reporting
per-episode returns and a summary table.

Both Stage 1.5 checkpoints (``q_imagination_epoch_*.pt``) and Stage 2
checkpoints (``q_head_epoch_*.pt``) use the same ``{"q_head": ...}`` key, so
this script works with either.

Usage:
    python Breakout/eval_policy.py \\
        --encoder-checkpoint  Breakout/checkpoints/jepa/jepa_epoch_020.pt \\
        --q-checkpoint        Breakout/checkpoints/q_imagination/q_imagination_epoch_010.pt \\
        --episodes            20

Optional render (requires a display):
    python Breakout/eval_policy.py ... --render
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import torch

try:
    from .envs import create_breakout_env
    from .models import ConvEncoder, QHead
except ImportError:
    from envs import create_breakout_env
    from models import ConvEncoder, QHead

NUM_BREAKOUT_ACTIONS = 4
PIXEL_NORM_SCALE: float = 1.0 / 255.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a JEPA Q-Head in ALE/Breakout-v5."
    )
    parser.add_argument(
        "--encoder-checkpoint", type=Path, required=True,
        help="Path to a Stage 1 JEPA checkpoint (contains 'encoder' key).",
    )
    parser.add_argument(
        "--q-checkpoint", type=Path, required=True,
        help="Path to a Q-Head checkpoint (Stage 1.5 or Stage 2; contains 'q_head' key).",
    )
    parser.add_argument("--context-length", type=int, default=4)
    # Default 256: validated best latent dimensionality from Stage 1.
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.05,
        help="ε-greedy rate during evaluation (small value for near-greedy play).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=10_000,
        help="Hard cap on steps per episode (prevents infinite loops on no-op policies).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--render", action="store_true",
        help="Open a human-render window (requires a display).",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(args: argparse.Namespace) -> tuple[ConvEncoder, QHead]:
    """Load and freeze the encoder and Q-Head from their respective checkpoints."""
    # Encoder
    enc_ckpt = torch.load(args.encoder_checkpoint, map_location=args.device, weights_only=False)
    encoder = ConvEncoder(
        input_channels=args.context_length, latent_dim=args.latent_dim
    ).to(args.device)
    encoder.load_state_dict(enc_ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Q-Head (compatible with Stage 1.5 and Stage 2 checkpoint formats)
    q_ckpt = torch.load(args.q_checkpoint, map_location=args.device, weights_only=False)
    q_head = QHead(latent_dim=args.latent_dim, num_actions=NUM_BREAKOUT_ACTIONS).to(args.device)
    q_head.load_state_dict(q_ckpt["q_head"])
    q_head.eval()
    for p in q_head.parameters():
        p.requires_grad_(False)

    return encoder, q_head


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env,
    encoder: ConvEncoder,
    q_head: QHead,
    context_length: int,
    epsilon: float,
    max_steps: int,
    device: str,
) -> tuple[float, int]:
    """Run one episode and return ``(total_return, steps_taken)``.

    A deque of length ``context_length`` is used as a sliding frame buffer.
    It is pre-filled with the initial observation so the encoder always
    receives a full ``(context_length, 84, 84)`` stack, even on the first step.
    """
    obs, _ = env.reset()
    # Pre-fill the context buffer with the initial frame.
    frame_buffer: collections.deque[np.ndarray] = collections.deque(
        [obs] * context_length, maxlen=context_length
    )

    total_return = 0.0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated) and steps < max_steps:
        # Build float32 context tensor: (1, C, H, W)
        context = np.stack(list(frame_buffer), axis=0).astype(np.float32) * PIXEL_NORM_SCALE
        context_tensor = torch.from_numpy(context).unsqueeze(0).to(device)

        with torch.no_grad():
            z = encoder(context_tensor)        # (1, D)
            q_values = q_head(z)               # (1, A)

        # ε-greedy action
        if epsilon > 0.0 and torch.rand(1).item() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(q_values.argmax(dim=1).item())

        obs, reward, terminated, truncated, _ = env.step(action)
        frame_buffer.append(obs)
        total_return += float(reward)
        steps += 1

    return total_return, steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"Device:             {args.device}")
    print(f"Encoder checkpoint: {args.encoder_checkpoint}")
    print(f"Q checkpoint:       {args.q_checkpoint}")
    print(f"Latent dim:         {args.latent_dim}  context: {args.context_length}")
    print(f"Episodes:           {args.episodes}  ε={args.epsilon}  max_steps={args.max_steps}")
    print()

    encoder, q_head = load_models(args)

    env = create_breakout_env(
        render_mode="human" if args.render else None,
        seed=args.seed,
    )

    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    for ep in range(1, args.episodes + 1):
        ret, length = run_episode(
            env, encoder, q_head,
            args.context_length, args.epsilon, args.max_steps, args.device,
        )
        episode_returns.append(ret)
        episode_lengths.append(length)
        print(f"  Episode {ep:3d}:  return={ret:7.1f}  steps={length:5d}")

    env.close()

    mean_ret = float(np.mean(episode_returns))
    std_ret = float(np.std(episode_returns))
    mean_len = float(np.mean(episode_lengths))

    print()
    print("=" * 52)
    print(f"  Episodes:          {args.episodes}")
    print(f"  Mean return:       {mean_ret:.2f} ± {std_ret:.2f}")
    print(f"  Min / Max return:  {min(episode_returns):.1f} / {max(episode_returns):.1f}")
    print(f"  Mean episode len:  {mean_len:.1f} steps")
    print("=" * 52)


if __name__ == "__main__":
    main()

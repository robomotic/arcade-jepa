"""Stage 1.5: Train a Q-Head via Latent Imagination.

After JEPA pretraining the Predictor already knows how the world moves and the
RewardHead knows what earns points. This script uses those two frozen models to
"dream" multi-step latent rollouts entirely in embedding space — no env.step()
required — and trains a Q-Head using Bellman targets computed from imagined
rewards and bootstrapped Q-values.

Key insight: Because the world model was trained on raw random data (not expert
play), the imagination trajectories cover a wide distribution of game states.
Training the Q-Head here is substantially better than behaviour cloning, which
would only teach the agent to reproduce random actions.

Usage (after Stage 1):
    python Breakout/train_latent_imagination.py \\
        --data-dir             Breakout/data/random \\
        --jepa-checkpoint      Breakout/checkpoints/jepa/jepa_epoch_020.pt \\
        --output-dir           Breakout/checkpoints/q_imagination \\
        --rollout-length       5 \\
        --epochs               10
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .dataset import BreakoutTransitionDataset
    from .models import ActionConditionedPredictor, ConvEncoder, QHead, RewardHead
except ImportError:
    from dataset import BreakoutTransitionDataset
    from models import ActionConditionedPredictor, ConvEncoder, QHead, RewardHead

NUM_BREAKOUT_ACTIONS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1.5: train a Q-Head by rolling out imagined latent trajectories."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Breakout/data/random"))
    parser.add_argument("--jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("Breakout/checkpoints/q_imagination"))
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=5,
        help="Number of imagination steps per real starting state.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--target-sync-epochs",
        type=int,
        default=1,
        help="Sync the target Q-Head every N epochs.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_world_model(
    checkpoint_path: Path,
    context_length: int,
    latent_dim: int,
    device: str,
) -> tuple[ConvEncoder, ActionConditionedPredictor, RewardHead]:
    """Load and freeze the encoder, predictor, and reward head from a JEPA checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder = ConvEncoder(input_channels=context_length, latent_dim=latent_dim).to(device)
    encoder.load_state_dict(checkpoint["encoder"])

    predictor = ActionConditionedPredictor(latent_dim=latent_dim).to(device)
    predictor.load_state_dict(checkpoint["predictor"])

    reward_head = RewardHead(latent_dim=latent_dim, num_actions=NUM_BREAKOUT_ACTIONS).to(device)
    reward_head.load_state_dict(checkpoint["reward_head"])

    for model in (encoder, predictor, reward_head):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    return encoder, predictor, reward_head


def imagination_loss(
    z0: torch.Tensor,
    predictor: ActionConditionedPredictor,
    reward_head: RewardHead,
    q_head: QHead,
    target_q_head: QHead,
    rollout_length: int,
    gamma: float,
) -> torch.Tensor:
    """Roll out latent trajectories and return mean 1-step TD loss across the rollout.

    At each step the Q-Head picks a greedy action (exploitation), the frozen
    Predictor advances the latent state, and the frozen RewardHead supplies an
    imagined reward. A target-network copy of the Q-Head provides stable
    Bellman targets.

    Args:
        z0: Starting latent states ``(B, latent_dim)`` from real observations.
        predictor: Frozen world-model dynamics.
        reward_head: Frozen reward predictor.
        q_head: Online Q-Head being trained.
        target_q_head: Frozen copy of Q-Head for stable TD targets.
        rollout_length: Number of imagination steps.
        gamma: Discount factor.

    Returns:
        Scalar TD loss averaged over the rollout.
    """
    z = z0
    total_loss = torch.tensor(0.0, device=z.device)

    for _ in range(rollout_length):
        # Greedy action from the online Q-Head
        actions = q_head(z).argmax(dim=1)  # (B,)

        with torch.no_grad():
            z_next = predictor(z, actions)                          # (B, D)
            imagined_reward = reward_head(z, actions)               # (B,)
            next_q = target_q_head(z_next).max(dim=1).values       # (B,)
            td_target = imagined_reward + gamma * next_q

        # Loss only on the selected action's Q-value
        q_taken = q_head(z).gather(1, actions.unsqueeze(1)).squeeze(1)
        total_loss = total_loss + F.smooth_l1_loss(q_taken, td_target)

        z = z_next.detach()  # advance; block gradients through time

    return total_loss / rollout_length


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = BreakoutTransitionDataset(args.data_dir, context_length=args.context_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    encoder, predictor, reward_head = load_world_model(
        args.jepa_checkpoint, args.context_length, args.latent_dim, args.device
    )

    q_head = QHead(latent_dim=args.latent_dim, num_actions=NUM_BREAKOUT_ACTIONS).to(args.device)
    target_q_head = copy.deepcopy(q_head)
    target_q_head.eval()
    for p in target_q_head.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(q_head.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for batch in dataloader:
            context = batch["context"].to(args.device)

            # Encode real observations to get starting latent states
            with torch.no_grad():
                z0 = encoder(context)

            loss = imagination_loss(
                z0=z0,
                predictor=predictor,
                reward_head=reward_head,
                q_head=q_head,
                target_q_head=target_q_head,
                rollout_length=args.rollout_length,
                gamma=args.gamma,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        # Periodically sync the target Q-Head for stable Bellman targets
        if epoch % args.target_sync_epochs == 0:
            target_q_head.load_state_dict(q_head.state_dict())

        average_loss = running_loss / max(len(dataloader), 1)
        checkpoint_path = args.output_dir / f"q_imagination_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "q_head": q_head.state_dict(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "epoch": epoch,
                "loss": average_loss,
            },
            checkpoint_path,
        )
        print(f"Epoch {epoch}: imagination_td_loss={average_loss:.6f} saved={checkpoint_path}")


if __name__ == "__main__":
    main()

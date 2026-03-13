from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .dataset import BreakoutTransitionDataset
    from .models import ConvEncoder, QHead
except ImportError:
    from dataset import BreakoutTransitionDataset
    from models import ConvEncoder, QHead

NUM_BREAKOUT_ACTIONS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: train a Q-Head on frozen JEPA latents using 1-step TD Q-learning."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Breakout/data/random"))
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("Breakout/checkpoints/q_head"))
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_encoder(checkpoint_path: Path, context_length: int, latent_dim: int, device: str) -> ConvEncoder:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = ConvEncoder(input_channels=context_length, latent_dim=latent_dim).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = BreakoutTransitionDataset(args.data_dir, context_length=args.context_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    encoder = load_encoder(args.encoder_checkpoint, args.context_length, args.latent_dim, args.device)
    q_head = QHead(latent_dim=args.latent_dim, num_actions=NUM_BREAKOUT_ACTIONS).to(args.device)
    optimizer = torch.optim.Adam(q_head.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for batch in dataloader:
            context = batch["context"].to(args.device)
            target = batch["target"].to(args.device)
            actions = batch["action"].to(args.device)
            rewards = batch["reward"].to(args.device)
            terminated = batch["terminated"].to(args.device)
            truncated = batch["truncated"].to(args.device)

            with torch.no_grad():
                z_t = encoder(context)   # current-state latent  (B, D)
                z_t1 = encoder(target)   # next-state latent      (B, D)

                # 1-step Bellman target:
                # Q*(s,a) ≈ r + γ · max_{a'} Q(s', a')  (no bootstrap at episode end)
                done = (terminated | truncated).float()
                next_q_max = q_head(z_t1).max(dim=1).values
                td_target = rewards + args.gamma * (1.0 - done) * next_q_max

            # Select Q-value of the action actually taken
            q_values = q_head(z_t)
            q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q_taken, td_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        average_loss = running_loss / max(len(dataloader), 1)
        checkpoint_path = args.output_dir / f"q_head_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "q_head": q_head.state_dict(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "epoch": epoch,
                "loss": average_loss,
            },
            checkpoint_path,
        )
        print(f"Epoch {epoch}: td_loss={average_loss:.6f} saved={checkpoint_path}")


if __name__ == "__main__":
    main()

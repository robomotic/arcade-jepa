from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

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
    # Default 256: validated best latent dimensionality from Stage 1 grid sweep.
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--target-sync-epochs",
        type=int,
        default=1,
        help="Copy online Q-Head weights into the target Q-Head every N epochs.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data reserved for validation (monitoring only, no gradient).",
    )
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

    full_dataset = BreakoutTransitionDataset(args.data_dir, context_length=args.context_length)
    n = len(full_dataset)
    n_val = max(1, int(math.floor(n * args.val_split)))
    n_train = n - n_val
    # Deterministic split (no shuffle): first n_train → train, last n_val → val.
    indices = list(range(n))
    train_dataset = Subset(full_dataset, indices[:n_train])
    val_dataset   = Subset(full_dataset, indices[n_train:])
    print(f"Dataset: {n} samples  →  train={n_train}  val={n_val}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    encoder = load_encoder(args.encoder_checkpoint, args.context_length, args.latent_dim, args.device)
    q_head = QHead(latent_dim=args.latent_dim, num_actions=NUM_BREAKOUT_ACTIONS).to(args.device)
    # Target network: frozen copy of the online Q-Head used to compute stable
    # Bellman targets, avoiding the moving-target instability of the deadly triad.
    target_q_head = copy.deepcopy(q_head)
    target_q_head.eval()
    for p in target_q_head.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(q_head.parameters(), lr=args.learning_rate)

    def run_epoch(loader: DataLoader, train: bool) -> tuple[float, float, float]:
        """Returns (avg_td_loss, avg_q_std, avg_entropy)."""
        total_loss = total_q_std = total_entropy = 0.0
        count = 0
        q_head.train(train)
        ctx_mgr = torch.enable_grad() if train else torch.no_grad()
        with ctx_mgr:
            for batch in loader:
                context   = batch["context"].to(args.device)
                target    = batch["target"].to(args.device)
                actions   = batch["action"].to(args.device)
                rewards   = batch["reward"].to(args.device)
                terminated = batch["terminated"].to(args.device)
                truncated  = batch["truncated"].to(args.device)

                with torch.no_grad():
                    z_t  = encoder(context)
                    z_t1 = encoder(target)
                    done = (terminated | truncated).float()
                    next_q_max = target_q_head(z_t1).max(dim=1).values
                    td_target = rewards + args.gamma * (1.0 - done) * next_q_max

                q_values = q_head(z_t)                                    # (B, A)
                q_taken  = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(q_taken, td_target)

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    q_std_batch = q_values.std(dim=1).mean()
                    probs = torch.softmax(q_values, dim=1)
                    entropy_batch = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

                total_loss    += float(loss.detach())
                total_q_std   += float(q_std_batch)
                total_entropy += float(entropy_batch)
                count += 1

        return total_loss / max(count, 1), total_q_std / max(count, 1), total_entropy / max(count, 1)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_qstd, tr_ent = run_epoch(train_loader, train=True)

        # Hard-copy online → target Q-Head every N epochs.
        if epoch % args.target_sync_epochs == 0:
            target_q_head.load_state_dict(q_head.state_dict())

        val_loss, val_qstd, val_ent = run_epoch(val_loader, train=False)

        checkpoint_path = args.output_dir / f"q_head_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "q_head": q_head.state_dict(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
            },
            checkpoint_path,
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.6f}  val_loss={val_loss:.6f} | "
            f"q_std={tr_qstd:.4f}  entropy={tr_ent:.4f} | "
            f"saved={checkpoint_path.name}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

try:
    from .dataset import BreakoutTransitionDataset
    from .models import ActionConditionedPredictor, ConvEncoder, RewardHead
except ImportError:
    from dataset import BreakoutTransitionDataset
    from models import ActionConditionedPredictor, ConvEncoder, RewardHead


@torch.no_grad()
def update_ema(target_encoder: nn.Module, online_encoder: nn.Module, momentum: float) -> None:
    for target_param, online_param in zip(target_encoder.parameters(), online_encoder.parameters()):
        target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)


def apply_random_mask(frames: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
    """Zero out a random fraction of spatial positions across the frame stack.

    Each pixel location is independently masked with probability ``mask_ratio``.
    The mask is shared across channels (frames in the stack) so whole spatial
    regions go dark, forcing the encoder to reason about object identities
    (paddle, ball) rather than memorising individual pixel values.

    This is the Action-Conditioned Masking trick that distinguishes a true
    JEPA from a simple latent predictor: the online encoder must infer the
    missing content from context, while the EMA target encoder always sees
    the clean shifted window.

    Args:
        frames: float32 ``(B, C, H, W)`` context frames in ``[0.0, 1.0]``.
        mask_ratio: fraction of spatial positions to zero out (default 0.5).
    Returns:
        Masked tensor of same shape and dtype.
    """
    B, C, H, W = frames.shape
    noise = torch.rand(B, 1, H, W, device=frames.device, dtype=frames.dtype)
    mask = (noise > mask_ratio).to(frames.dtype)
    return frames * mask


def build_train_val_subsets(
    dataset: BreakoutTransitionDataset,
    val_ratio: float,
    seed: int,
) -> tuple[Subset, Subset]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    total_size = len(dataset)
    val_size = max(1, int(total_size * val_ratio))
    train_size = total_size - val_size
    if train_size <= 0:
        raise ValueError(
            f"Dataset too small for val_ratio={val_ratio}. "
            f"Need at least 2 samples, got {total_size}."
        )

    generator = torch.Generator().manual_seed(seed)
    permuted_indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = permuted_indices[:train_size]
    val_indices = permuted_indices[train_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple JEPA-style predictor on Breakout transition shards.")
    parser.add_argument("--data-dir", type=Path, default=Path("Breakout/data/random"))
    parser.add_argument("--output-dir", type=Path, default=Path("Breakout/checkpoints"))
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--train-horizon", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a checkpoint .pt file to resume training from. "
             "Epoch numbering continues after the saved epoch.",
    )
    parser.add_argument(
        "--reward-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Scalar multiplier on the RewardHead auxiliary loss relative to the JEPA loss. "
            "The default (1.0) preserves the original equal-weight behaviour. "
            "Raising this (e.g. 10.0) forces the RewardHead to fit the sparse reward "
            "distribution more tightly, which reduces the negative bias that causes "
            "Bellman-target collapse in Stage 1.5 imagination rollouts."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.train_horizon < 1:
        raise ValueError(f"train_horizon must be >= 1, got {args.train_horizon}")

    torch.manual_seed(args.seed)
    dataset = BreakoutTransitionDataset(
        args.data_dir,
        context_length=args.context_length,
        prediction_horizon=args.train_horizon,
    )
    train_subset, val_subset = build_train_val_subsets(dataset, val_ratio=args.val_ratio, seed=args.seed)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    encoder = ConvEncoder(input_channels=args.context_length, latent_dim=args.latent_dim).to(args.device)
    target_encoder = copy.deepcopy(encoder).to(args.device)
    predictor = ActionConditionedPredictor(latent_dim=args.latent_dim).to(args.device)
    reward_head = RewardHead(latent_dim=args.latent_dim).to(args.device)
    for parameter in target_encoder.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()) + list(reward_head.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    jepa_loss_fn = nn.SmoothL1Loss()
    reward_loss_fn = nn.SmoothL1Loss()

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        predictor.train()
        reward_head.train()

        train_total_loss = 0.0
        train_jepa_loss = 0.0
        train_reward_loss = 0.0
        train_copy_baseline = 0.0
        train_action_sensitivity = 0.0
        train_rollout_drift = 0.0

        train_batch_count = 0
        for batch in train_loader:
            train_batch_count += 1
            context = batch["context"].to(args.device)
            actions = batch["action"].to(args.device)
            future_actions = batch["future_actions"].to(args.device)
            horizon_targets = batch["horizon_target"].to(args.device)
            rewards = batch["reward"].to(args.device)

            # Action-conditioned masking: zero out mask_ratio of spatial pixels
            # in the context before the online encoder. The EMA target encoder
            # always sees the clean shifted window. This forces the encoder to
            # model object identities (ball, paddle) rather than pixel patterns.
            masked_context = apply_random_mask(context, mask_ratio=args.mask_ratio)
            online_latent = encoder(masked_context)
            with torch.no_grad():
                target_latent = target_encoder(horizon_targets)

            # Primary JEPA loss: optional k-step latent prediction.
            # train_horizon=1 recovers the original one-step objective.
            predicted_latent = online_latent
            for h in range(args.train_horizon):
                predicted_latent = predictor(predicted_latent, future_actions[:, h])
            jepa_loss = jepa_loss_fn(predicted_latent, target_latent)

            # Auxiliary reward loss: enriches the checkpoint for Stage 1.5
            # latent imagination (detach so reward head doesn't bias encoder).
            # reward_loss_weight scales the relative importance of this term;
            # higher values push the RewardHead to fit the sparse reward
            # distribution precisely, reducing the negative-bias artefact that
            # collapses Bellman targets in imagination rollouts.
            predicted_reward = reward_head(online_latent.detach(), actions)
            reward_loss = reward_loss_fn(predicted_reward, rewards)

            loss = jepa_loss + args.reward_loss_weight * reward_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_ema(target_encoder, encoder, args.ema_momentum)

            with torch.no_grad():
                copy_baseline = jepa_loss_fn(online_latent, target_latent)
                action_count = predictor.action_embedding.num_embeddings
                shifted_actions = (actions + 1) % action_count
                alt_predicted_latent = predictor(online_latent.detach(), shifted_actions)
                for h in range(1, args.train_horizon):
                    alt_predicted_latent = predictor(alt_predicted_latent, future_actions[:, h])
                action_sensitivity = (predicted_latent.detach() - alt_predicted_latent).abs().mean()

                rollout_latent = online_latent.detach()
                rollout_start = rollout_latent
                for _ in range(args.rollout_steps):
                    rollout_latent = predictor(rollout_latent, actions)
                rollout_drift = torch.linalg.vector_norm(
                    rollout_latent - rollout_start,
                    ord=2,
                    dim=1,
                ).mean()

            train_total_loss += float(loss.item())
            train_jepa_loss += float(jepa_loss.item())
            train_reward_loss += float(reward_loss.item())
            train_copy_baseline += float(copy_baseline.item())
            train_action_sensitivity += float(action_sensitivity.item())
            train_rollout_drift += float(rollout_drift.item())

            if args.max_train_batches > 0 and train_batch_count >= args.max_train_batches:
                break

        train_den = max(train_batch_count, 1)
        train_total_loss /= train_den
        train_jepa_loss /= train_den
        train_reward_loss /= train_den
        train_copy_baseline /= train_den
        train_action_sensitivity /= train_den
        train_rollout_drift /= train_den

        encoder.eval()
        predictor.eval()
        reward_head.eval()

        val_total_loss = 0.0
        val_jepa_loss = 0.0
        val_reward_loss = 0.0
        val_copy_baseline = 0.0
        val_action_sensitivity = 0.0
        val_rollout_drift = 0.0
        with torch.no_grad():
            val_batch_count = 0
            for batch in val_loader:
                val_batch_count += 1
                context = batch["context"].to(args.device)
                actions = batch["action"].to(args.device)
                future_actions = batch["future_actions"].to(args.device)
                horizon_targets = batch["horizon_target"].to(args.device)
                rewards = batch["reward"].to(args.device)

                masked_context = apply_random_mask(context, mask_ratio=args.mask_ratio)
                online_latent = encoder(masked_context)
                target_latent = target_encoder(horizon_targets)
                predicted_latent = online_latent
                for h in range(args.train_horizon):
                    predicted_latent = predictor(predicted_latent, future_actions[:, h])
                predicted_reward = reward_head(online_latent, actions)

                jepa_loss = jepa_loss_fn(predicted_latent, target_latent)
                reward_loss = reward_loss_fn(predicted_reward, rewards)
                loss = jepa_loss + args.reward_loss_weight * reward_loss

                copy_baseline = jepa_loss_fn(online_latent, target_latent)
                action_count = predictor.action_embedding.num_embeddings
                shifted_actions = (actions + 1) % action_count
                alt_predicted_latent = predictor(online_latent, shifted_actions)
                for h in range(1, args.train_horizon):
                    alt_predicted_latent = predictor(alt_predicted_latent, future_actions[:, h])
                action_sensitivity = (predicted_latent - alt_predicted_latent).abs().mean()

                rollout_latent = online_latent
                rollout_start = rollout_latent
                for _ in range(args.rollout_steps):
                    rollout_latent = predictor(rollout_latent, actions)
                rollout_drift = torch.linalg.vector_norm(
                    rollout_latent - rollout_start,
                    ord=2,
                    dim=1,
                ).mean()

                val_total_loss += float(loss.item())
                val_jepa_loss += float(jepa_loss.item())
                val_reward_loss += float(reward_loss.item())
                val_copy_baseline += float(copy_baseline.item())
                val_action_sensitivity += float(action_sensitivity.item())
                val_rollout_drift += float(rollout_drift.item())

                if args.max_val_batches > 0 and val_batch_count >= args.max_val_batches:
                    break

            val_den = max(val_batch_count, 1)
        val_total_loss /= val_den
        val_jepa_loss /= val_den
        val_reward_loss /= val_den
        val_copy_baseline /= val_den
        val_action_sensitivity /= val_den
        val_rollout_drift /= val_den

        checkpoint_path = args.output_dir / f"jepa_epoch_{epoch:03d}.pt"
        checkpoint_args = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "target_encoder": target_encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "reward_head": reward_head.state_dict(),
                "args": checkpoint_args,
                "epoch": epoch,
                "loss": train_total_loss,
                "metrics": {
                    "train_total_loss": train_total_loss,
                    "train_jepa_loss": train_jepa_loss,
                    "train_reward_loss": train_reward_loss,
                    "train_copy_baseline": train_copy_baseline,
                    "train_action_sensitivity": train_action_sensitivity,
                    "train_rollout_drift": train_rollout_drift,
                    "val_total_loss": val_total_loss,
                    "val_jepa_loss": val_jepa_loss,
                    "val_reward_loss": val_reward_loss,
                    "val_copy_baseline": val_copy_baseline,
                    "val_action_sensitivity": val_action_sensitivity,
                    "val_rollout_drift": val_rollout_drift,
                },
            },
            checkpoint_path,
        )
        print(
            f"Epoch {epoch}: "
            f"train_total={train_total_loss:.6f} "
            f"(jepa={train_jepa_loss:.6f}, reward={train_reward_loss:.6f}, "
            f"copy={train_copy_baseline:.6f}, sens={train_action_sensitivity:.6f}, "
            f"drift={train_rollout_drift:.6f}) | "
            f"val_total={val_total_loss:.6f} "
            f"(jepa={val_jepa_loss:.6f}, reward={val_reward_loss:.6f}, "
            f"copy={val_copy_baseline:.6f}, sens={val_action_sensitivity:.6f}, "
            f"drift={val_rollout_drift:.6f}) | "
            f"saved={checkpoint_path}"
        )


if __name__ == "__main__":
    main()

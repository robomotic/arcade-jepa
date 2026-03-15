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

Design choices
--------------
* ε-greedy exploration (``--epsilon``, default 0.1): pure argmax collapses to a
  single repeated action early in training when Q-values are random. A small ε
  injects diverse actions and avoids that degenerate fixed-point.
* Train / validation split (``--val-split``, default 0.1): mirrors Stage 1's
  diagnostic protocol — val loss / metrics reveal over-fitting and degenerate
  Q-value collapse before they reach Stage 2.
* Diagnostics per epoch: imagination TD loss, mean imagined reward, Q-value
  spread (std across actions), and greedy-action entropy — all reported for
  both train and val.
* Graceful reward_head fallback: checkpoints saved before the RewardHead was
  added to Stage 1 are accepted; a randomly-initialised (frozen) head is used
  so imagination still runs, and a warning is printed.

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
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from .dataset import BreakoutTransitionDataset
    from .models import ActionConditionedPredictor, ConvEncoder, QHead, RewardHead
except ImportError:
    from dataset import BreakoutTransitionDataset
    from models import ActionConditionedPredictor, ConvEncoder, QHead, RewardHead

NUM_BREAKOUT_ACTIONS = 4


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1.5: train a Q-Head by rolling out imagined latent trajectories."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Breakout/data/random"))
    parser.add_argument("--jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("Breakout/checkpoints/q_imagination"))
    parser.add_argument("--context-length", type=int, default=4)
    # Default 256: validated best latent dimensionality from Stage 1 grid sweep.
    parser.add_argument("--latent-dim", type=int, default=256)
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
        "--epsilon",
        type=float,
        default=0.1,
        help="ε-greedy exploration rate during imagined rollouts (0 = greedy).",
    )
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
        help="Fraction of starting states held out for validation (no grad).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_world_model(
    checkpoint_path: Path,
    context_length: int,
    latent_dim: int,
    device: str,
) -> tuple[ConvEncoder, ActionConditionedPredictor, RewardHead]:
    """Load and freeze the encoder, predictor, and reward head from a JEPA checkpoint.

    If the checkpoint pre-dates the ``RewardHead`` addition (missing key), a
    randomly-initialised frozen head is used instead and a warning is printed.
    Imagination will still run; reward signals will simply be uninformative
    until Stage 1 is re-run with the current code.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder = ConvEncoder(input_channels=context_length, latent_dim=latent_dim).to(device)
    encoder.load_state_dict(checkpoint["encoder"])

    predictor = ActionConditionedPredictor(latent_dim=latent_dim).to(device)
    predictor.load_state_dict(checkpoint["predictor"])

    reward_head = RewardHead(latent_dim=latent_dim, num_actions=NUM_BREAKOUT_ACTIONS).to(device)
    if "reward_head" in checkpoint:
        reward_head.load_state_dict(checkpoint["reward_head"])
    else:
        print(
            "WARNING: checkpoint has no 'reward_head' key — using a randomly-initialised "
            "frozen RewardHead. Re-run Stage 1 with the current code to fix this."
        )

    for model in (encoder, predictor, reward_head):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    return encoder, predictor, reward_head


# ---------------------------------------------------------------------------
# Imagination rollout
# ---------------------------------------------------------------------------

class RolloutMetrics(NamedTuple):
    """Per-step averages collected during one imagination rollout over a batch."""
    loss: torch.Tensor       # differentiable TD loss (mean over steps)
    mean_reward: float       # mean imagined reward per step
    mean_q_std: float        # mean std of Q-values across actions (spread)
    mean_entropy: float      # mean entropy of the empirical action distribution


def imagination_rollout(
    z0: torch.Tensor,
    predictor: ActionConditionedPredictor,
    reward_head: RewardHead,
    q_head: QHead,
    target_q_head: QHead,
    rollout_length: int,
    gamma: float,
    epsilon: float,
) -> RolloutMetrics:
    """Roll out latent trajectories and return TD loss plus diagnostic metrics.

    Action selection uses ε-greedy so early-training random Q-values don't
    collapse the rollout to a single repeated action.

    Episode termination is not tracked inside the imagined rollout (there is no
    ground-truth ``done`` signal in latent space). The rollout therefore
    bootstraps across imagined trajectory boundaries, which is the expected
    behaviour for a world-model-based dreaming loop.

    Args:
        z0: Starting latent states ``(B, D)`` encoded from real observations.
        predictor: Frozen world-model dynamics.
        reward_head: Frozen reward predictor.
        q_head: Online Q-Head being trained.
        target_q_head: Frozen copy of Q-Head providing stable Bellman targets.
        rollout_length: Number of imagination steps.
        gamma: Discount factor.
        epsilon: ε-greedy exploration rate (0 → fully greedy).

    Returns:
        :class:`RolloutMetrics` with the differentiable loss and scalar
        diagnostics averaged across the rollout.
    """
    z = z0
    total_loss = torch.tensor(0.0, device=z.device)
    total_reward = 0.0
    total_q_std = 0.0
    total_entropy = 0.0
    B = z.size(0)

    for _ in range(rollout_length):
        # ----- action selection (ε-greedy, no gradient) -----
        with torch.no_grad():
            q_values = q_head(z)                           # (B, A)
        greedy_actions = q_values.argmax(dim=1)            # (B,)
        if epsilon > 0.0:
            random_actions = torch.randint(
                0, NUM_BREAKOUT_ACTIONS, (B,), device=z.device
            )
            explore_mask = torch.rand(B, device=z.device) < epsilon
            actions = torch.where(explore_mask, random_actions, greedy_actions)
        else:
            actions = greedy_actions

        # ----- diagnostics (no gradient) -----
        q_std = float(q_values.std(dim=1).mean().item())
        action_counts = torch.bincount(actions, minlength=NUM_BREAKOUT_ACTIONS).float()
        action_probs = action_counts / B
        # Entropy of the empirical action distribution over this batch
        entropy = float(-(action_probs * (action_probs + 1e-8).log()).sum().item())

        # ----- world-model step (no gradient through frozen models) -----
        with torch.no_grad():
            z_next = predictor(z, actions)                          # (B, D)
            imagined_reward = reward_head(z, actions).clamp(min=0.0)  # (B,)
            next_q = target_q_head(z_next).max(dim=1).values        # (B,)
            td_target = imagined_reward + gamma * next_q

        # ----- TD loss on the online Q-Head (gradient flows here) -----
        q_taken = q_head(z).gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
        step_loss = F.smooth_l1_loss(q_taken, td_target)
        total_loss = total_loss + step_loss

        total_reward += float(imagined_reward.mean().item())
        total_q_std += q_std
        total_entropy += entropy

        z = z_next.detach()  # advance; block gradients through time

    n = float(rollout_length)
    return RolloutMetrics(
        loss=total_loss / n,
        mean_reward=total_reward / n,
        mean_q_std=total_q_std / n,
        mean_entropy=total_entropy / n,
    )


# ---------------------------------------------------------------------------
# Epoch runner (shared by train and val)
# ---------------------------------------------------------------------------

def run_epoch(
    dataloader: DataLoader,
    encoder: ConvEncoder,
    predictor: ActionConditionedPredictor,
    reward_head: RewardHead,
    q_head: QHead,
    target_q_head: QHead,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    *,
    train: bool,
) -> dict[str, float]:
    """Run one pass over *dataloader* and return averaged metrics.

    When ``train=True`` gradients are computed and the optimizer is stepped.
    When ``train=False`` the function runs under ``torch.no_grad()``.
    """
    q_head.train(train)
    accum = {"loss": 0.0, "reward": 0.0, "q_std": 0.0, "entropy": 0.0}
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in dataloader:
            context = batch["context"].to(args.device)

            # Encode real observations → starting latent states (no gradient
            # flows into the frozen encoder regardless of train mode).
            with torch.no_grad():
                z0 = encoder(context)

            metrics = imagination_rollout(
                z0=z0,
                predictor=predictor,
                reward_head=reward_head,
                q_head=q_head,
                target_q_head=target_q_head,
                rollout_length=args.rollout_length,
                gamma=args.gamma,
                # Exploration only during training; val uses greedy policy.
                epsilon=args.epsilon if train else 0.0,
            )

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                metrics.loss.backward()
                optimizer.step()

            accum["loss"] += float(metrics.loss.item())
            accum["reward"] += metrics.mean_reward
            accum["q_std"] += metrics.mean_q_std
            accum["entropy"] += metrics.mean_entropy
            n_batches += 1

    n = max(n_batches, 1)
    return {k: v / n for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset — split into train and val before creating DataLoaders
    # ------------------------------------------------------------------
    full_dataset = BreakoutTransitionDataset(args.data_dir, context_length=args.context_length)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    # Deterministic split for reproducibility.
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(0)).tolist()
    train_dataset = Subset(full_dataset, indices[:n_train])
    val_dataset = Subset(full_dataset, indices[n_train:])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print(
        f"Dataset: {n_total} samples  →  train={n_train}  val={n_val}  "
        f"({len(train_loader)} / {len(val_loader)} batches)"
    )

    # ------------------------------------------------------------------
    # World model (frozen)
    # ------------------------------------------------------------------
    encoder, predictor, reward_head = load_world_model(
        args.jepa_checkpoint, args.context_length, args.latent_dim, args.device
    )

    # ------------------------------------------------------------------
    # Q-Head + target Q-Head
    # ------------------------------------------------------------------
    q_head = QHead(latent_dim=args.latent_dim, num_actions=NUM_BREAKOUT_ACTIONS).to(args.device)
    target_q_head = copy.deepcopy(q_head)
    target_q_head.eval()
    for p in target_q_head.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(q_head.parameters(), lr=args.learning_rate)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            train_loader, encoder, predictor, reward_head,
            q_head, target_q_head, optimizer, args, train=True,
        )
        val_metrics = run_epoch(
            val_loader, encoder, predictor, reward_head,
            q_head, target_q_head, None, args, train=False,
        )

        # Hard-copy online → target Q-Head every N epochs for stable targets.
        if epoch % args.target_sync_epochs == 0:
            target_q_head.load_state_dict(q_head.state_dict())

        checkpoint_path = args.output_dir / f"q_imagination_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "q_head": q_head.state_dict(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            },
            checkpoint_path,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f}  val_loss={val_metrics['loss']:.6f} | "
            f"imag_reward={train_metrics['reward']:.4f}  "
            f"q_std={train_metrics['q_std']:.4f}  "
            f"entropy={train_metrics['entropy']:.4f} | "
            f"saved={checkpoint_path.name}"
        )


if __name__ == "__main__":
    main()

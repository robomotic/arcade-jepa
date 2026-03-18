"""Train an Actor-Critic policy on Breakout using a frozen JEPA encoder.

The JEPA encoder (ConvEncoder) is loaded from a Stage-1 checkpoint and kept
frozen throughout training.  A small ActorCriticHead (shared trunk → policy
logit head + value head) is trained on-policy using Proximal Policy
Optimisation (PPO) with Generalised Advantage Estimation (GAE).

The environment reward (brick-breaking score) provides the learning signal —
solving the goal-specification problem that intrinsic MPPI could not address.

Usage
-----
    python Breakout/train_actor_critic.py \\
        --encoder-checkpoint Breakout/checkpoints/stage1_run_20260318/jepa_epoch_008.pt \\
        --output-dir Breakout/checkpoints/ac_ppo
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .envs import create_breakout_env
    from .models import ActorCriticHead, ConvEncoder
    from .plan_mppi import PIXEL_NORM_SCALE, load_world_model
except ImportError:
    from envs import create_breakout_env
    from models import ActorCriticHead, ConvEncoder
    from plan_mppi import PIXEL_NORM_SCALE, load_world_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO Actor-Critic on frozen JEPA encoder for Breakout.")

    # World model
    p.add_argument("--encoder-checkpoint", type=Path, required=True,
                   help="Path to a Stage-1 JEPA .pt checkpoint.")
    p.add_argument("--context-length", type=int, default=0,
                   help="Override context (frame-stack) length (0 = read from ckpt).")
    p.add_argument("--latent-dim", type=int, default=0,
                   help="Override latent dim (0 = read from ckpt).")

    # Actor-critic head
    p.add_argument("--hidden-dim", type=int, default=256,
                   help="Hidden-layer width of the ActorCriticHead trunk.")

    # Rollout / PPO
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--rollout-steps", type=int, default=512,
                   help="Number of environment steps collected before each update.")
    p.add_argument("--num-envs", type=int, default=1,
                   help="Parallel environment count (currently serial, kept for API symmetry).")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)

    # PPO update hyper-params
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # Optimiser
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--anneal-lr", action="store_true",
                   help="Linearly anneal LR to 0 over total-timesteps.")

    # Env
    p.add_argument("--frameskip", type=int, default=4)
    p.add_argument("--repeat-action-probability", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    # I/O
    p.add_argument("--output-dir", type=Path, default=Path("Breakout/checkpoints/ac_ppo"))
    p.add_argument("--save-interval-updates", type=int, default=100,
                   help="Save a checkpoint every N PPO update cycles.")
    p.add_argument("--log-interval-updates", type=int, default=10)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Rollout buffer helpers
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-size on-policy rollout buffer with GAE computation."""

    def __init__(self, rollout_steps: int, latent_dim: int, device: str):
        self.rollout_steps = rollout_steps
        self.device = device
        self.latents = torch.zeros(rollout_steps, latent_dim, device=device)
        self.actions = torch.zeros(rollout_steps, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(rollout_steps, device=device)
        self.rewards = torch.zeros(rollout_steps, device=device)
        self.dones = torch.zeros(rollout_steps, device=device)
        self.values = torch.zeros(rollout_steps, device=device)
        self.ptr = 0

    def add(
        self,
        latent: torch.Tensor,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        i = self.ptr
        self.latents[i] = latent.squeeze(0)
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.rollout_steps

    def reset(self) -> None:
        self.ptr = 0

    def compute_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (advantages, returns) using GAE."""
        advantages = torch.zeros(self.rollout_steps, device=self.device)
        last_gae = 0.0
        last_val = float(last_value)
        for t in reversed(range(self.rollout_steps)):
            next_non_terminal = 1.0 - float(self.dones[t])
            next_val = last_val if t == self.rollout_steps - 1 else float(self.values[t + 1])
            delta = float(self.rewards[t]) + gamma * next_val * next_non_terminal - float(self.values[t])
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns


# ---------------------------------------------------------------------------
# Observation → latent helper
# ---------------------------------------------------------------------------

NOOP_ACTION = 0
FIRE_ACTION = 1


def obs_to_latent(
    context_frames: deque[np.ndarray],
    encoder: ConvEncoder,
    device: str,
) -> torch.Tensor:
    stacked = np.stack(list(context_frames), axis=0)  # (C, H, W)
    t = torch.from_numpy(stacked).to(device=device, dtype=torch.float32).unsqueeze(0)
    t = t * PIXEL_NORM_SCALE
    return encoder(t)  # (1, latent_dim)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load frozen JEPA encoder
    context_length_arg = None if args.context_length <= 0 else args.context_length
    latent_dim_arg = None if args.latent_dim <= 0 else args.latent_dim
    encoder, _predictor, context_length, latent_dim = load_world_model(
        args.encoder_checkpoint,
        device=args.device,
        context_length=context_length_arg,
        latent_dim=latent_dim_arg,
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Actor-critic head (the only trainable component)
    ac_head = ActorCriticHead(
        latent_dim=latent_dim,
        num_actions=4,
        hidden_dim=args.hidden_dim,
    ).to(args.device)

    optimiser = torch.optim.Adam(ac_head.parameters(), lr=args.learning_rate, eps=1e-5)

    env = create_breakout_env(
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
        seed=args.seed,
    )

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics_csv = args.output_dir / "train_metrics.csv"
    csv_fields = [
        "run_id", "update", "total_steps",
        "mean_ep_return", "mean_ep_length",
        "policy_loss", "value_loss", "entropy_loss", "total_loss",
        "approx_kl", "clip_frac", "learning_rate",
    ]
    write_header = not metrics_csv.exists()
    csv_f = metrics_csv.open("a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
    if write_header:
        csv_writer.writeheader()

    buffer = RolloutBuffer(args.rollout_steps, latent_dim, args.device)

    # Env state
    raw_obs, _ = env.reset(seed=args.seed)
    obs = np.asarray(raw_obs, dtype=np.uint8)
    context_frames: deque[np.ndarray] = deque(
        [obs.copy() for _ in range(context_length)], maxlen=context_length
    )
    done = False

    ep_return = 0.0
    ep_length = 0
    recent_returns: deque[float] = deque(maxlen=50)
    recent_lengths: deque[int] = deque(maxlen=50)

    total_steps = 0
    update_count = 0
    num_updates = args.total_timesteps // args.rollout_steps

    print(f"Training for {num_updates} updates × {args.rollout_steps} steps = {num_updates * args.rollout_steps} env steps")
    print(f"Encoder frozen: {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"AC head trainable: {sum(p.numel() for p in ac_head.parameters()):,} params")
    sys.stdout.flush()

    for update in range(1, num_updates + 1):
        # Optional linear LR annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            for pg in optimiser.param_groups:
                pg["lr"] = args.learning_rate * frac

        # ---------------------------------------------------------------
        # Collect rollout
        # ---------------------------------------------------------------
        buffer.reset()
        ac_head.eval()

        with torch.no_grad():
            while not buffer.full():
                latent = obs_to_latent(context_frames, encoder, args.device)
                action_t, log_prob_t, _, value_t = ac_head.get_action_and_value(latent)
                action = int(action_t.item())
                log_prob = float(log_prob_t.item())
                value = float(value_t.item())

                raw_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                obs = np.asarray(raw_obs, dtype=np.uint8)
                context_frames.append(obs)

                buffer.add(latent, action, log_prob, float(reward), done, value)

                ep_return += float(reward)
                ep_length += 1
                total_steps += 1

                if done:
                    recent_returns.append(ep_return)
                    recent_lengths.append(ep_length)
                    ep_return = 0.0
                    ep_length = 0
                    raw_obs, _ = env.reset()
                    obs = np.asarray(raw_obs, dtype=np.uint8)
                    context_frames = deque(
                        [obs.copy() for _ in range(context_length)], maxlen=context_length
                    )

            # Bootstrap last value
            latent_last = obs_to_latent(context_frames, encoder, args.device)
            _, _, _, last_value_t = ac_head.get_action_and_value(latent_last)
            last_value = float(last_value_t.item())

        advantages, returns = buffer.compute_advantages(
            last_value, done, args.gamma, args.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---------------------------------------------------------------
        # PPO update
        # ---------------------------------------------------------------
        ac_head.train()
        minibatch_size = args.rollout_steps // args.num_minibatches
        indices = np.arange(args.rollout_steps)

        pg_loss_total = v_loss_total = ent_loss_total = total_loss_total = 0.0
        approx_kl_total = clip_frac_total = 0.0
        num_mb = 0

        for _ in range(args.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, args.rollout_steps, minibatch_size):
                mb_idx = indices[start: start + minibatch_size]
                mb_latents = buffer.latents[mb_idx]
                mb_actions = buffer.actions[mb_idx]
                mb_old_log_probs = buffer.log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                _, new_log_prob, entropy, new_value = ac_head.get_action_and_value(mb_latents, mb_actions)

                log_ratio = new_log_prob - mb_old_log_probs
                ratio = log_ratio.exp()

                # PPO clipped surrogate loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = F.mse_loss(new_value.squeeze(-1), mb_returns)

                # Entropy bonus (negated: we maximise entropy)
                ent_loss = -entropy.mean()

                loss = pg_loss + args.vf_coef * v_loss + args.ent_coef * ent_loss

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac_head.parameters(), args.max_grad_norm)
                optimiser.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

                pg_loss_total += pg_loss.item()
                v_loss_total += v_loss.item()
                ent_loss_total += ent_loss.item()
                total_loss_total += loss.item()
                approx_kl_total += approx_kl
                clip_frac_total += clip_frac
                num_mb += 1

        update_count += 1
        denom = max(num_mb, 1)
        mean_ep_return = float(np.mean(recent_returns)) if recent_returns else 0.0
        mean_ep_length = float(np.mean(recent_lengths)) if recent_lengths else 0.0
        current_lr = optimiser.param_groups[0]["lr"]

        if update % args.log_interval_updates == 0:
            print(
                f"update={update}/{num_updates}  steps={total_steps}  "
                f"mean_return={mean_ep_return:.2f}  "
                f"pg_loss={pg_loss_total/denom:.4f}  "
                f"v_loss={v_loss_total/denom:.4f}  "
                f"approx_kl={approx_kl_total/denom:.4f}  "
                f"lr={current_lr:.2e}"
            )
            sys.stdout.flush()

        csv_writer.writerow({
            "run_id": run_id,
            "update": update,
            "total_steps": total_steps,
            "mean_ep_return": round(mean_ep_return, 4),
            "mean_ep_length": round(mean_ep_length, 1),
            "policy_loss": round(pg_loss_total / denom, 6),
            "value_loss": round(v_loss_total / denom, 6),
            "entropy_loss": round(ent_loss_total / denom, 6),
            "total_loss": round(total_loss_total / denom, 6),
            "approx_kl": round(approx_kl_total / denom, 6),
            "clip_frac": round(clip_frac_total / denom, 4),
            "learning_rate": current_lr,
        })
        csv_f.flush()

        if update % args.save_interval_updates == 0 or update == num_updates:
            ckpt_path = args.output_dir / f"ac_update_{update:06d}.pt"
            torch.save(
                {
                    "update": update,
                    "total_steps": total_steps,
                    "ac_head": ac_head.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "args": vars(args),
                    "context_length": context_length,
                    "latent_dim": latent_dim,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    csv_f.close()
    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()

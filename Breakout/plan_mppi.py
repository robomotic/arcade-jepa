from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from .models import ActionConditionedPredictor, ConvEncoder
except ImportError:
    from models import ActionConditionedPredictor, ConvEncoder


NUM_BREAKOUT_ACTIONS = 4
PIXEL_NORM_SCALE = 1.0 / 255.0


@dataclass(frozen=True)
class MppiPlanResult:
    action: int
    best_score: float
    mean_score: float


def load_world_model(
    encoder_checkpoint: Path,
    *,
    device: str,
    context_length: int | None = None,
    latent_dim: int | None = None,
) -> tuple[ConvEncoder, ActionConditionedPredictor, int, int]:
    checkpoint = torch.load(encoder_checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})

    resolved_context_length = int(context_length or checkpoint_args.get("context_length", 4))
    resolved_latent_dim = int(latent_dim or checkpoint_args.get("latent_dim", 512))

    encoder = ConvEncoder(
        input_channels=resolved_context_length,
        latent_dim=resolved_latent_dim,
    ).to(device)
    predictor = ActionConditionedPredictor(latent_dim=resolved_latent_dim).to(device)

    encoder.load_state_dict(checkpoint["encoder"])
    predictor.load_state_dict(checkpoint["predictor"])

    encoder.eval()
    predictor.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)

    return encoder, predictor, resolved_context_length, resolved_latent_dim


class MppiPlanner:
    def __init__(
        self,
        encoder: ConvEncoder,
        predictor: ActionConditionedPredictor,
        *,
        device: str,
        num_actions: int = NUM_BREAKOUT_ACTIONS,
        num_samples: int = 512,
        horizon: int = 5,
        gamma: float = 0.99,
        temperature: float = 1.0,
        action_switch_penalty: float = 0.02,
        latent_norm_penalty: float = 0.01,
        reuse_best_sequence: bool = True,
        guided_fraction: float = 0.2,
        guided_noise_prob: float = 0.25,
    ):
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")

        self.encoder = encoder
        self.predictor = predictor
        self.device = device
        self.num_actions = num_actions
        self.num_samples = num_samples
        self.horizon = horizon
        self.gamma = gamma
        self.temperature = temperature
        self.action_switch_penalty = action_switch_penalty
        self.latent_norm_penalty = latent_norm_penalty
        self.reuse_best_sequence = reuse_best_sequence
        self.guided_fraction = guided_fraction
        self.guided_noise_prob = guided_noise_prob

        self._best_sequence: torch.Tensor | None = None

    def _sample_action_sequences(self) -> torch.Tensor:
        sequences = torch.randint(
            low=0,
            high=self.num_actions,
            size=(self.num_samples, self.horizon),
            device=self.device,
        )

        if self.reuse_best_sequence and self._best_sequence is not None:
            guided_count = int(self.num_samples * self.guided_fraction)
            if guided_count > 0:
                guided = self._best_sequence.unsqueeze(0).repeat(guided_count, 1)
                random_actions = torch.randint(
                    low=0,
                    high=self.num_actions,
                    size=(guided_count, self.horizon),
                    device=self.device,
                )
                noise_mask = torch.rand((guided_count, self.horizon), device=self.device) < self.guided_noise_prob
                guided = torch.where(noise_mask, random_actions, guided)
                sequences[:guided_count] = guided

        return sequences

    @torch.no_grad()
    def plan_tensor(self, context_tensor: torch.Tensor, step_index: int = 0) -> MppiPlanResult:
        if context_tensor.ndim == 3:
            context_batch = context_tensor.unsqueeze(0)
        elif context_tensor.ndim == 4:
            context_batch = context_tensor
        else:
            raise ValueError(f"Expected context tensor rank 3 or 4, got {context_tensor.shape}")

        context_batch = context_batch.to(self.device, dtype=torch.float32)
        latent = self.encoder(context_batch)
        latent = latent.repeat(self.num_samples, 1)

        action_sequences = self._sample_action_sequences()
        scores = torch.zeros(self.num_samples, device=self.device)
        prev_actions: torch.Tensor | None = None
        discount = 1.0

        for h in range(self.horizon):
            actions = action_sequences[:, h]
            next_latent = self.predictor(latent, actions)

            latent_step = torch.linalg.vector_norm(next_latent - latent, ord=2, dim=1)
            latent_norm = torch.linalg.vector_norm(next_latent, ord=2, dim=1)

            if prev_actions is None:
                switch_penalty = torch.zeros_like(latent_step)
            else:
                switch_penalty = (actions != prev_actions).to(latent_step.dtype)

            step_score = latent_step
            step_score = step_score - self.latent_norm_penalty * latent_norm
            step_score = step_score - self.action_switch_penalty * switch_penalty
            scores = scores + discount * step_score

            latent = next_latent
            prev_actions = actions
            discount *= self.gamma

        best_index = int(torch.argmax(scores).item())
        best_sequence = action_sequences[best_index]

        if self.reuse_best_sequence:
            self._best_sequence = torch.cat((best_sequence[1:], best_sequence[-1:]), dim=0)

        if self.temperature > 0.0:
            scaled_scores = scores / self.temperature
            probs = torch.softmax(scaled_scores, dim=0)
            first_action_probs = torch.zeros(self.num_actions, device=self.device)
            first_actions = action_sequences[:, 0]
            first_action_probs.scatter_add_(0, first_actions, probs)
            chosen_action = int(torch.argmax(first_action_probs).item())
        else:
            chosen_action = int(best_sequence[0].item())

        return MppiPlanResult(
            action=chosen_action,
            best_score=float(scores[best_index].item()),
            mean_score=float(scores.mean().item()),
        )
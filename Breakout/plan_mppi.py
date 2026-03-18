from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from .models import ActionConditionedPredictor, ConvEncoder
except ImportError:
    from models import ActionConditionedPredictor, ConvEncoder


NUM_BREAKOUT_ACTIONS = 4
PIXEL_NORM_SCALE = 1.0 / 255.0


@dataclass
class MppiPlanResult:
    action: int
    best_score: float
    mean_score: float
    scores: np.ndarray
    action_sequences: np.ndarray


def build_context_tensor(context_frames: deque[np.ndarray], device: str) -> torch.Tensor:
    stacked = np.stack(list(context_frames), axis=0).astype(np.float32) * PIXEL_NORM_SCALE
    return torch.from_numpy(stacked).unsqueeze(0).to(device)


def load_world_model(
    checkpoint_path: Path,
    context_length: int,
    latent_dim: int,
    device: str,
) -> tuple[ConvEncoder, ActionConditionedPredictor]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder = ConvEncoder(input_channels=context_length, latent_dim=latent_dim).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    predictor = ActionConditionedPredictor(latent_dim=latent_dim, num_actions=NUM_BREAKOUT_ACTIONS).to(device)
    predictor.load_state_dict(checkpoint["predictor"])
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)

    return encoder, predictor


class MppiPlanner:
    def __init__(
        self,
        encoder: ConvEncoder,
        predictor: ActionConditionedPredictor,
        *,
        num_actions: int = NUM_BREAKOUT_ACTIONS,
        num_samples: int = 512,
        horizon: int = 5,
        gamma: float = 0.99,
        device: str = "cpu",
        seed: int = 42,
        force_fire_on_reset: bool = True,
        reuse_best_sequence: bool = True,
    ) -> None:
        self.encoder = encoder
        self.predictor = predictor
        self.num_actions = num_actions
        self.num_samples = num_samples
        self.horizon = horizon
        self.gamma = gamma
        self.device = device
        self.force_fire_on_reset = force_fire_on_reset
        self.reuse_best_sequence = reuse_best_sequence
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.previous_best_sequence: torch.Tensor | None = None

    def reset_episode(self) -> None:
        self.previous_best_sequence = None

    def _sample_action_sequences(self) -> torch.Tensor:
        sequences = torch.randint(
            low=0,
            high=self.num_actions,
            size=(self.num_samples, self.horizon),
            generator=self.generator,
            dtype=torch.long,
        )
        if self.reuse_best_sequence and self.previous_best_sequence is not None and self.horizon > 0:
            next_tail = torch.randint(
                low=0,
                high=self.num_actions,
                size=(1,),
                generator=self.generator,
                dtype=torch.long,
            )
            carry = torch.cat((self.previous_best_sequence[1:].clone(), next_tail), dim=0)
            sequences[0] = carry
        return sequences.to(self.device)

    def _score_trajectories(self, start_latent: torch.Tensor, action_sequences: torch.Tensor) -> torch.Tensor:
        latents = start_latent.expand(action_sequences.size(0), -1)
        scores = torch.zeros(action_sequences.size(0), device=self.device)
        discount = 1.0

        for step_idx in range(self.horizon):
            next_latents = self.predictor(latents, action_sequences[:, step_idx])
            step_curiosity = torch.linalg.vector_norm(next_latents - latents, ord=2, dim=1)
            scores += discount * step_curiosity
            latents = next_latents
            discount *= self.gamma

        return scores

    @torch.no_grad()
    def plan_tensor(self, context_tensor: torch.Tensor, step_index: int) -> MppiPlanResult:
        if self.force_fire_on_reset and step_index == 1:
            action_sequences = np.full((1, self.horizon), 1, dtype=np.int64)
            return MppiPlanResult(
                action=1,
                best_score=0.0,
                mean_score=0.0,
                scores=np.zeros(1, dtype=np.float32),
                action_sequences=action_sequences,
            )

        start_latent = self.encoder(context_tensor)
        action_sequences = self._sample_action_sequences()
        scores = self._score_trajectories(start_latent, action_sequences)

        best_index = int(scores.argmax().item())
        best_sequence = action_sequences[best_index].detach().cpu()
        self.previous_best_sequence = best_sequence.clone()

        return MppiPlanResult(
            action=int(best_sequence[0].item()),
            best_score=float(scores[best_index].item()),
            mean_score=float(scores.mean().item()),
            scores=scores.detach().cpu().numpy(),
            action_sequences=action_sequences.detach().cpu().numpy(),
        )

    @torch.no_grad()
    def plan(self, context_frames: deque[np.ndarray], step_index: int) -> int:
        context_tensor = build_context_tensor(context_frames, self.device)
        return self.plan_tensor(context_tensor, step_index).action

    @torch.no_grad()
    def plan_debug(self, context_frames: deque[np.ndarray], step_index: int) -> MppiPlanResult:
        context_tensor = build_context_tensor(context_frames, self.device)
        return self.plan_tensor(context_tensor, step_index)
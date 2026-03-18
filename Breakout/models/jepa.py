from __future__ import annotations

import torch
from torch import nn


class ConvEncoder(nn.Module):
    """CNN + MLP encoder that maps a grayscale frame stack to a latent vector.

    Expected input: float32 tensor of shape ``(B, input_channels, 84, 84)``
    with pixel values in ``[0.0, 1.0]``.

    Output: ``LayerNorm``-normalised latent vector of shape ``(B, latent_dim)``.
    The ``LayerNorm`` at the final layer keeps latent magnitudes stable
    throughout training, which is critical for the SmoothL1 JEPA loss
    ``‖ẑ_{t+1} − z̄_{t+1}‖`` to remain well-conditioned.
    """

    def __init__(self, input_channels: int = 4, latent_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
            # Normalise the latent vector so its scale stays bounded.
            # Without this the encoder can minimise JEPA loss by collapsing
            # or inflating latent magnitudes rather than learning dynamics.
            nn.LayerNorm(latent_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Args:
            observations: float32 ``(B, C, 84, 84)`` in ``[0.0, 1.0]``.
        Returns:
            LayerNorm-normalised latent ``(B, latent_dim)``.
        """
        return self.network(observations)


class ActionConditionedPredictor(nn.Module):
    def __init__(self, latent_dim: int = 512, num_actions: int = 4, hidden_dim: int = 512):
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, latent_dim)
        self.network = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_latent = self.action_embedding(action)
        combined = torch.cat((latent_state, action_latent), dim=-1)
        return self.network(combined)

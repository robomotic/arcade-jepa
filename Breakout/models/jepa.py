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


class QHead(nn.Module):
    """Q-value head for DQN-style policy learning on frozen JEPA latents.

    Expected input: latent vector ``(B, latent_dim)`` from a frozen ``ConvEncoder``.
    Output: Q-values ``(B, num_actions)`` — one scalar per discrete action.

    Training signal: 1-step Bellman targets from stored or imagined transitions:
        ``Q(z_t, a_t) ← r_t + γ · max_{a'} Q(z_{t+1}, a')``

    This is strictly better than behaviour cloning on random-policy data,
    which would simply train the agent to reproduce random actions.
    """

    def __init__(self, latent_dim: int = 512, num_actions: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, num_actions),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Args:
            latent: ``(B, latent_dim)`` encoder output.
        Returns:
            Q-values ``(B, num_actions)``.
        """
        return self.network(latent)


class RewardHead(nn.Module):
    """Auxiliary head that predicts immediate reward from (z_t, a_t).

    Trained alongside the JEPA encoder in Stage 1 as an auxiliary loss.
    Storing this head in the JEPA checkpoint enriches the world model with
    a reward-aware signal, enabling Stage 1.5 latent imagination without
    any real environment interaction.
    """

    def __init__(self, latent_dim: int = 512, num_actions: int = 4):
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, latent_dim)
        self.network = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Args:
            latent: ``(B, latent_dim)`` encoder output.
            action: ``(B,)`` long tensor of discrete action indices.
        Returns:
            Predicted reward ``(B,)``.
        """
        action_emb = self.action_embedding(action)
        combined = torch.cat((latent, action_emb), dim=-1)
        return self.network(combined).squeeze(-1)

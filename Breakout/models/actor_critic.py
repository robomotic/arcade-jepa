from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCriticHead(nn.Module):
    """Policy + value head that sits on top of a frozen JEPA encoder.

    Input: latent vector ``(B, latent_dim)`` produced by :class:`ConvEncoder`.
    Outputs: action logits ``(B, num_actions)`` and state-value ``(B, 1)``.

    A shared two-layer MLP trunk feeds both heads to encourage representation
    reuse while keeping the policy expressive enough to learn from sparse
    Breakout rewards.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_actions: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Orthogonal initialisation — standard for PPO.
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2**0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent: float32 ``(B, latent_dim)``.
        Returns:
            logits: ``(B, num_actions)``
            value:  ``(B, 1)``
        """
        trunk_out = self.trunk(latent)
        return self.policy_head(trunk_out), self.value_head(trunk_out)

    def get_action_and_value(
        self,
        latent: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) an action and compute log-prob + entropy.

        Args:
            latent: float32 ``(B, latent_dim)``.
            action: optional int64 ``(B,)`` — if given, evaluates its log-prob
                    instead of sampling a new action.
        Returns:
            action:   int64 ``(B,)``
            log_prob: float32 ``(B,)``
            entropy:  float32 ``(B,)``
            value:    float32 ``(B, 1)``
        """
        logits, value = self.forward(latent)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

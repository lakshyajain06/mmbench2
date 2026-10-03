"""Small action-chunking policy used as the first BC baseline."""

from __future__ import annotations

import torch
from torch import nn


class ChunkedBCPolicy(nn.Module):
    """Encode a short state history and predict a continuous action chunk."""

    def __init__(
        self,
        observation_dim: int,
        *,
        num_tasks: int,
        context_length: int,
        chunk_size: int,
        action_dim: int = 16,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.observation_dim = int(observation_dim)
        self.num_tasks = int(num_tasks)
        self.context_length = int(context_length)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)

        self.observation_projection = nn.Sequential(
            nn.Linear(self.observation_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, context_length, d_model))
        self.task_embedding = nn.Embedding(num_tasks, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, chunk_size * action_dim)

        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.task_embedding.weight, std=0.02)
        nn.init.normal_(self.action_head.weight, std=0.01)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, observation: torch.Tensor, task_index: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 3:
            raise ValueError("observation must have shape (batch, context, features)")
        if observation.shape[1:] != (self.context_length, self.observation_dim):
            raise ValueError(
                f"Expected observation tail {(self.context_length, self.observation_dim)}, "
                f"got {tuple(observation.shape[1:])}"
            )
        hidden = self.observation_projection(observation)
        hidden = hidden + self.position_embedding + self.task_embedding(task_index)[:, None]
        hidden = self.transformer(hidden)
        action = self.action_head(self.output_norm(hidden[:, -1]))
        return torch.tanh(action.view(-1, self.chunk_size, self.action_dim))


def masked_action_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if not (prediction.shape == target.shape == mask.shape):
        raise ValueError("prediction, target, and mask must have identical shapes")
    squared_error = (prediction.float() - target.float()).square()
    weights = mask.to(squared_error.dtype)
    return (squared_error * weights).sum() / weights.sum().clamp_min(1.0)

"""Dense-token visual action-chunk policy."""

from __future__ import annotations

import torch
from torch import nn


def block_causal_mask(context_length: int, tokens_per_frame: int) -> torch.Tensor:
    """Allow all spatial attention within a frame and only past frames across time."""
    time = torch.arange(context_length).repeat_interleave(tokens_per_frame)
    return time[None, :] > time[:, None]


class DenseVisualChunkPolicy(nn.Module):
    """Predict action chunks from frozen dense visual tokens.

    The vision encoder is deliberately external and frozen. Inputs have shape
    ``(B, T, P, D_visual)`` where P spatial tokens are retained per frame.
    """

    def __init__(
        self,
        visual_dim: int,
        tokens_per_frame: int,
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
        self.visual_dim = int(visual_dim)
        self.tokens_per_frame = int(tokens_per_frame)
        self.num_tasks = int(num_tasks)
        self.context_length = int(context_length)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)

        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.spatial_embedding = nn.Parameter(torch.zeros(1, 1, tokens_per_frame, d_model))
        self.temporal_embedding = nn.Parameter(torch.zeros(1, context_length, 1, d_model))
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
        self.register_buffer(
            "attention_mask",
            block_causal_mask(context_length, tokens_per_frame),
            persistent=False,
        )
        nn.init.normal_(self.spatial_embedding, std=0.02)
        nn.init.normal_(self.temporal_embedding, std=0.02)
        nn.init.normal_(self.task_embedding.weight, std=0.02)
        nn.init.normal_(self.action_head.weight, std=0.01)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, visual_tokens: torch.Tensor, task_index: torch.Tensor) -> torch.Tensor:
        expected = (self.context_length, self.tokens_per_frame, self.visual_dim)
        if visual_tokens.ndim != 4 or tuple(visual_tokens.shape[1:]) != expected:
            raise ValueError(f"Expected visual tokens (B, {expected}), got {tuple(visual_tokens.shape)}")
        hidden = self.visual_projection(visual_tokens)
        hidden = (
            hidden
            + self.spatial_embedding
            + self.temporal_embedding
            + self.task_embedding(task_index)[:, None, None]
        )
        batch = hidden.shape[0]
        hidden = hidden.reshape(batch, self.context_length * self.tokens_per_frame, -1)
        hidden = self.transformer(hidden, mask=self.attention_mask)
        final_frame = hidden[:, -self.tokens_per_frame :].mean(dim=1)
        action = self.action_head(self.output_norm(final_frame))
        return torch.tanh(action.view(batch, self.chunk_size, self.action_dim))

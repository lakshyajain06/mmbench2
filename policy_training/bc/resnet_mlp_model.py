"""ResNet + MLP visual behavior-cloning policy."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNetMLPPolicy(nn.Module):
    """Predict future actions from an RGB observation history and task identity."""

    def __init__(
        self,
        *,
        num_tasks: int,
        context_length: int = 1,
        chunk_size: int = 8,
        action_dim: int = 16,
        hidden_dim: int = 512,
        task_embedding_dim: int = 64,
        mlp_layers: int = 3,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        if min(
            num_tasks,
            context_length,
            chunk_size,
            action_dim,
            hidden_dim,
            task_embedding_dim,
            mlp_layers,
        ) < 1:
            raise ValueError("All architecture dimensions must be positive")
        self.num_tasks = int(num_tasks)
        self.context_length = int(context_length)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.task_embedding_dim = int(task_embedding_dim)
        self.mlp_layers = int(mlp_layers)
        self.dropout = float(dropout)
        self.pretrained_backbone = bool(pretrained_backbone)

        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_dim)

        layers: list[nn.Module] = []
        # ResNet is shared across time. Concatenating its per-frame features
        # preserves both temporal order and information from every observation.
        input_dim = 512 * context_length + task_embedding_dim
        for _ in range(mlp_layers):
            layers.extend(
                (
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

        nn.init.normal_(self.task_embedding.weight, std=0.02)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, frames: torch.Tensor, task_index: torch.Tensor) -> dict[str, torch.Tensor]:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape (batch, context, 3, height, width)")
        if frames.shape[1] != self.context_length:
            raise ValueError(
                f"Expected {self.context_length} observation frames, got {frames.shape[1]}"
            )
        batch, context, channels, height, width = frames.shape
        images = frames.reshape(batch * context, channels, height, width).float().div(255.0)
        images = (images - self.image_mean) / self.image_std
        visual = self.backbone(images).reshape(batch, context * 512)
        features = torch.cat((visual, self.task_embedding(task_index)), dim=-1)
        actions = torch.tanh(self.mlp(features))
        return {"actions": actions.view(-1, self.chunk_size, self.action_dim)}


def resnet_mlp_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    action_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = output["actions"]
    if prediction.shape != target.shape or target.shape != action_mask.shape:
        raise ValueError("Prediction, target, and mask shapes must match")
    weights = action_mask.to(target.dtype)
    mse = ((prediction.float() - target.float()).square() * weights).sum()
    mse = mse / weights.sum().clamp_min(1.0)
    return mse, {"reconstruction": mse.detach(), "kl": mse.detach().new_zeros(())}

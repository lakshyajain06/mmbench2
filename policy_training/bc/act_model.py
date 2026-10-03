"""A compact visual Action Chunking with Transformers policy."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Features(nn.Module):
    """ResNet-18 through layer4, preserving its spatial feature map."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        network = resnet18(weights=weights)
        self.layers = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
            network.layer3,
            network.layer4,
        )
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float().div(255.0)
        image = (image - self.image_mean) / self.image_std
        return self.layers(image)


class SimpleACTPolicy(nn.Module):
    """Visual ACT with a ResNet-18 backbone and a conditional VAE.

    During training, ``actions`` conditions a posterior over the style latent.
    During inference, omit ``actions`` and the prior mean ``z=0`` is used.
    Only the newest frame in the supplied observation history is consumed.
    """

    def __init__(
        self,
        *,
        num_tasks: int,
        chunk_size: int = 8,
        action_dim: int = 16,
        d_model: int = 256,
        n_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 4,
        latent_dim: int = 32,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.num_tasks = int(num_tasks)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)
        self.pretrained_backbone = bool(pretrained_backbone)

        self.backbone = ResNet18Features(pretrained=pretrained_backbone)
        self.visual_projection = nn.Conv2d(512, d_model, kernel_size=1)
        self.visual_position = nn.Parameter(torch.zeros(1, d_model, 7, 7))
        self.task_embedding = nn.Embedding(num_tasks, d_model)

        # Posterior q(z | task, demonstrated action chunk).
        self.posterior_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.posterior_position = nn.Parameter(torch.zeros(1, chunk_size + 1, d_model))
        self.posterior_action = nn.Linear(action_dim, d_model)
        posterior_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.posterior = nn.TransformerEncoder(posterior_layer, num_layers=encoder_layers)
        self.posterior_stats = nn.Linear(d_model, 2 * latent_dim)
        self.latent_projection = nn.Linear(latent_dim, d_model)

        # Transformer action decoder.
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, action_dim)

        for parameter in (
            self.visual_position,
            self.posterior_cls,
            self.posterior_position,
            self.action_queries,
        ):
            nn.init.normal_(parameter, std=0.02)
        nn.init.normal_(self.task_embedding.weight, std=0.02)
        nn.init.normal_(self.action_head.weight, std=0.01)
        nn.init.zeros_(self.action_head.bias)

    def encode_posterior(
        self, actions: torch.Tensor, task_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = actions.shape[0]
        task = self.task_embedding(task_index)
        cls = self.posterior_cls.expand(batch, -1, -1) + task[:, None]
        action_tokens = self.posterior_action(actions) + task[:, None]
        tokens = torch.cat((cls, action_tokens), dim=1) + self.posterior_position
        encoded = self.posterior(tokens)
        mean, log_variance = self.posterior_stats(encoded[:, 0]).chunk(2, dim=-1)
        log_variance = log_variance.clamp(-10.0, 10.0)
        noise = torch.randn_like(mean)
        latent = mean + noise * torch.exp(0.5 * log_variance)
        return latent, mean, log_variance

    def forward(
        self,
        frames: torch.Tensor,
        task_index: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape (batch, context, 3, height, width)")
        batch = frames.shape[0]
        image = frames[:, -1]
        feature_map = self.visual_projection(self.backbone(image))
        height, width = feature_map.shape[-2:]
        position = F.interpolate(
            self.visual_position, size=(height, width), mode="bilinear", align_corners=False
        )
        memory = (feature_map + position).flatten(2).transpose(1, 2)
        task = self.task_embedding(task_index)

        if actions is None:
            latent = torch.zeros(batch, self.latent_dim, device=frames.device, dtype=memory.dtype)
            mean = log_variance = None
        else:
            if tuple(actions.shape[1:]) != (self.chunk_size, self.action_dim):
                raise ValueError("actions have the wrong chunk or action dimension")
            latent, mean, log_variance = self.encode_posterior(actions, task_index)
        condition = task + self.latent_projection(latent)
        memory = memory + condition[:, None]
        queries = self.action_queries.expand(batch, -1, -1) + task[:, None]
        decoded = self.decoder(queries, memory)
        predicted_actions = torch.tanh(self.action_head(self.output_norm(decoded)))
        return {
            "actions": predicted_actions,
            "posterior_mean": mean,
            "posterior_log_variance": log_variance,
        }


def act_loss(
    output: dict[str, torch.Tensor | None],
    target: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = output["actions"]
    if prediction is None or prediction.shape != target.shape or target.shape != action_mask.shape:
        raise ValueError("ACT prediction, target, and mask shapes must match")
    weights = action_mask.to(dtype=target.dtype)
    reconstruction = ((prediction.float() - target.float()).square() * weights).sum()
    reconstruction = reconstruction / weights.sum().clamp_min(1.0)
    mean = output["posterior_mean"]
    log_variance = output["posterior_log_variance"]
    if mean is None or log_variance is None:
        kl = reconstruction.new_zeros(())
    else:
        kl = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(dim=-1).mean()
    total = reconstruction + float(kl_weight) * kl
    return total, {"reconstruction": reconstruction.detach(), "kl": kl.detach()}

"""skrl components for visual PPO initialized from MMBench behavior cloning."""

from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from skrl.agents.torch.ppo import PPO
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from torchvision.models import ResNet18_Weights, resnet18

from .env import make_rgb_env


class _VisualModel(Model):
    """Common image conversion and ImageNet-normalized ResNet feature trunk."""

    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str | torch.device,
        image_size: int,
        context_length: int,
        pretrained_backbone: bool,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        self.image_size = int(image_size)
        self.context_length = int(context_length)
        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = torch.nn.Identity()
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _images(self, observations: torch.Tensor) -> torch.Tensor:
        expected = self.context_length * 3 * self.image_size * self.image_size
        if observations.numel() % expected:
            raise ValueError(
                f"Observation with shape {tuple(observations.shape)} cannot be reshaped "
                f"to {self.context_length} RGB {self.image_size}x{self.image_size} frames"
            )
        images = observations.reshape(
            -1, self.context_length, 3, self.image_size, self.image_size
        ).float()
        images = images.div(255.0)
        images = (images - self.image_mean.unsqueeze(1)) / self.image_std.unsqueeze(1)
        return images

    def _visual(self, observations: torch.Tensor) -> torch.Tensor:
        images = self._images(observations)
        batch, context, channels, height, width = images.shape
        features = self.backbone(
            images.reshape(batch * context, channels, height, width)
        )
        return features.reshape(batch, context * 512)

    def train(self, mode: bool = True):
        super().train(mode)
        # Rollouts run in evaluation mode while PPO updates run in training
        # mode. Frozen BatchNorm statistics keep both policies identical before
        # the optimizer step, avoiding false KL spikes.
        for module in self.backbone.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self


class ResNetGaussianPolicy(GaussianMixin, _VisualModel):
    """Gaussian PPO actor matching the complete ResNet-MLP BC actor."""

    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str | torch.device,
        image_size: int = 64,
        context_length: int = 1,
        hidden_dim: int = 512,
        task_embedding_dim: int = 64,
        mlp_layers: int = 3,
        initial_log_std: float = -1.0,
        pretrained_backbone: bool = True,
    ) -> None:
        _VisualModel.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            image_size=image_size,
            context_length=context_length,
            pretrained_backbone=pretrained_backbone,
        )
        GaussianMixin.__init__(
            self,
            # PPO stores and reevaluates the sampled Gaussian action. Clipping
            # happens only at the environment boundary so its log probability
            # remains the probability of the action that was actually sampled.
            clip_actions=False,
            clip_mean_actions=True,
            min_log_std=-5.0,
            max_log_std=1.0,
            reduction="sum",
        )
        self.hidden_dim = int(hidden_dim)
        self.task_embedding_dim = int(task_embedding_dim)
        self.mlp_layers = int(mlp_layers)
        self.task_embedding = torch.nn.Parameter(torch.zeros(task_embedding_dim))
        layers: list[torch.nn.Module] = []
        input_dim = 512 * context_length + task_embedding_dim
        for _ in range(mlp_layers):
            layers.extend(
                (
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.GELU(),
                    # BC dropout is deliberately omitted for on-policy
                    # rollout/update log-probability consistency.
                    torch.nn.Identity(),
                )
            )
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, self.num_actions))
        self.mlp = torch.nn.Sequential(*layers)
        self.log_std_parameter = torch.nn.Parameter(
            torch.full((self.num_actions,), float(initial_log_std))
        )

    def _mean(self, observations: torch.Tensor) -> torch.Tensor:
        visual = self._visual(observations)
        task = self.task_embedding.unsqueeze(0).expand(visual.shape[0], -1)
        return torch.tanh(self.mlp(torch.cat((visual, task), dim=-1)))

    def compute(self, inputs: dict[str, Any], role: str = ""):
        return self._mean(inputs["observations"]), {"log_std": self.log_std_parameter}

    @torch.inference_mode()
    def predict(self, observation: np.ndarray | torch.Tensor) -> np.ndarray:
        tensor = torch.as_tensor(observation, device=self.device)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return self._mean(tensor)[0].cpu().numpy()


class ResNetValue(DeterministicMixin, _VisualModel):
    """Value model with a separate visual trunk from the pretrained actor."""

    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str | torch.device,
        image_size: int = 64,
        context_length: int = 1,
        hidden_dim: int = 512,
        pretrained_backbone: bool = True,
    ) -> None:
        _VisualModel.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            image_size=image_size,
            context_length=context_length,
            pretrained_backbone=pretrained_backbone,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        self.value = torch.nn.Sequential(
            torch.nn.Linear(512 * context_length, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def compute(self, inputs: dict[str, Any], role: str = ""):
        visual = self._visual(inputs["observations"])
        return self.value(visual), {}


def load_bc_policy_metadata(checkpoint_path: str | Path, task: str) -> tuple[dict, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("policy_type") != "resnet_mlp_visual_bc":
        raise ValueError("PPO initialization requires a resnet_mlp_visual_bc checkpoint")
    if task not in checkpoint["tasks"]:
        raise ValueError(f"Task {task!r} is not represented in the BC checkpoint")
    return checkpoint, checkpoint["tasks"].index(task)


def initialize_policy_from_bc(
    policy: ResNetGaussianPolicy, checkpoint: dict, task_index: int
) -> None:
    """Restore the visual trunk, task embedding, and first BC chunk action."""
    state = checkpoint["model"]
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    missing, unexpected = policy.backbone.load_state_dict(backbone_state, strict=False)
    if missing or unexpected:
        raise ValueError(f"BC backbone mismatch: missing={missing}, unexpected={unexpected}")
    mlp_layers = int(checkpoint["model_config"]["mlp_layers"])
    with torch.no_grad():
        policy.task_embedding.copy_(state["task_embedding.weight"][task_index])
        for index in range(mlp_layers * 4):
            target = policy.mlp[index]
            for name in ("weight", "bias"):
                key = f"mlp.{index}.{name}"
                if key in state:
                    getattr(target, name).copy_(state[key])
        output_index = mlp_layers * 4
        policy.mlp[output_index].weight.copy_(
            state[f"mlp.{output_index}.weight"][: policy.num_actions]
        )
        policy.mlp[output_index].bias.copy_(
            state[f"mlp.{output_index}.bias"][: policy.num_actions]
        )


def save_portable_checkpoint(
    path: str | Path,
    policy: ResNetGaussianPolicy,
    value: ResNetValue | None,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "mmbench2_skrl_visual_ppo_v1",
            "policy": policy.state_dict(),
            "value": value.state_dict() if value is not None else None,
            "metadata": metadata,
        },
        path,
    )


def load_portable_policy(
    path: str | Path, *, device: str | torch.device
) -> tuple[ResNetGaussianPolicy, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "mmbench2_skrl_visual_ppo_v1":
        raise ValueError("Not an MMBench2 skrl visual PPO checkpoint")
    metadata = checkpoint["metadata"]
    action_dim = int(metadata["action_dim"])
    image_size = int(metadata["image_size"])
    context_length = int(metadata.get("context_length", 1))
    observation_shape = (
        (3, image_size, image_size)
        if context_length == 1
        else (context_length, 3, image_size, image_size)
    )
    policy = ResNetGaussianPolicy(
        observation_space=gym.spaces.Box(0, 255, observation_shape, dtype=np.uint8),
        action_space=gym.spaces.Box(-1, 1, (action_dim,), dtype=np.float32),
        device=device,
        image_size=image_size,
        context_length=context_length,
        hidden_dim=int(metadata["hidden_dim"]),
        task_embedding_dim=int(metadata["task_embedding_dim"]),
        mlp_layers=int(metadata["mlp_layers"]),
        pretrained_backbone=False,
    )
    policy.load_state_dict(checkpoint["policy"])
    policy.to(device).eval()
    return policy, metadata


def _video_frame(observation: np.ndarray | torch.Tensor) -> np.ndarray:
    frame = torch.as_tensor(observation).detach().cpu().numpy()
    if frame.ndim != 3:
        raise ValueError(f"Expected a 3-D RGB frame, got {frame.shape}")
    if frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    return frame.astype(np.uint8, copy=False)


def evaluate_ppo(
    policy: ResNetGaussianPolicy,
    task: str,
    *,
    episodes: int,
    seed: int,
    image_size: int,
    output_dir: str | Path | None = None,
    video_episodes: int = 0,
    video_fps: int = 20,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0 <= video_episodes <= episodes:
        raise ValueError("video_episodes must be between zero and episodes")
    evaluation_dir = Path(output_dir) if output_dir is not None else None
    video_dir = evaluation_dir / "videos" if evaluation_dir is not None else None
    if video_episodes and video_dir is None:
        raise ValueError("output_dir is required when recording videos")
    if evaluation_dir is not None:
        evaluation_dir.mkdir(parents=True, exist_ok=True)
    if video_episodes:
        assert video_dir is not None
        video_dir.mkdir(parents=True, exist_ok=True)

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_rgb_env(task, seed=seed, image_size=image_size)
    successes_once: list[float] = []
    successes_at_end: list[float] = []
    returns: list[float] = []
    lengths: list[int] = []
    video_paths: list[str] = []
    policy.eval()
    context_length = int(getattr(policy, "context_length", 1))
    try:
        for episode_index in range(episodes):
            observation, info = env.reset()
            history = deque(
                (torch.as_tensor(observation).clone() for _ in range(context_length)),
                maxlen=context_length,
            )
            writer = None
            if episode_index < video_episodes:
                assert video_dir is not None
                video_path = video_dir / f"episode_{episode_index:03d}.mp4"
                writer = imageio.get_writer(video_path, fps=video_fps, codec="libx264")
                writer.append_data(_video_frame(observation))
                video_paths.append(str(video_path))
            success_once = float(np.asarray(info.get("success", 0.0)).reshape(-1)[0])
            success_at_end = success_once
            episode_return = 0.0
            length = 0
            done = False
            try:
                while not done:
                    policy_observation = (
                        history[-1] if context_length == 1 else torch.stack(tuple(history))
                    )
                    action = policy.predict(policy_observation)
                    observation, reward, terminated, truncated, info = env.step(action)
                    history.append(torch.as_tensor(observation).clone())
                    if writer is not None:
                        writer.append_data(_video_frame(observation))
                    episode_return += float(reward)
                    length += 1
                    success_at_end = float(
                        np.asarray(info.get("success", 0.0)).reshape(-1)[0]
                    )
                    success_once = max(success_once, success_at_end)
                    done = bool(terminated or truncated)
            finally:
                if writer is not None:
                    writer.close()
            successes_once.append(float(success_once > 0))
            successes_at_end.append(float(success_at_end > 0))
            returns.append(episode_return)
            lengths.append(length)
    finally:
        env.close()
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    metrics = {
        "episodes": episodes,
        # Keep success_rate as the historical success-once alias.
        "success_rate": float(np.mean(successes_once)),
        "success_once_rate": float(np.mean(successes_once)),
        "success_at_end_rate": float(np.mean(successes_at_end)),
        "mean_return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "videos": video_paths,
    }
    if evaluation_dir is not None:
        (evaluation_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


class EvaluationPPO(PPO):
    """PPO with a generic periodic hook expressed in environment samples."""

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor | None,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        bootstrap_observations = final_observations_for_bootstrap(
            next_observations, truncated, infos
        )
        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=bootstrap_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )
        # The trainer must continue from the reset observations. Only the
        # one-step timeout value above should use terminal observations.
        if self.training:
            self._current_next_observations = next_observations

    def set_periodic_hook(
        self, hook: Callable[["EvaluationPPO", int], None] | None, *, num_envs: int
    ) -> None:
        self._periodic_hook = hook
        self._hook_num_envs = int(num_envs)

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        super().post_interaction(timestep=timestep, timesteps=timesteps)
        hook = getattr(self, "_periodic_hook", None)
        if hook is not None:
            hook(self, (timestep + 1) * self._hook_num_envs)


def final_observations_for_bootstrap(
    next_observations: torch.Tensor,
    truncated: torch.Tensor,
    infos: Any,
) -> torch.Tensor:
    """Replace auto-reset observations with terminal ones for timeout values."""
    if not isinstance(infos, dict) or "final_observation" not in infos:
        return next_observations
    final_observation = infos["final_observation"]
    if not isinstance(final_observation, torch.Tensor):
        return next_observations
    mask = truncated.flatten().bool()
    available = infos.get("_final_observation")
    if isinstance(available, torch.Tensor):
        mask &= available.flatten().bool()
    if not mask.any():
        return next_observations
    if final_observation.shape != next_observations.shape:
        raise ValueError(
            "final_observation shape does not match next_observations: "
            f"{tuple(final_observation.shape)} != {tuple(next_observations.shape)}"
        )
    result = next_observations.clone()
    result[mask] = final_observation[mask]
    return result

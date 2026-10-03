"""Evaluate a visual ACT checkpoint with live environment rollouts.

This module is both a reusable Python API and a command-line program.  The
trainer calls :func:`evaluate_policy` periodically; saved checkpoints can be
evaluated independently with ``python -m policy_training.bc.eval_act``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .act_model import SimpleACTPolicy
from .resnet_mlp_model import ResNetMLPPolicy


EnvFactory = Callable[[str, int, int], Any]


def _as_scalar(value: Any) -> float:
    array = np.asarray(value)
    return float(array.reshape(-1)[0]) if array.size else 0.0


def observation_to_rgb(observation: Any, image_size: int = 224) -> torch.Tensor:
    """Convert a repository environment observation to CHW uint8 RGB."""
    frame = observation["rgb"] if isinstance(observation, dict) else observation
    frame = torch.as_tensor(frame)
    if frame.ndim != 3:
        raise ValueError(f"Expected a 3-D RGB observation, got {tuple(frame.shape)}")
    if frame.shape[0] != 3 and frame.shape[-1] == 3:
        frame = frame.permute(2, 0, 1)
    if frame.shape[0] != 3:
        raise ValueError(f"Could not identify RGB channels in {tuple(frame.shape)}")
    if frame.dtype != torch.uint8:
        frame = frame.float()
        if float(frame.max()) <= 1.0:
            frame = frame * 255.0
        frame = frame.clamp(0, 255).to(torch.uint8)
    if tuple(frame.shape[-2:]) != (image_size, image_size):
        resized = F.interpolate(
            frame[None].float() / 255.0,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        frame = (resized[0].clamp(0, 1) * 255).to(torch.uint8)
    return frame.contiguous()


def make_repository_env(task: str, seed: int, image_size: int):
    """Construct one task through MMBench2's normal environment interface."""
    repository_root = Path(__file__).resolve().parents[2]
    source_root = str(repository_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    # Lazy imports keep offline policy_training/tests independent of simulator imports.
    from env_wrapper import _EnvCfg
    from envs import make_env

    cfg = _EnvCfg(task, img_size=image_size, seed=seed)
    return make_env(cfg)


@torch.inference_mode()
def evaluate_policy(
    policy: torch.nn.Module,
    tasks: Sequence[str],
    *,
    device: torch.device | str,
    episodes: int = 10,
    seed: int = 0,
    image_size: int = 224,
    execution_horizon: int = 1,
    env_factory: EnvFactory = make_repository_env,
    task_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Roll out ``policy`` and return aggregate and per-task metrics.

    Success is counted if the environment reports ``info['success']`` at any
    point in the episode. A fresh action chunk is predicted after every
    ``execution_horizon`` actions; the default replans every step.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if execution_horizon < 1 or execution_horizon > policy.chunk_size:
        raise ValueError("execution_horizon must be in [1, policy.chunk_size]")
    if task_indices is None:
        task_indices = list(range(len(tasks)))
    if len(task_indices) != len(tasks):
        raise ValueError("task_indices must have one entry per task")
    if any(index < 0 or index >= policy.num_tasks for index in task_indices):
        raise ValueError("A task embedding index is outside the policy's task range")

    device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy.eval()
    per_task: dict[str, dict[str, Any]] = {}

    for evaluation_index, (task, task_index) in enumerate(zip(tasks, task_indices)):
        env = env_factory(task, seed + evaluation_index * 10_000, image_size)
        successes: list[float] = []
        returns: list[float] = []
        lengths: list[int] = []
        try:
            action_dim = int(env.action_space.shape[0])
            for _episode in range(episodes):
                observation, reset_info = env.reset()
                first_frame = observation_to_rgb(observation, image_size)
                context_length = int(getattr(policy, "context_length", 1))
                history = deque(
                    (first_frame.clone() for _ in range(context_length)),
                    maxlen=context_length,
                )
                success = _as_scalar(reset_info.get("success", 0.0))
                episode_return = 0.0
                length = 0
                done = False
                while not done:
                    frames = torch.stack(tuple(history))[None].to(
                        device, non_blocking=True
                    )
                    task_tensor = torch.tensor([task_index], device=device)
                    predicted = policy(frames, task_tensor)["actions"]
                    assert predicted is not None
                    action_chunk = predicted[0, :, :action_dim].float().cpu().numpy()
                    for action in action_chunk[:execution_horizon]:
                        observation, reward, terminated, truncated, info = env.step(action)
                        history.append(observation_to_rgb(observation, image_size))
                        episode_return += _as_scalar(reward)
                        length += 1
                        success = max(success, _as_scalar(info.get("success", 0.0)))
                        done = bool(_as_scalar(terminated) or _as_scalar(truncated))
                        if done:
                            break
                successes.append(float(success > 0.0))
                returns.append(episode_return)
                lengths.append(length)
        finally:
            env.close()

        per_task[task] = {
            "episodes": episodes,
            "success_rate": float(np.mean(successes)),
            "mean_return": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
        }

    return {
        "episodes_per_task": episodes,
        "mean_success_rate": float(
            np.mean([metrics["success_rate"] for metrics in per_task.values()])
        ),
        "mean_return": float(np.mean([metrics["mean_return"] for metrics in per_task.values()])),
        "tasks": per_task,
    }


def load_policy_checkpoint(path: str | Path, device: torch.device | str):
    """Reconstruct a policy without downloading backbone weights."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    policy_type = checkpoint.get("policy_type")
    policy_classes = {
        "simple_visual_act": SimpleACTPolicy,
        "resnet_mlp_visual_bc": ResNetMLPPolicy,
    }
    if policy_type not in policy_classes:
        raise ValueError(f"Unsupported policy type: {policy_type!r}")
    config = dict(checkpoint["model_config"])
    # The complete trained backbone is in the state dict. Avoid a redundant
    # ImageNet download while reconstructing it.
    config["pretrained_backbone"] = False
    policy = policy_classes[policy_type](**config)
    policy.load_state_dict(checkpoint["model"])
    policy.to(device)
    return policy, list(checkpoint["tasks"]), checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--tasks", nargs="+", help="Evaluate a checkpoint task subset")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="Optional JSON metrics path")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    policy, checkpoint_tasks, checkpoint = load_policy_checkpoint(args.checkpoint, device)
    tasks = checkpoint_tasks if args.tasks is None else args.tasks
    unknown = sorted(set(tasks) - set(checkpoint_tasks))
    if unknown:
        raise ValueError(f"Tasks are not represented in the checkpoint: {unknown}")
    task_indices = [checkpoint_tasks.index(task) for task in tasks]
    metrics = evaluate_policy(
        policy,
        tasks,
        device=device,
        episodes=args.episodes,
        seed=args.seed,
        image_size=args.image_size,
        execution_horizon=args.execution_horizon,
        task_indices=task_indices,
    )
    metrics.update({"checkpoint": str(Path(args.checkpoint)), "epoch": checkpoint.get("epoch")})
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    return metrics


if __name__ == "__main__":
    main()

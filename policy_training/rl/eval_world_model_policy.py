"""Run a portable visual PPO policy inside :class:`WorldModelEnv`.

The evaluator intentionally does not report a success rate: the dynamics model
has no success/termination head. It reports learned reward-head returns and
flow-instability diagnostics, and can save generated rollout videos.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from .env import make_rgb_env
from .ppo import load_portable_policy
from .world_model_env import make_world_model_env


def _rgb_hwc(observation: np.ndarray | torch.Tensor) -> np.ndarray:
    frame = torch.as_tensor(observation).detach().cpu().numpy()
    if frame.ndim != 3:
        raise ValueError(f"Expected a 3-D RGB frame, got {frame.shape}")
    if frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    return frame.astype(np.uint8, copy=False)


def _comparison_frame(
    simulator_observation: np.ndarray | torch.Tensor,
    world_model_render: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Make a labeled simulator-left/world-model-right video frame."""
    simulator = _rgb_hwc(simulator_observation)
    world_model = _rgb_hwc(world_model_render)
    if simulator.shape != world_model.shape:
        raise ValueError(
            f"Comparison frames differ: simulator={simulator.shape}, "
            f"world_model={world_model.shape}"
        )
    height, width, _ = simulator.shape
    # 224 + 32 = 256, avoiding implicit ffmpeg resizing at the default
    # 16-pixel macroblock boundary.
    header_height = 32
    canvas = Image.new("RGB", (2 * width, height + header_height), (18, 18, 18))
    canvas.paste(Image.fromarray(simulator), (0, header_height))
    canvas.paste(Image.fromarray(world_model), (width, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), "SIMULATOR POLICY", fill=(255, 255, 255))
    draw.text((width + 8, 10), "WORLD MODEL POLICY", fill=(255, 255, 255))
    draw.line((width, 0, width, height + header_height), fill=(255, 255, 255), width=1)
    return np.asarray(canvas)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("policy_checkpoint")
    p.add_argument("--world-model", default="src/checkpoints/combined_xl")
    p.add_argument("--task", help="Defaults to the task stored in the policy")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=30_000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--video-episodes", type=int, default=5)
    p.add_argument("--video-fps", type=int, default=10)
    return p


def evaluate_world_model_policy(
    policy,
    metadata: dict[str, Any],
    *,
    world_model: str | Path,
    task: str,
    episodes: int,
    steps: int,
    seed: int,
    device: str | torch.device,
    output_dir: str | Path,
    video_episodes: int = 5,
    video_fps: int = 10,
) -> dict[str, Any]:
    if episodes < 1 or steps < 1:
        raise ValueError("episodes and steps must be positive")
    if not 0 <= video_episodes <= episodes:
        raise ValueError("video_episodes must be between zero and episodes")

    output_dir = Path(output_dir)
    video_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    if video_episodes:
        video_dir.mkdir(parents=True, exist_ok=True)

    env = make_world_model_env(
        task=task,
        checkpoint_dir=world_model,
        observation_mode="rgb",
        reward_mode="predicted",
        max_episode_steps=steps,
        device=device,
        seed_from_env=False,
    )
    policy_image_size = int(metadata["image_size"])
    if env.observation_space.shape != (3, policy_image_size, policy_image_size):
        env.close()
        raise ValueError(
            f"Policy expects 3x{policy_image_size}x{policy_image_size}, but the "
            f"world model emits {env.observation_space.shape}"
        )
    if env.action_space.shape != (int(metadata["action_dim"]),):
        env.close()
        raise ValueError(
            f"Policy action dimension {metadata['action_dim']} does not match "
            f"task action dimension {env.action_space.shape[0]}"
        )
    simulator_env = make_rgb_env(task, seed=seed, image_size=policy_image_size)

    returns: list[float] = []
    mean_instabilities: list[float] = []
    max_instabilities: list[float] = []
    mean_action_norms: list[float] = []
    mean_simulator_action_norms: list[float] = []
    saturated_action_fractions: list[float] = []
    episode_records: list[dict[str, Any]] = []
    simulator_returns: list[float] = []
    simulator_successes: list[float] = []
    video_paths: list[str] = []
    policy.eval()
    context_length = int(metadata.get("context_length", 1))
    try:
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            simulator_observation, simulator_reset_info = simulator_env.reset(
                seed=episode_seed
            )
            observation, reset_info = env.reset(
                seed=episode_seed,
                options={"initial_observation": simulator_observation},
            )
            world_history = deque(
                (np.asarray(observation).copy() for _ in range(context_length)),
                maxlen=context_length,
            )
            simulator_history = deque(
                (np.asarray(simulator_observation).copy() for _ in range(context_length)),
                maxlen=context_length,
            )
            writer = None
            if episode_index < video_episodes:
                video_path = video_dir / f"episode_{episode_index:03d}.mp4"
                writer = imageio.get_writer(video_path, fps=video_fps, codec="libx264")
                writer.append_data(
                    _comparison_frame(simulator_observation, env.render())
                )
                video_paths.append(str(video_path))

            episode_return = 0.0
            simulator_return = 0.0
            simulator_success = float(
                np.asarray(
                    simulator_reset_info.get(
                        "is_success", simulator_reset_info.get("success", False)
                    )
                ).reshape(-1)[0]
            )
            simulator_done = False
            instabilities: list[float] = []
            action_norms: list[float] = []
            simulator_action_norms: list[float] = []
            saturated = 0
            action_values = 0
            length = 0
            done = False
            try:
                while not done:
                    world_model_action = np.asarray(
                        policy.predict(
                            world_history[-1]
                            if context_length == 1
                            else np.stack(tuple(world_history))
                        ),
                        dtype=np.float32,
                    )
                    if not simulator_done:
                        simulator_action = np.asarray(
                            policy.predict(
                                simulator_history[-1]
                                if context_length == 1
                                else np.stack(tuple(simulator_history))
                            ),
                            dtype=np.float32,
                        )
                        (
                            simulator_observation,
                            simulator_reward,
                            simulator_terminated,
                            simulator_truncated,
                            simulator_info,
                        ) = simulator_env.step(simulator_action)
                        simulator_history.append(np.asarray(simulator_observation).copy())
                        simulator_return += float(simulator_reward)
                        simulator_action_norms.append(
                            float(np.linalg.norm(simulator_action))
                        )
                        simulator_success = max(
                            simulator_success,
                            float(bool(simulator_info.get("is_success", False))),
                        )
                        simulator_done = bool(
                            simulator_terminated or simulator_truncated
                        )
                    observation, reward, terminated, truncated, info = env.step(
                        world_model_action
                    )
                    world_history.append(np.asarray(observation).copy())
                    if writer is not None:
                        writer.append_data(
                            _comparison_frame(simulator_observation, env.render())
                        )
                    episode_return += float(reward)
                    instabilities.append(float(info["flow_instability"]))
                    action_norms.append(float(np.linalg.norm(world_model_action)))
                    saturated += int(
                        np.count_nonzero(np.abs(world_model_action) >= 0.95)
                    )
                    action_values += int(world_model_action.size)
                    length += 1
                    done = bool(terminated or truncated)
            finally:
                if writer is not None:
                    writer.close()

            record = {
                "episode": episode_index,
                "seed": episode_seed,
                "seed_source": reset_info["seed_source"],
                "length": length,
                "predicted_return": episode_return,
                "simulator_return": simulator_return,
                "simulator_success": simulator_success,
                "mean_flow_instability": float(np.mean(instabilities)),
                "max_flow_instability": float(np.max(instabilities)),
                "mean_action_l2": float(np.mean(action_norms)),
                "mean_simulator_action_l2": float(
                    np.mean(simulator_action_norms)
                ),
                "saturated_action_fraction": saturated / max(action_values, 1),
            }
            episode_records.append(record)
            returns.append(record["predicted_return"])
            simulator_returns.append(record["simulator_return"])
            simulator_successes.append(record["simulator_success"])
            mean_instabilities.append(record["mean_flow_instability"])
            max_instabilities.append(record["max_flow_instability"])
            mean_action_norms.append(record["mean_action_l2"])
            mean_simulator_action_norms.append(record["mean_simulator_action_l2"])
            saturated_action_fractions.append(record["saturated_action_fraction"])
    finally:
        env.close()
        simulator_env.close()

    metrics = {
        "task": task,
        "episodes": episodes,
        "steps_per_episode": steps,
        "predicted_return_mean": float(np.mean(returns)),
        "predicted_return_std": float(np.std(returns)),
        "simulator_return_mean": float(np.mean(simulator_returns)),
        "simulator_return_std": float(np.std(simulator_returns)),
        "simulator_success_rate": float(np.mean(simulator_successes)),
        "mean_flow_instability": float(np.mean(mean_instabilities)),
        "max_flow_instability": float(np.max(max_instabilities)),
        "mean_action_l2": float(np.mean(mean_action_norms)),
        "mean_simulator_action_l2": float(np.mean(mean_simulator_action_norms)),
        "saturated_action_fraction": float(np.mean(saturated_action_fractions)),
        "success_rate": None,
        "success_rate_note": "Unavailable: the world model has no success/termination head.",
        "comparison_action_source": "independent_closed_loop",
        "video_layout": "simulator_left_world_model_right",
        "videos": video_paths,
        "episodes_detail": episode_records,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parser().parse_args(argv)
    policy, metadata = load_portable_policy(
        args.policy_checkpoint, device=args.device
    )
    task = args.task or metadata["task"]
    metrics = evaluate_world_model_policy(
        policy,
        metadata,
        world_model=args.world_model,
        task=task,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        video_episodes=args.video_episodes,
        video_fps=args.video_fps,
    )
    metrics.update(
        {
            "policy_checkpoint": str(Path(args.policy_checkpoint)),
            "world_model": str(Path(args.world_model)),
        }
    )
    (Path(args.output_dir) / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()

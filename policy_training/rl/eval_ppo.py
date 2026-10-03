"""Evaluate and record a portable MMBench skrl PPO policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .ppo import evaluate_ppo, load_portable_policy


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--task", help="Defaults to the task stored in the checkpoint")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--image-size", type=int, help="Must match the checkpoint when provided")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output-dir",
        required=True,
        help="Folder for metrics.json and the videos/ subdirectory",
    )
    p.add_argument("--video-episodes", type=int, default=5)
    p.add_argument("--video-fps", type=int, default=20)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = parser().parse_args(argv)
    policy, metadata = load_portable_policy(args.checkpoint, device=args.device)
    task = args.task or metadata["task"]
    image_size = int(metadata["image_size"])
    if args.image_size is not None and args.image_size != image_size:
        raise ValueError(
            f"Checkpoint uses image-size {image_size}, but {args.image_size} was requested"
        )
    metrics = evaluate_ppo(
        policy,
        task,
        episodes=args.episodes,
        seed=args.seed,
        image_size=image_size,
        output_dir=args.output_dir,
        video_episodes=args.video_episodes,
        video_fps=args.video_fps,
    )
    metrics.update({"checkpoint": str(Path(args.checkpoint)), "task": task})
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    (Path(args.output_dir) / "metrics.json").write_text(rendered + "\n")
    return metrics


if __name__ == "__main__":
    main()

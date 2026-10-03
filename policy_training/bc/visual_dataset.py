"""RGB-strip dataset for visual behavior cloning on MMBench2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


@dataclass
class _VisualTrajectory:
    task: str
    task_index: int
    action_dim: int
    frames: torch.Tensor
    actions: torch.Tensor
    episodes: torch.Tensor


def read_frame_strips(directory: Path, task: str) -> torch.Tensor:
    """Read the repository's horizontal PNG strips into (N, 3, 224, 224)."""
    chunks: list[torch.Tensor] = []
    strip_index = 0
    while True:
        path = directory / f"{task}-{strip_index}.png"
        if not path.exists():
            break
        strip = read_image(str(path))
        channels, height, total_width = strip.shape
        if channels != 3 or height != 224 or total_width % 224:
            raise ValueError(f"Unexpected RGB strip shape {tuple(strip.shape)} in {path}")
        frame_count = total_width // 224
        frames = strip.view(3, 224, frame_count, 224).permute(2, 0, 1, 3)
        chunks.append(frames.contiguous())
        strip_index += 1
    if not chunks:
        raise FileNotFoundError(f"No RGB strips found for {task} in {directory}")
    return torch.cat(chunks, dim=0)


class MMBenchVisualBCDataset(Dataset):
    """Aligned RGB histories and future action chunks.

    All selected strips are decoded once at construction. This is appropriate
    for the small expert robotics subsets and avoids repeatedly decoding very
    wide PNGs. Larger corpora should first use ``preprocess_dataset.py`` and a
    future sharded version of this adapter.
    """

    def __init__(
        self,
        root: str | Path,
        splits: Sequence[str],
        tasks: Sequence[str],
        *,
        tasks_json: str | Path,
        context_length: int = 2,
        chunk_size: int = 8,
        max_action_dim: int = 16,
    ) -> None:
        if context_length < 1 or chunk_size < 1:
            raise ValueError("context_length and chunk_size must be positive")
        self.root = Path(root).expanduser()
        self.tasks = list(tasks)
        self.task_to_index = {task: index for index, task in enumerate(self.tasks)}
        self.context_length = int(context_length)
        self.chunk_size = int(chunk_size)
        self.max_action_dim = int(max_action_dim)
        self.image_shape = (3, 224, 224)

        metadata = json.loads(Path(tasks_json).read_text())
        self.trajectories: list[_VisualTrajectory] = []
        self.samples: list[tuple[int, int]] = []

        for split in splits:
            split_dir = self.root / split
            for task in self.tasks:
                demo_path = split_dir / f"{task}.pt"
                if not demo_path.exists():
                    continue
                if task not in metadata:
                    raise KeyError(f"{task!r} is missing from {tasks_json}")
                action_dim = int(metadata[task]["action_dim"])
                if not 1 <= action_dim <= self.max_action_dim:
                    raise ValueError(f"Invalid action dimension for {task}: {action_dim}")

                data = torch.load(demo_path, map_location="cpu", weights_only=False)
                actions = data["action"].float().contiguous()
                episodes = data["episode"].long().contiguous()
                frames = read_frame_strips(split_dir, task)
                if not (len(frames) == len(actions) == len(episodes)):
                    raise ValueError(
                        f"Frame/metadata length mismatch for {task} in {split}: "
                        f"{len(frames)}, {len(actions)}, {len(episodes)}"
                    )

                trajectory_index = len(self.trajectories)
                self.trajectories.append(
                    _VisualTrajectory(
                        task=task,
                        task_index=self.task_to_index[task],
                        action_dim=action_dim,
                        frames=frames,
                        actions=actions,
                        episodes=episodes,
                    )
                )
                last_start = len(frames) - self.chunk_size - 1
                for start in range(max(0, last_start + 1)):
                    end = start + self.chunk_size
                    if episodes[start] != episodes[end]:
                        continue
                    target = actions[start + 1 : end + 1, :action_dim]
                    if torch.isfinite(target).all():
                        self.samples.append((trajectory_index, start))

        if not self.trajectories:
            raise FileNotFoundError("No visual trajectories found for the requested selection")
        if not self.samples:
            raise ValueError("No valid visual action chunks found")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        trajectory_index, start = self.samples[index]
        trajectory = self.trajectories[trajectory_index]
        episode = trajectory.episodes[start]

        context_start = start
        while (
            context_start > 0
            and start - context_start + 1 < self.context_length
            and trajectory.episodes[context_start - 1] == episode
        ):
            context_start -= 1
        frames = trajectory.frames[context_start : start + 1]
        if len(frames) < self.context_length:
            padding = frames[:1].expand(self.context_length - len(frames), -1, -1, -1)
            frames = torch.cat((padding, frames), dim=0)

        target = torch.zeros(self.chunk_size, self.max_action_dim, dtype=torch.float32)
        mask = torch.zeros_like(target, dtype=torch.bool)
        target[:, : trajectory.action_dim] = trajectory.actions[
            start + 1 : start + 1 + self.chunk_size, : trajectory.action_dim
        ].clamp(-1.0, 1.0)
        mask[:, : trajectory.action_dim] = True
        return {
            "frames": frames,
            "action": target,
            "action_mask": mask,
            "task_index": torch.tensor(trajectory.task_index, dtype=torch.long),
        }

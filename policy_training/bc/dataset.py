"""Offline state/action dataset for standalone behavior cloning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset


DOMAIN_PREFIXES = {"maniskill": "ms-", "metaworld": "mw-"}


@dataclass(frozen=True)
class ObservationStats:
    mean: torch.Tensor
    std: torch.Tensor

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "ObservationStats":
        return cls(mean=state["mean"].float(), std=state["std"].float())


@dataclass
class _Trajectory:
    task: str
    task_index: int
    action_dim: int
    observations: torch.Tensor
    actions: torch.Tensor
    episodes: torch.Tensor


def discover_tasks(
    root: str | Path,
    splits: Sequence[str],
    *,
    domain: str | None = None,
    tasks: Sequence[str] | None = None,
) -> list[str]:
    """Return tasks present in at least one requested split."""
    root = Path(root).expanduser()
    found = {
        path.stem
        for split in splits
        for path in (root / split).glob("*.pt")
    }
    if domain is not None:
        if domain not in DOMAIN_PREFIXES:
            raise ValueError(f"Unknown domain {domain!r}; choose from {sorted(DOMAIN_PREFIXES)}")
        prefix = DOMAIN_PREFIXES[domain]
        found = {task for task in found if task.startswith(prefix)}
    if tasks:
        requested = set(tasks)
        missing = requested - found
        if missing:
            raise FileNotFoundError(
                f"Tasks not found in splits {list(splits)}: {sorted(missing)}"
            )
        found &= requested
    if not found:
        raise FileNotFoundError(
            f"No matching .pt trajectories under {root} for splits {list(splits)}"
        )
    return sorted(found)


class MMBenchStateBCDataset(Dataset):
    """Aligned state histories and future action chunks.

    Raw MMBench2 files store the action causing ``obs[t-1] -> obs[t]`` at
    position ``t``. A sample anchored at observation ``t`` therefore targets
    raw actions ``t+1:t+1+chunk_size``.
    """

    def __init__(
        self,
        root: str | Path,
        splits: Sequence[str],
        tasks: Sequence[str],
        *,
        tasks_json: str | Path,
        context_length: int = 1,
        chunk_size: int = 8,
        max_action_dim: int = 16,
        observation_stats: ObservationStats | None = None,
    ) -> None:
        if context_length < 1 or chunk_size < 1:
            raise ValueError("context_length and chunk_size must be positive")
        self.root = Path(root).expanduser()
        self.tasks = list(tasks)
        self.task_to_index = {task: index for index, task in enumerate(self.tasks)}
        self.context_length = int(context_length)
        self.chunk_size = int(chunk_size)
        self.max_action_dim = int(max_action_dim)

        metadata = json.loads(Path(tasks_json).read_text())
        self.trajectories: list[_Trajectory] = []
        self.samples: list[tuple[int, int]] = []
        observations_for_stats: list[torch.Tensor] = []

        for split in splits:
            for task in self.tasks:
                path = self.root / split / f"{task}.pt"
                if not path.exists():
                    continue
                if task not in metadata:
                    raise KeyError(f"{task!r} is missing from {tasks_json}")
                action_dim = int(metadata[task]["action_dim"])
                if not 1 <= action_dim <= self.max_action_dim:
                    raise ValueError(f"Invalid action dimension for {task}: {action_dim}")

                # These repository datasets are trusted TensorDict files and
                # require regular torch.load rather than weights_only loading.
                data = torch.load(path, map_location="cpu", weights_only=False)
                observations = data["obs"].float().contiguous()
                actions = data["action"].float().contiguous()
                episodes = data["episode"].long().contiguous()
                if observations.ndim != 2 or actions.ndim != 2 or episodes.ndim != 1:
                    raise ValueError(f"Unexpected tensor ranks in {path}")
                if not (len(observations) == len(actions) == len(episodes)):
                    raise ValueError(f"Mismatched trajectory lengths in {path}")
                if actions.shape[1] < action_dim:
                    raise ValueError(f"{path} has only {actions.shape[1]} action columns")

                trajectory_index = len(self.trajectories)
                self.trajectories.append(
                    _Trajectory(
                        task=task,
                        task_index=self.task_to_index[task],
                        action_dim=action_dim,
                        observations=observations,
                        actions=actions,
                        episodes=episodes,
                    )
                )
                observations_for_stats.append(observations)

                last_start = len(observations) - self.chunk_size - 1
                for start in range(max(0, last_start + 1)):
                    end = start + self.chunk_size
                    if episodes[start] != episodes[end]:
                        continue
                    target = actions[start + 1 : end + 1, :action_dim]
                    if torch.isfinite(target).all():
                        self.samples.append((trajectory_index, start))

        if not self.trajectories:
            raise FileNotFoundError("No trajectories found for the requested tasks and splits")
        if not self.samples:
            raise ValueError("No valid action chunks found; reduce chunk_size or inspect the data")

        obs_dims = {trajectory.observations.shape[1] for trajectory in self.trajectories}
        if len(obs_dims) != 1:
            raise ValueError(f"All selected tasks must share an observation size, got {obs_dims}")
        self.observation_dim = obs_dims.pop()

        if observation_stats is None:
            all_observations = torch.cat(observations_for_stats, dim=0)
            if not torch.isfinite(all_observations).all():
                raise ValueError("Non-finite observations found")
            mean = all_observations.mean(dim=0)
            std = all_observations.std(dim=0).clamp_min(1e-6)
            observation_stats = ObservationStats(mean=mean, std=std)
        if observation_stats.mean.numel() != self.observation_dim:
            raise ValueError("Observation statistics do not match observation dimension")
        self.observation_stats = observation_stats

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
        history = trajectory.observations[context_start : start + 1]
        if len(history) < self.context_length:
            padding = history[:1].expand(self.context_length - len(history), -1)
            history = torch.cat((padding, history), dim=0)
        history = (history - self.observation_stats.mean) / self.observation_stats.std

        target = torch.zeros(self.chunk_size, self.max_action_dim, dtype=torch.float32)
        mask = torch.zeros_like(target, dtype=torch.bool)
        action_slice = trajectory.actions[
            start + 1 : start + 1 + self.chunk_size, : trajectory.action_dim
        ]
        target[:, : trajectory.action_dim] = action_slice.clamp(-1.0, 1.0)
        mask[:, : trajectory.action_dim] = True
        return {
            "observation": history,
            "action": target,
            "action_mask": mask,
            "task_index": torch.tensor(trajectory.task_index, dtype=torch.long),
        }


def format_task_summary(dataset: MMBenchStateBCDataset) -> str:
    counts = {task: 0 for task in dataset.tasks}
    for trajectory_index, _ in dataset.samples:
        counts[dataset.trajectories[trajectory_index].task] += 1
    return ", ".join(f"{task}={counts[task]}" for task in dataset.tasks)

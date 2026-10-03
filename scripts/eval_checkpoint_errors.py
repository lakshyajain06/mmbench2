#!/usr/bin/env python3
"""Per-task rollout-error evaluation using the repository's dynamics eval path.

The implementation deliberately reuses the checkpoint loaders and autoregressive
rollout helpers used by ``src/train_dynamics.py``.  It reads the released raw
MMBench2 PNG strips directly so a second, very large sharded copy is unnecessary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from interactive import load_dynamics_from_ckpt, load_tokenizer_from_ckpt  # noqa: E402
from model import pack_bottleneck_to_spatial, temporal_patchify  # noqa: E402
from train_dynamics import (  # noqa: E402
    decode_packed_to_frames,
    make_tau_schedule,
    sample_autoregressive_packed_sequence,
)


Image.MAX_IMAGE_PIXELS = None


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "little") + int(seed)) % (2**31)


def read_task_frames(data_dir: Path, task: str) -> torch.Tensor:
    """Load all horizontal PNG strips for one task as NCHW uint8."""
    chunks: list[torch.Tensor] = []
    index = 0
    while True:
        path = data_dir / f"{task}-{index}.png"
        if not path.exists():
            break
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        height, width, channels = rgb.shape
        if height != 224 or width % 224 or channels != 3:
            raise ValueError(f"unexpected image strip shape for {path}: {rgb.shape}")
        frames = torch.from_numpy(rgb).view(224, width // 224, 224, 3)
        chunks.append(frames.permute(1, 3, 0, 2).contiguous())
        index += 1
    if not chunks:
        raise FileNotFoundError(f"no PNG strips found for {task} in {data_dir}")
    return torch.cat(chunks, dim=0)


def choose_windows(
    episode_ids: torch.Tensor,
    *,
    task: str,
    sequence_length: int,
    max_episodes: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Choose one reproducible valid window per episode."""
    rng = np.random.default_rng(stable_seed(task, seed))
    windows: list[tuple[int, int]] = []
    for episode_id in torch.unique(episode_ids, sorted=True).tolist():
        indices = torch.nonzero(episode_ids == episode_id, as_tuple=False).flatten()
        if len(indices) < sequence_length:
            continue
        first = int(indices[0])
        last_start = int(indices[-1]) - sequence_length + 1
        start = int(rng.integers(first, last_start + 1))
        windows.append((int(episode_id), start))
    if max_episodes > 0 and len(windows) > max_episodes:
        picks = np.sort(rng.choice(len(windows), size=max_episodes, replace=False))
        windows = [windows[int(i)] for i in picks]
    return windows


def load_task_batch_data(
    data_dir: Path,
    task: str,
    *,
    context: int,
    horizon: int,
    max_episodes: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    payload = torch.load(data_dir / f"{task}.pt", map_location="cpu", weights_only=False)
    episode_ids = payload["episode"].to(torch.int64).cpu()
    raw_actions = payload["action"].to(torch.float32).cpu()
    frames = read_task_frames(data_dir, task)
    usable = min(len(episode_ids), len(raw_actions), len(frames))
    episode_ids, raw_actions, frames = episode_ids[:usable], raw_actions[:usable], frames[:usable]

    length = context + horizon
    windows = choose_windows(
        episode_ids,
        task=task,
        sequence_length=length,
        max_episodes=max_episodes,
        seed=seed,
    )
    if not windows:
        raise RuntimeError(f"{task} has no episodes with at least {length} frames")

    frame_windows = []
    action_windows = []
    selected_episodes = []
    for episode_id, start in windows:
        frame_windows.append(frames[start : start + length])
        # Match train_dynamics.parse_batch: action[k] produced observation[k],
        # hence the first context frame gets a zero action.
        action = torch.zeros((length, 16), dtype=torch.float32)
        action[1:] = raw_actions[start + 1 : start + length]
        action_windows.append(action)
        selected_episodes.append(episode_id)
    return torch.stack(frame_windows), torch.stack(action_windows), selected_episodes


def task_metadata(tasks_json: Path, task: str) -> tuple[torch.Tensor, torch.Tensor]:
    metadata = json.loads(tasks_json.read_text())
    task_info = metadata[task]
    action_dim = max(0, min(16, int(task_info.get("action_dim", 16))))
    action_mask = torch.zeros(16, dtype=torch.float32)
    action_mask[:action_dim] = 1.0
    language = torch.tensor(task_info["text_embedding"], dtype=torch.float32)
    return action_mask, language


@torch.inference_mode()
def evaluate_task(
    *,
    task: str,
    checkpoint_label: str,
    data_dir: Path,
    tasks_json: Path,
    tokenizer,
    dynamics,
    tokenizer_info: dict,
    dynamics_info: dict,
    packing_factor: int,
    context: int,
    horizon: int,
    batch_size: int,
    max_episodes: int,
    seed: int,
    tau_ctx: float,
    schedule: dict,
    device: torch.device,
) -> dict:
    frames_u8, actions, episode_ids = load_task_batch_data(
        data_dir,
        task,
        context=context,
        horizon=horizon,
        max_episodes=max_episodes,
        seed=seed,
    )
    action_mask_1d, language = task_metadata(tasks_json, task)

    sample_mse: list[float] = []
    sample_floor_mse: list[float] = []
    per_t_sse = torch.zeros(horizon, dtype=torch.float64)
    per_t_count = torch.zeros(horizon, dtype=torch.float64)
    total_sse = 0.0
    floor_sse = 0.0
    total_count = 0

    height = int(tokenizer_info["H"])
    width = int(tokenizer_info["W"])
    channels = int(tokenizer_info["C"])
    patch = int(tokenizer_info["patch"])
    n_spatial = int(dynamics_info["n_spatial"])

    started = time.time()
    for batch_start in range(0, len(frames_u8), batch_size):
        batch_end = min(batch_start + batch_size, len(frames_u8))
        batch_frames = frames_u8[batch_start:batch_end].to(device).float().div_(255.0)
        batch_actions = actions[batch_start:batch_end].to(device)
        batch_mask = action_mask_1d.to(device).view(1, 1, -1).expand(
            len(batch_frames), context + horizon, -1
        )
        batch_actions = batch_actions.clamp(-1, 1) * batch_mask
        batch_language = language.to(device).view(1, -1).expand(len(batch_frames), -1)

        patches = temporal_patchify(batch_frames, patch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            encoded, _ = tokenizer.encoder(patches)
        packed = pack_bottleneck_to_spatial(
            encoded,
            n_spatial=n_spatial,
            k=packing_factor,
        )

        # Reset per batch so both checkpoints see identical diffusion/context noise.
        torch.manual_seed(stable_seed(f"{task}:{batch_start}", seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(stable_seed(f"{task}:{batch_start}", seed))
        predicted_packed = sample_autoregressive_packed_sequence(
            dynamics,
            z_gt_packed=packed,
            ctx_length=context,
            horizon=horizon,
            k_max=int(dynamics_info["k_max"]),
            sched=schedule,
            actions=batch_actions,
            act_mask=batch_mask,
            tau_ctx=tau_ctx,
            lang_emb=batch_language,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predicted_frames = decode_packed_to_frames(
                tokenizer.decoder,
                z_packed=predicted_packed,
                H=height,
                W=width,
                C=channels,
                patch=patch,
                packing_factor=packing_factor,
            )

        truth = batch_frames[:, context : context + horizon].float()
        prediction = predicted_frames[:, context : context + horizon].float()
        floor = batch_frames[:, context - 1 : context].expand(-1, horizon, -1, -1, -1)
        squared = (prediction - truth).square()
        floor_squared = (floor - truth).square()

        sample_mse.extend(squared.mean(dim=(1, 2, 3, 4)).cpu().tolist())
        sample_floor_mse.extend(floor_squared.mean(dim=(1, 2, 3, 4)).cpu().tolist())
        total_sse += float(squared.sum().item())
        floor_sse += float(floor_squared.sum().item())
        total_count += squared.numel()
        per_t_sse += squared.sum(dim=(0, 2, 3, 4)).double().cpu()
        per_t_count += torch.full((horizon,), squared.shape[0] * channels * height * width, dtype=torch.float64)

        del batch_frames, patches, encoded, packed, predicted_packed, predicted_frames

    mse = total_sse / total_count
    floor_mse = floor_sse / total_count
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    floor_psnr = 10.0 * math.log10(1.0 / max(floor_mse, 1e-12))
    sample_array = np.asarray(sample_mse, dtype=np.float64)
    floor_array = np.asarray(sample_floor_mse, dtype=np.float64)
    domain = "maniskill" if task.startswith("ms-") else "metaworld"
    return {
        "checkpoint": checkpoint_label,
        "task": task,
        "domain": domain,
        "episodes": len(sample_mse),
        "episode_ids": episode_ids,
        "context": context,
        "horizon": horizon,
        "mse": mse,
        "mse_episode_std": float(sample_array.std(ddof=1)) if len(sample_array) > 1 else 0.0,
        "mse_episode_sem": float(sample_array.std(ddof=1) / math.sqrt(len(sample_array))) if len(sample_array) > 1 else 0.0,
        "floor_mse": floor_mse,
        "floor_mse_episode_std": float(floor_array.std(ddof=1)) if len(floor_array) > 1 else 0.0,
        "mse_ratio_pred_over_floor": mse / max(floor_mse, 1e-12),
        "psnr": psnr,
        "floor_psnr": floor_psnr,
        "psnr_gain_over_floor_db": psnr - floor_psnr,
        "mse_per_timestep": (per_t_sse / per_t_count.clamp_min(1)).tolist(),
        "seconds": time.time() - started,
    }


def parse_tasks(args: argparse.Namespace) -> list[str]:
    if args.tasks_file:
        tasks = [line.strip() for line in Path(args.tasks_file).read_text().splitlines() if line.strip()]
    elif args.tasks:
        tasks = [part.strip() for part in args.tasks.split(",") if part.strip()]
    else:
        tasks = sorted(path.stem for path in args.data_dir.glob("*.pt") if path.stem.startswith(("ms-", "mw-")))
    return tasks[args.shard_index :: args.num_shards]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tasks-json", type=Path, default=ROOT / "tasks.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default=None, help="Comma-separated task names")
    parser.add_argument("--tasks-file", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--packing-factor", type=int, default=2)
    parser.add_argument("--eval-d", type=float, default=0.25)
    parser.add_argument("--tau-ctx", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must be in [0, num-shards)")
    tasks = parse_tasks(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this evaluation")
    device = torch.device("cuda:0")
    tokenizer, tokenizer_info = load_tokenizer_from_ckpt(
        str(args.checkpoint_dir / "tokenizer.pt"), device
    )
    dynamics, _reward_head, _policy_head, dynamics_info = load_dynamics_from_ckpt(
        str(args.checkpoint_dir / "dynamics.pt"),
        device=device,
        d_bottleneck=int(tokenizer_info["d_bottleneck"]),
        n_latents=int(tokenizer_info["n_latents"]),
        packing_factor=args.packing_factor,
    )
    schedule = make_tau_schedule(
        k_max=int(dynamics_info["k_max"]), schedule="shortcut", d=args.eval_d
    )

    completed = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                completed.add(json.loads(line)["task"])

    for index, task in enumerate(tasks, 1):
        if task in completed:
            print(f"[{index}/{len(tasks)}] {task}: already complete", flush=True)
            continue
        print(f"[{index}/{len(tasks)}] {args.checkpoint_label} {task}", flush=True)
        result = evaluate_task(
            task=task,
            checkpoint_label=args.checkpoint_label,
            data_dir=args.data_dir,
            tasks_json=args.tasks_json,
            tokenizer=tokenizer,
            dynamics=dynamics,
            tokenizer_info=tokenizer_info,
            dynamics_info=dynamics_info,
            packing_factor=args.packing_factor,
            context=args.context,
            horizon=args.horizon,
            batch_size=args.batch_size,
            max_episodes=args.max_episodes,
            seed=args.seed,
            tau_ctx=args.tau_ctx,
            schedule=schedule,
            device=device,
        )
        with args.output.open("a") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        print(
            f"  mse={result['mse']:.6f} ratio={result['mse_ratio_pred_over_floor']:.3f} "
            f"psnr={result['psnr']:.2f}dB seconds={result['seconds']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

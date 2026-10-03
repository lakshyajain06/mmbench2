"""Train visual BC using the frozen MMBench2 tokenizer and dense tokens."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import discover_tasks
from .model import masked_action_mse
from .visual_dataset import MMBenchVisualBCDataset
from .visual_model import DenseVisualChunkPolicy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_encoder(checkpoint: str, device: torch.device):
    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from train_dynamics import load_frozen_tokenizer_from_pt_ckpt

    encoder, decoder, info = load_frozen_tokenizer_from_pt_ckpt(checkpoint, device=device)
    del decoder
    return encoder, info


@torch.no_grad()
def encode_frames(encoder, frames: torch.Tensor, patch: int, device, amp: bool) -> torch.Tensor:
    from model import temporal_patchify

    frames = frames.to(device, non_blocking=True).float().div_(255.0)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        patches = temporal_patchify(frames, patch)
        tokens, _ = encoder(patches)
    return tokens


@torch.no_grad()
def evaluate(policy, encoder, loader, patch, device, amp: bool) -> float:
    policy.eval()
    total = 0.0
    batches = 0
    for batch in loader:
        tokens = encode_frames(encoder, batch["frames"], patch, device, amp)
        task = batch["task_index"].to(device, non_blocking=True)
        target = batch["action"].to(device, non_blocking=True)
        mask = batch["action_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            loss = masked_action_mse(policy(tokens, task), target, mask)
        total += float(loss)
        batches += 1
    if not batches:
        raise ValueError("Validation loader produced no batches")
    return total / batches


def save_checkpoint(path, policy, optimizer, epoch, step, args, tasks, encoder_info) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "policy_type": "dense_visual_chunk_bc",
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "tasks": tasks,
            "policy_config": {
                key: getattr(policy, key)
                for key in (
                    "visual_dim", "tokens_per_frame", "num_tasks", "context_length",
                    "chunk_size", "action_dim"
                )
            },
            "encoder": {
                "type": "mmbench2_tokenizer",
                "checkpoint": str(Path(args.tokenizer_checkpoint).resolve()),
                "patch": int(encoder_info["patch"]),
                "n_latents": int(encoder_info["n_latents"]),
                "d_bottleneck": int(encoder_info["d_bottleneck"]),
            },
            "train_args": vars(args),
        },
        path,
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="~/datasets/mmbench2_robotics")
    p.add_argument("--tasks-json", default="tasks.json")
    p.add_argument("--tokenizer-checkpoint", default="src/checkpoints/combined/tokenizer.pt")
    p.add_argument("--train-splits", nargs="+", default=["expert"])
    p.add_argument("--val-splits", nargs="+", default=["val"])
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--domain", choices=["maniskill", "metaworld"])
    selection.add_argument("--tasks", nargs="+")
    p.add_argument("--context-length", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="policy_training/runs/visual_bc")
    p.add_argument("--save-every", type=int, default=10)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    root = Path(args.data_root).expanduser()
    tasks = discover_tasks(root, args.train_splits, domain=args.domain, tasks=args.tasks)
    print(f"loading RGB strips for {tasks}")
    train_data = MMBenchVisualBCDataset(
        root, args.train_splits, tasks, tasks_json=args.tasks_json,
        context_length=args.context_length, chunk_size=args.chunk_size,
    )
    val_data = MMBenchVisualBCDataset(
        root, args.val_splits, tasks, tasks_json=args.tasks_json,
        context_length=args.context_length, chunk_size=args.chunk_size,
    )
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)

    print(f"loading frozen tokenizer from {args.tokenizer_checkpoint}")
    encoder, encoder_info = load_encoder(args.tokenizer_checkpoint, device)
    policy = DenseVisualChunkPolicy(
        visual_dim=int(encoder_info["d_bottleneck"]),
        tokens_per_frame=int(encoder_info["n_latents"]),
        num_tasks=len(tasks),
        context_length=args.context_length,
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp = device.type == "cuda"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({**vars(args), "tasks": tasks}, indent=2) + "\n")
    print(f"train samples: {len(train_data)}, val samples: {len(val_data)}")
    print(f"trainable parameters: {sum(p.numel() for p in policy.parameters()):,}")

    best = float("inf")
    step = 0
    patch = int(encoder_info["patch"])
    for epoch in range(1, args.epochs + 1):
        policy.train()
        total = 0.0
        batches = 0
        for batch in train_loader:
            tokens = encode_frames(encoder, batch["frames"], patch, device, amp)
            task = batch["task_index"].to(device, non_blocking=True)
            target = batch["action"].to(device, non_blocking=True)
            mask = batch["action_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss = masked_action_mse(policy(tokens, task), target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
            step += 1
        val = evaluate(policy, encoder, val_loader, patch, device, amp)
        print(f"epoch={epoch:04d} train_mse={total/max(batches,1):.6f} val_mse={val:.6f}")
        save_checkpoint(output / "latest.pt", policy, optimizer, epoch, step, args, tasks, encoder_info)
        if val < best:
            best = val
            save_checkpoint(output / "best.pt", policy, optimizer, epoch, step, args, tasks, encoder_info)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output / f"epoch_{epoch:04d}.pt", policy, optimizer, epoch, step,
                args, tasks, encoder_info,
            )


if __name__ == "__main__":
    main()

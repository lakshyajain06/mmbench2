"""Train a standalone action-chunking behavior-cloning policy."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import MMBenchStateBCDataset, discover_tasks, format_task_summary
from .model import ChunkedBCPolicy, masked_action_mse


@dataclass(frozen=True)
class ModelConfig:
    observation_dim: int
    num_tasks: int
    context_length: int
    chunk_size: int
    action_dim: int
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, amp: bool) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        observation = batch["observation"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        mask = batch["action_mask"].to(device, non_blocking=True)
        task_index = batch["task_index"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            prediction = model(observation, task_index)
            loss = masked_action_mse(prediction, action, mask)
        total_loss += float(loss)
        total_batches += 1
    if not total_batches:
        raise ValueError("Validation loader produced no batches")
    return total_loss / total_batches


def save_checkpoint(path, model, optimizer, step, epoch, model_config, args, dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "policy_type": "chunked_state_bc",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "model_config": asdict(model_config),
            "tasks": dataset.tasks,
            "observation_stats": dataset.observation_stats.state_dict(),
            "train_args": vars(args),
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="~/datasets/mmbench2_robotics")
    parser.add_argument("--tasks-json", default="tasks.json")
    parser.add_argument("--train-splits", nargs="+", default=["expert"])
    parser.add_argument("--val-splits", nargs="+", default=["val"])
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--domain", choices=["maniskill", "metaworld"])
    selection.add_argument("--tasks", nargs="+")
    parser.add_argument("--context-length", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="policy_training/runs/bc")
    parser.add_argument("--save-every", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)
    data_root = Path(args.data_root).expanduser()
    tasks_json = Path(args.tasks_json).expanduser()
    selected_tasks = discover_tasks(
        data_root,
        args.train_splits,
        domain=args.domain,
        tasks=args.tasks,
    )
    train_dataset = MMBenchStateBCDataset(
        data_root,
        args.train_splits,
        selected_tasks,
        tasks_json=tasks_json,
        context_length=args.context_length,
        chunk_size=args.chunk_size,
    )
    val_dataset = MMBenchStateBCDataset(
        data_root,
        args.val_splits,
        selected_tasks,
        tasks_json=tasks_json,
        context_length=args.context_length,
        chunk_size=args.chunk_size,
        observation_stats=train_dataset.observation_stats,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model_config = ModelConfig(
        observation_dim=train_dataset.observation_dim,
        num_tasks=len(selected_tasks),
        context_length=args.context_length,
        chunk_size=args.chunk_size,
        action_dim=train_dataset.max_action_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = ChunkedBCPolicy(**asdict(model_config)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp = device.type == "cuda"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps({**vars(args), "tasks": selected_tasks}, indent=2) + "\n"
    )

    print(f"tasks: {selected_tasks}")
    print(f"train samples: {len(train_dataset)} ({format_task_summary(train_dataset)})")
    print(f"val samples: {len(val_dataset)} ({format_task_summary(val_dataset)})")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    step = 0
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            observation = batch["observation"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            mask = batch["action_mask"].to(device, non_blocking=True)
            task_index = batch["task_index"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                prediction = model(observation, task_index)
                loss = masked_action_mse(prediction, action, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.detach())
            train_batches += 1
            step += 1

        val_loss = evaluate(model, val_loader, device, amp)
        mean_train_loss = train_loss / max(train_batches, 1)
        print(f"epoch={epoch:04d} train_mse={mean_train_loss:.6f} val_mse={val_loss:.6f}")
        save_checkpoint(
            output / "latest.pt", model, optimizer, step, epoch, model_config, args, train_dataset
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                output / "best.pt", model, optimizer, step, epoch, model_config, args, train_dataset
            )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                step,
                epoch,
                model_config,
                args,
                train_dataset,
            )


if __name__ == "__main__":
    main()

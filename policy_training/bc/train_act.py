"""Train an ACT or ResNet-MLP visual policy on MMBench2 trajectory strips."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .act_model import SimpleACTPolicy, act_loss
from .dataset import discover_tasks
from .eval_act import evaluate_policy
from .resnet_mlp_model import ResNetMLPPolicy, resnet_mlp_loss
from .visual_dataset import MMBenchVisualBCDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rollout_metrics_for_logging(metrics: dict) -> dict[str, float]:
    """Flatten rollout metrics into stable W&B metric names."""
    logged = {
        "eval/mean_success_rate": float(metrics["mean_success_rate"]),
        "eval/mean_return": float(metrics["mean_return"]),
    }
    for task, task_metrics in metrics["tasks"].items():
        prefix = f"eval/task/{task}"
        for key in ("success_rate", "mean_return", "return_std", "mean_length"):
            logged[f"{prefix}/{key}"] = float(task_metrics[key])
    return logged


@torch.no_grad()
def evaluate(model, loader, device, amp: bool) -> float:
    """Evaluate the inference-time prior (z=0), not posterior reconstruction."""
    model.eval()
    total = 0.0
    batches = 0
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        task = batch["task_index"].to(device, non_blocking=True)
        target = batch["action"].to(device, non_blocking=True)
        mask = batch["action_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            output = model(frames, task)
            weights = mask.to(target.dtype)
            mse = ((output["actions"].float() - target.float()).square() * weights).sum()
            mse = mse / weights.sum().clamp_min(1.0)
        total += float(mse)
        batches += 1
    if not batches:
        raise ValueError("Validation loader produced no batches")
    return total / batches


def save_checkpoint(path, model, optimizer, epoch, step, args, tasks) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, SimpleACTPolicy):
        policy_type = "simple_visual_act"
        config_keys = (
            "num_tasks", "chunk_size", "action_dim", "d_model", "n_heads",
            "encoder_layers", "decoder_layers", "latent_dim", "dropout",
            "pretrained_backbone",
        )
    elif isinstance(model, ResNetMLPPolicy):
        policy_type = "resnet_mlp_visual_bc"
        config_keys = (
            "num_tasks", "context_length", "chunk_size", "action_dim", "hidden_dim",
            "task_embedding_dim", "mlp_layers", "dropout", "pretrained_backbone",
        )
    else:
        raise TypeError(f"Unsupported policy class: {type(model).__name__}")
    torch.save(
        {
            "format_version": 1,
            "policy_type": policy_type,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "tasks": tasks,
            "model_config": {key: getattr(model, key) for key in config_keys},
            "train_args": vars(args),
        },
        path,
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--architecture",
        choices=["act", "resnet-mlp"],
        default="act",
        help="Policy head to train on the shared visual BC pipeline",
    )
    p.add_argument("--data-root", default="~/datasets/mmbench2_robotics")
    p.add_argument("--tasks-json", default="tasks.json")
    p.add_argument("--train-splits", nargs="+", default=["expert"])
    p.add_argument("--val-splits", nargs="+", default=["val"])
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--domain", choices=["maniskill", "metaworld"])
    selection.add_argument("--tasks", nargs="+")
    p.add_argument("--context-length", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--kl-weight", type=float, default=10.0)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--decoder-layers", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--task-embedding-dim", type=int, default=64)
    p.add_argument("--mlp-layers", type=int, default=3)
    p.add_argument(
        "--pretrained-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torchvision ImageNet ResNet-18 weights (downloaded if not cached)",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="policy_training/runs/act")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Run live environment evaluation every N epochs; 0 disables it",
    )
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-seed", type=int, default=10_000)
    p.add_argument(
        "--eval-execution-horizon",
        type=int,
        default=1,
        help="Actions to execute from each predicted chunk before replanning",
    )
    p.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log configuration and metrics to Weights & Biases",
    )
    p.add_argument("--wandb-project", default="mmbench2-policy")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-run-name")
    p.add_argument("--wandb-group")
    p.add_argument("--wandb-tags", nargs="*", default=[])
    p.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
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
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)

    if args.architecture == "act":
        model = SimpleACTPolicy(
            num_tasks=len(tasks),
            chunk_size=args.chunk_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
            pretrained_backbone=args.pretrained_backbone,
        )
        loss_fn = lambda output, target, mask: act_loss(
            output, target, mask, kl_weight=args.kl_weight
        )
    else:
        model = ResNetMLPPolicy(
            num_tasks=len(tasks),
            context_length=args.context_length,
            chunk_size=args.chunk_size,
            hidden_dim=args.hidden_dim,
            task_embedding_dim=args.task_embedding_dim,
            mlp_layers=args.mlp_layers,
            dropout=args.dropout,
            pretrained_backbone=args.pretrained_backbone,
        )
        loss_fn = resnet_mlp_loss
    model = model.to(device)
    backbone_parameters = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    policy_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_learning_rate},
            {"params": policy_parameters, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    amp = device.type == "cuda"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({**vars(args), "tasks": tasks}, indent=2) + "\n"
    )
    print(f"train samples: {len(train_data)}, val samples: {len(val_data)}")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as error:
            raise ImportError(
                "W&B logging was requested but wandb is unavailable; install it or pass --no-wandb"
            ) from error
        wandb_config = {
            **vars(args),
            "tasks": tasks,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            tags=args.wandb_tags,
            mode=args.wandb_mode,
            dir=str(output_dir),
            config=wandb_config,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    best = float("inf")
    best_success = -1.0
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_reconstruction = total_kl = 0.0
        batches = 0
        for batch in train_loader:
            frames = batch["frames"].to(device, non_blocking=True)
            task = batch["task_index"].to(device, non_blocking=True)
            target = batch["action"].to(device, non_blocking=True)
            mask = batch["action_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                if args.architecture == "act":
                    output = model(frames, task, target)
                else:
                    output = model(frames, task)
                loss, components = loss_fn(output, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            total_reconstruction += float(components["reconstruction"])
            total_kl += float(components["kl"])
            batches += 1
            step += 1
        val_mse = evaluate(model, val_loader, device, amp)
        epoch_metrics = {
            "epoch": epoch,
            "train/step": step,
            "train/loss": total_loss / max(batches, 1),
            "train/reconstruction": total_reconstruction / max(batches, 1),
            "train/kl": total_kl / max(batches, 1),
            "val/action_mse": val_mse,
            "optimizer/backbone_lr": optimizer.param_groups[0]["lr"],
            "optimizer/policy_lr": optimizer.param_groups[1]["lr"],
        }
        print(
            f"epoch={epoch:04d} loss={total_loss/max(batches,1):.6f} "
            f"recon={total_reconstruction/max(batches,1):.6f} "
            f"kl={total_kl/max(batches,1):.6f} val_prior_mse={val_mse:.6f}"
        )
        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, step, args, tasks)
        if val_mse < best:
            best = val_mse
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, step, args, tasks)
            if wandb_run is not None:
                wandb_run.summary["best/val_action_mse"] = best
                wandb_run.summary["best/val_epoch"] = epoch
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, step, args, tasks
            )
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            metrics = evaluate_policy(
                model,
                tasks,
                device=device,
                episodes=args.eval_episodes,
                seed=args.eval_seed,
                image_size=224,
                execution_horizon=args.eval_execution_horizon,
            )
            metrics.update({"epoch": epoch, "step": step})
            eval_dir = output_dir / "eval"
            eval_dir.mkdir(parents=True, exist_ok=True)
            rendered = json.dumps(metrics, sort_keys=True)
            (eval_dir / f"epoch_{epoch:04d}.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
            with (output_dir / "eval_metrics.jsonl").open("a") as handle:
                handle.write(rendered + "\n")
            print(
                f"rollout_eval epoch={epoch:04d} "
                f"success={metrics['mean_success_rate']:.3f} "
                f"return={metrics['mean_return']:.3f}"
            )
            epoch_metrics.update(rollout_metrics_for_logging(metrics))
            if metrics["mean_success_rate"] > best_success:
                best_success = metrics["mean_success_rate"]
                save_checkpoint(
                    output_dir / "best_success.pt", model, optimizer, epoch, step, args, tasks
                )
                if wandb_run is not None:
                    wandb_run.summary["best/success_rate"] = best_success
                    wandb_run.summary["best/success_epoch"] = epoch
        if wandb_run is not None:
            wandb_run.log(epoch_metrics)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

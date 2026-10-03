"""Train or BC-fine-tune a visual policy with skrl PPO."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from skrl.agents.torch import ExperimentCfg
from skrl.agents.torch.ppo import PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from .env import make_training_env
from .ppo import (
    EvaluationPPO,
    ResNetGaussianPolicy,
    ResNetValue,
    evaluate_ppo,
    initialize_policy_from_bc,
    load_bc_policy_metadata,
    save_portable_checkpoint,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total transitions across all vector environments",
    )
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--sim-backend", default="physx_cuda")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument(
        "--context-length",
        type=int,
        default=1,
        help="Number of current/past RGB observations supplied to actor and critic",
    )
    p.add_argument("--n-steps", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.1)
    p.add_argument("--value-clip", type=float, default=0.1)
    p.add_argument("--target-kl", type=float, default=0.01)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--policy-hidden-dim", type=int, default=512)
    p.add_argument("--actor-layers", type=int, default=3)
    p.add_argument("--task-embedding-dim", type=int, default=64)
    p.add_argument("--initial-log-std", type=float, default=-1.0)
    p.add_argument(
        "--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--bc-checkpoint")
    p.add_argument("--resume", help="Optional full skrl agent checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="auto",
        help="Learner device; auto uses the wrapped environment device",
    )
    p.add_argument("--output", default="policy_training/runs/ppo")
    p.add_argument("--checkpoint-every", type=int, default=100_000)
    p.add_argument("--log-every", type=int, default=5_000)
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-seed", type=int, default=20_000)
    p.add_argument("--eval-output-dir")
    p.add_argument("--eval-video-episodes", type=int, default=3)
    p.add_argument("--eval-video-fps", type=int, default=20)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--wandb-project", default="mmbench2-policy")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-run-name")
    p.add_argument("--wandb-group")
    p.add_argument("--wandb-tags", nargs="*", default=[])
    p.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    return p


def _validate(args: argparse.Namespace) -> None:
    for name in (
        "total_timesteps", "num_envs", "image_size", "context_length", "n_steps", "batch_size"
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    rollout_size = args.n_steps * args.num_envs
    if args.batch_size > rollout_size:
        raise ValueError("batch-size cannot exceed n-steps * num-envs")
    if rollout_size % args.batch_size:
        raise ValueError("n-steps * num-envs must be divisible by batch-size")
    if args.total_timesteps < rollout_size:
        raise ValueError("total-timesteps must cover at least one PPO rollout")


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    _validate(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    eval_root = Path(args.eval_output_dir) if args.eval_output_dir else output / "eval"
    set_seed(args.seed)

    bc_checkpoint = None
    bc_task_index = None
    if args.bc_checkpoint:
        bc_checkpoint, bc_task_index = load_bc_policy_metadata(args.bc_checkpoint, args.task)
        bc_config = bc_checkpoint["model_config"]
        args.policy_hidden_dim = int(bc_config["hidden_dim"])
        args.task_embedding_dim = int(bc_config["task_embedding_dim"])
        args.actor_layers = int(bc_config["mlp_layers"])
        args.context_length = int(bc_config.get("context_length", 1))
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    env = make_training_env(
        args.task,
        num_envs=args.num_envs,
        seed=args.seed,
        image_size=args.image_size,
        context_length=args.context_length,
        sim_backend=args.sim_backend,
    )
    device = env.device if args.device == "auto" else torch.device(args.device)
    if args.task.startswith("ms-") and torch.device(env.device) != torch.device(device):
        env.close()
        raise ValueError(
            f"ManiSkill environment is on {env.device}, but learner device is {device}; "
            "use --device auto to keep rollouts on the simulator device"
        )
    policy = ResNetGaussianPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        image_size=args.image_size,
        context_length=args.context_length,
        hidden_dim=args.policy_hidden_dim,
        task_embedding_dim=args.task_embedding_dim,
        mlp_layers=args.actor_layers,
        initial_log_std=args.initial_log_std,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    value = ResNetValue(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        image_size=args.image_size,
        context_length=args.context_length,
        hidden_dim=args.policy_hidden_dim,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    if bc_checkpoint is not None:
        assert bc_task_index is not None
        initialize_policy_from_bc(policy, bc_checkpoint, bc_task_index)
        print(f"initialized skrl PPO ResNet and actor from {args.bc_checkpoint}")

    metadata = {
        "task": args.task,
        "action_dim": policy.num_actions,
        "image_size": args.image_size,
        "context_length": args.context_length,
        "hidden_dim": args.policy_hidden_dim,
        "task_embedding_dim": args.task_embedding_dim,
        "mlp_layers": args.actor_layers,
        "num_envs": args.num_envs,
        "sim_backend": args.sim_backend,
    }
    memory = RandomMemory(
        memory_size=args.n_steps,
        num_envs=args.num_envs,
        device=device,
    )
    vector_steps = math.ceil(args.total_timesteps / args.num_envs)
    cfg = PPO_CFG(
        rollouts=args.n_steps,
        learning_epochs=args.n_epochs,
        mini_batches=(args.n_steps * args.num_envs) // args.batch_size,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        grad_norm_clip=args.max_grad_norm,
        ratio_clip=args.clip_range,
        value_clip=args.value_clip,
        entropy_loss_scale=args.ent_coef,
        value_loss_scale=args.vf_coef,
        kl_threshold=args.target_kl,
        time_limit_bootstrap=True,
        mixed_precision=args.mixed_precision,
        experiment=ExperimentCfg(
            directory=str(output),
            experiment_name="skrl",
            write_interval=max(args.log_every // args.num_envs, 1),
            checkpoint_interval=(
                max(args.checkpoint_every // args.num_envs, 1)
                if args.checkpoint_every > 0
                else 0
            ),
            wandb=args.wandb,
            wandb_kwargs={
                key: value
                for key, value in {
                    "project": args.wandb_project,
                    "entity": args.wandb_entity,
                    "name": args.wandb_run_name,
                    "group": args.wandb_group,
                    "tags": args.wandb_tags,
                    "mode": args.wandb_mode,
                    "dir": str(output),
                    "config": vars(args),
                }.items()
                if value is not None
            },
        ),
    )
    agent = EvaluationPPO(
        models={"policy": policy, "value": value},
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        cfg=cfg,
    )

    best_success = -1.0
    next_eval = args.eval_every

    def periodic_eval(current_agent: EvaluationPPO, environment_steps: int) -> None:
        nonlocal best_success, next_eval
        if args.eval_every <= 0 or environment_steps < next_eval:
            return
        while next_eval <= environment_steps:
            next_eval += args.eval_every
        step_dir = eval_root / f"step_{environment_steps:09d}"
        metrics = evaluate_ppo(
            policy,
            args.task,
            episodes=args.eval_episodes,
            seed=args.eval_seed,
            image_size=args.image_size,
            output_dir=step_dir,
            video_episodes=args.eval_video_episodes,
            video_fps=args.eval_video_fps,
        )
        metrics["timesteps"] = environment_steps
        eval_root.mkdir(parents=True, exist_ok=True)
        with (eval_root / "eval_metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        save_portable_checkpoint(output / "latest_eval.pt", policy, value, metadata)
        if metrics["success_rate"] > best_success:
            best_success = metrics["success_rate"]
            save_portable_checkpoint(output / "best_success.pt", policy, value, metadata)
        for key, value_item in metrics.items():
            if isinstance(value_item, (int, float)):
                current_agent.track_data(f"Eval / {key}", float(value_item))
        if args.wandb:
            import wandb

            payload = {
                f"eval/{key}": value_item
                for key, value_item in metrics.items()
                if isinstance(value_item, (int, float)) and key != "timesteps"
            }
            if metrics["videos"]:
                payload["eval/videos"] = [
                    wandb.Video(path, fps=args.eval_video_fps, format="mp4")
                    for path in metrics["videos"]
                ]
            payload["eval/environment_steps"] = environment_steps
            # skrl synchronizes TensorBoard to this same run using vector-step
            # indices. Let W&B assign this media log's internal step instead of
            # jumping the shared global step to the transition count.
            wandb.log(payload)
            if wandb.run is not None:
                wandb.run.summary["best/success_rate"] = best_success
        print(
            f"ppo_eval steps={environment_steps} success={metrics['success_rate']:.3f} "
            f"return={metrics['mean_return']:.3f}"
        )

    agent.set_periodic_hook(periodic_eval, num_envs=args.num_envs)
    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={"timesteps": vector_steps, "headless": True},
    )
    if args.resume:
        agent.load(args.resume)
        print(f"resumed skrl agent from {args.resume}")
    try:
        trainer.train()
        agent.save(str(output / "final_agent.pt"))
        save_portable_checkpoint(output / "final.pt", policy, value, metadata)
    finally:
        env.close()
        if args.wandb:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()

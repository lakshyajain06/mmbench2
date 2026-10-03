# Behavior-cloning pretraining

This package contains the standalone offline policy-pretraining pipeline. It is
independent of the optimizer used to train the world model.

The first implementation is a state-based, deterministic action-chunking
baseline. It consumes short histories of the 128-D observations stored in each
task's `.pt` file and predicts several future continuous actions. This is the
alignment and checkpoint baseline to establish before adding image encoders or
a diffusion action head.

Train one policy per task:

```sh
python -m policy_training.bc \
  --tasks ms-push-cube \
  --train-splits expert mixed-small mixed-large \
  --output policy_training/runs/ms-push-cube
```

Train one multi-task policy per domain:

```sh
python -m policy_training.bc --domain maniskill --output policy_training/runs/maniskill
python -m policy_training.bc --domain metaworld --output policy_training/runs/metaworld
```

By default, validation uses the `val` partition and data is read from
`~/datasets/mmbench2_robotics`. Checkpoints contain the task list, model
configuration, and training-set observation statistics.

The next visual model should follow the central Patch Policy idea: retain dense
features from a frozen pretrained visual encoder and use block-causal attention
over space and time instead of pooling each image to one token. That visual
trunk can feed either this deterministic chunk head or a diffusion head.

The initial visual implementation now uses the released MMBench2 tokenizer as
that frozen encoder. It reads the repository's horizontal PNG strips, retains
all 64 tokenizer latent tokens per frame, and applies block-causal attention in
the policy Transformer:

```sh
python -m policy_training.bc.train_visual \
  --domain maniskill \
  --train-splits expert \
  --tokenizer-checkpoint src/checkpoints/combined/tokenizer.pt \
  --output policy_training/runs/maniskill_expert_visual_bc
```

This in-memory strip reader is intended for the small robotics subsets. Use
`src/preprocess_dataset.py` and a sharded policy loader before scaling to the
full MMBench2 corpus or `mixed-large`.

## Simple ACT baseline

The primary visual baseline is a compact ACT policy with a ResNet-18 backbone,
a conditional-VAE style encoder over demonstrated action chunks, and a
Transformer decoder with learned action queries. It consumes the newest RGB
frame and predicts eight future actions:

```sh
python -m policy_training.bc.train_act \
  --domain maniskill \
  --train-splits expert \
  --output policy_training/runs/maniskill_expert_act
```

ImageNet weights are enabled by default and may be downloaded by torchvision
on first use. Pass `--no-pretrained-backbone` for an offline smoke test. At
inference, omit the demonstrated action chunk; the policy uses the prior mean
`z=0`. The checkpoint includes the full ResNet and ACT model.

The same trainer can instead use the smaller ResNet-18 + MLP baseline. It uses
one shared ResNet to encode every frame in an adjustable observation history,
concatenates the ordered frame features with a learned task embedding, and
predicts actions strictly after the newest input frame:

```sh
python -m policy_training.bc.train_act \
  --architecture resnet-mlp \
  --tasks ms-reach \
  --train-splits expert \
  --context-length 4 \
  --output policy_training/runs/ms-reach-resnet-mlp
```

Both architectures share data loading, optimization, validation, checkpointing,
and live rollout evaluation. `--architecture act` remains the default.

## Weights & Biases

W&B logging is opt-in and works for both architectures. Enable it with:

```sh
python -m policy_training.bc.train_act \
  --architecture resnet-mlp \
  --tasks ms-reach \
  --output policy_training/runs/ms-reach-resnet-mlp \
  --wandb \
  --wandb-project mmbench2-policy \
  --wandb-run-name ms-reach-resnet-mlp
```

The run records the full configuration, train/reconstruction/KL losses,
offline action MSE, learning rates, aggregate rollout success and return, and
per-task rollout metrics. Best validation and success values are stored in the
run summary. Use `--wandb-mode offline` on a machine without network access;
training remains entirely local unless `--wandb` is passed.

Implemented components:

1. An MMBench2 dataset adapter with explicit action alignment and episode-safe
   context/chunk windows.
2. Per-task or per-domain task selection.
3. A compact temporal Transformer that predicts action chunks.
4. Masked loss for differing continuous action dimensions.
5. Train/validation loops plus latest, best, and periodic checkpoints.

The ACT trainer also runs live environment rollouts every 10 epochs by default.
Configure this with `--eval-every`, `--eval-episodes`, and
`--eval-execution-horizon`; pass `--eval-every 0` to disable simulator
evaluation. Results are written to `OUTPUT/eval/epoch_NNNN.json` and appended
to `OUTPUT/eval_metrics.jsonl`. `best_success.pt` tracks the checkpoint with
the best mean task success, independently from `best.pt` (offline action MSE).

The same evaluator can be run against any saved ACT checkpoint:

```sh
python -m policy_training.bc.eval_act policy_training/runs/ms-pick-cube/best_success.pt \
  --episodes 50 \
  --output policy_training/runs/ms-pick-cube/eval_50.json
```

It reports per-task success rate, mean/std return, and episode length. The
default execution mode replans every environment step while still predicting
full future-action chunks. At episode reset, missing history is padded by
repeating the first frame.

Do not import code from `evaluation/`; the policy pipeline should depend only
on the released dataset, model, and environment APIs.

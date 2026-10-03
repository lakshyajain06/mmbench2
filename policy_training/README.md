# Policy training

This directory is for policy learning built on top of MMBench2. It is kept
separate from `src/train_dynamics.py`: world-model training should produce
representations and checkpoints, while policy training should be usable with
either those representations or a live environment.

## Layout

- `bc/`: standalone behavior-cloning pretraining from offline trajectories.
- `rl/`: online skrl training against MMBench environments, including native
  ManiSkill GPU vectorization and BC-to-PPO initialization.
- `scripts/`: plotting utilities for policy-training results.
- `tests/`: focused unit tests for the policy-training code.

## Intended flow

1. Pretrain a policy with `policy_training/bc` on MMBench2 trajectories.
2. Save a policy checkpoint with enough metadata to reconstruct the model and
   observation preprocessing.
3. Load that checkpoint in `policy_training/rl` and continue training in a
   Gymnasium-compatible environment.

The BC and RL pipelines should own policy optimization. The optional
`PolicyHeadMTP` in `src/model.py` remains part of dynamics-model training and
is not the standalone policy API for this directory.

## Conventions

- Environments follow the Gymnasium `reset()` and `step()` API.
- Continuous actions are represented as `float32` values in `[-1, 1]` before
  conversion to an environment's native action bounds.
- A policy checkpoint must record its observation mode, action dimension,
  model configuration, and normalization statistics.
- In the existing MMBench2 trajectory format, the action stored with
  observation `t + 1` causes the transition from observation `t` to `t + 1`.
  Dataset adapters must convert this into explicit `(observation, action)`
  training pairs so the indexing convention does not leak into trainers.
- BC and RL should share the same policy module and checkpoint format, allowing
  RL initialization from a BC checkpoint without parameter conversion.

See `POLICY_TRAINING.md` for the supported path, experimental components, and
the ManiSkill PPO recovery plan.

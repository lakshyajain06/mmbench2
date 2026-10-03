# Reinforcement learning

This package fine-tunes the visual ResNet-MLP policy with skrl. The learner is
Gymnasium-oriented, while environment-specific behavior remains at the
boundary in MMBench.

For `ms-*` tasks, one MMBench ManiSkill environment is created with native
`num_envs=N` GPU batching and `physx_cuda`. It retains MMBench's task mapping,
custom registered tasks, object overrides, two simulator steps per policy
action, reward handling, render camera, and episode limits. No `CPUGymWrapper`
or subprocess fan-out is used during training. Non-ManiSkill
MMBench tasks use Gymnasium vectorization through the same skrl learner.

Install the pinned dependency through `environment.yaml`, then fine-tune BC:

```sh
python -m policy_training.rl.train_ppo \
  --task ms-reach \
  --bc-checkpoint policy_training/runs/ms-reach-resnet-mlp/best_success.pt \
  --num-envs 64 \
  --output policy_training/runs/ms-reach-skrl-ppo \
  --wandb \
  --wandb-run-name ms-reach-skrl-ppo
```

The default `--preset maniskill` uses the official RGB PickCube benchmark
schedule: 1,024 environments, 16 rollout steps, 32 minibatches, eight epochs,
50 million transitions, `3e-4` learning rate, `0.8/0.9` discount and GAE, and
the reference clipping and exploration settings. Override the scale for local
smoke tests as needed. `--preset legacy` restores the earlier MMBench2 PPO
defaults for controlled comparisons.

The CLI's `--total-timesteps`, checkpoint intervals, and evaluation intervals
are all measured in total transitions across environments. The default
64-pixel input keeps visual rollout memory tractable; ResNet accepts it even
when the BC checkpoint was trained from 224-pixel strips.

The actor restores the BC ResNet, selected task embedding, full MLP, and the
first action in the BC chunk. PPO adds a Gaussian standard deviation and an
independent visual critic. Actor dropout is omitted and ResNet BatchNorm
statistics are frozen to keep rollout and update log probabilities consistent.
BC checkpoints trained with `--context-length N` make PPO stack the current
frame and `N-1` preceding frames automatically. Both actor and critic receive
that ordered history; resets pad it by repeating the initial observation.

Output includes:

```text
OUTPUT/
  config.json
  run_metadata.json          # Git revision, package versions, and hardware
  final.pt                 # portable policy used by eval_ppo
  final_agent.pt           # full skrl state, including optimizer
  latest_eval.pt
  best_success.pt
  eval/
    eval_metrics.jsonl
    step_000025000/
      metrics.json
      videos/episode_000.mp4
  skrl/
    checkpoints/           # periodic full skrl checkpoints
```

Run reusable evaluation/video recording with:

```sh
python -m policy_training.rl.eval_ppo \
  policy_training/runs/ms-reach-skrl-ppo/best_success.pt \
  --episodes 50 \
  --video-episodes 5 \
  --output-dir policy_training/runs/ms-reach-skrl-ppo/final-eval
```

The task and image size are read from the checkpoint. `--task` can override the
task when deliberately testing transfer to another MMBench environment.
Evaluation reports both success-once and success-at-end; the historical
`success_rate` field remains an alias for success-once.

## World model as a Gymnasium environment

`WorldModelEnv` exposes a tokenizer/dynamics checkpoint pair through the normal
Gymnasium `reset`/`step` API. By default a live MMBench simulator supplies only
the initial frame; all later observations and rewards come from the world model.

```python
from policy_training.rl import make_world_model_env

env = make_world_model_env(
    task="mw-push",
    checkpoint_dir="src/checkpoints/combined_xl",
    observation_mode="rgb",       # "rgb", "latent", or "both"
    max_episode_steps=100,
    device="cuda:0",
)

obs, info = env.reset(seed=0)
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
env.close()
```

For simulator-free evaluation, provide an initial RGB observation instead:

```python
obs, info = env.reset(options={"initial_observation": frame})
```

`frame` may be CHW or HWC, uint8 or floating point. The observation space is
CHW uint8 RGB by default. The action space uses the task's true action dimension
and `[-1, 1]` bounds; actions are padded and masked internally for the universal
16-dimensional model interface.

The dynamics checkpoint's reward head provides `reward`. Since the model has no
termination head, `terminated` is always false and episodes end through
`truncated` at `max_episode_steps`. `info` includes the predicted reward, running
reward, flow-instability score, padded action, and model timestep. Use
`reward_mode="zero"` for reward-free rollouts or `reward_mode="custom"` with a
`reward_fn(frame_chw_uint8, info)` callback for task-specific evaluation. The
environment is directly compatible with Gymnasium evaluators and single-env RL
learners; wrap it with the vectorization mechanism expected by your RL library
when collecting from multiple environments.

Run a trained portable visual PPO policy inside the learned simulator with:

```sh
python -m policy_training.rl.eval_world_model_policy \
  policy_training/runs/ms-reach-skrl-ppo-bc-matched224/best_success.pt \
  --world-model src/checkpoints/combined_xl \
  --episodes 20 \
  --steps 25 \
  --video-episodes 5 \
  --output-dir policy_training/runs/ms-reach-skrl-ppo-bc-matched224/world-model-eval
```

This reports predicted return, real-simulator return/success, action statistics,
and flow instability. Videos show the real simulator on the left and the world
model on the right. Both begin from the exact same RGB frame and then run as
independent closed loops: the simulator policy acts from simulator frames and
the world-model policy acts from generated frames. A world-model success rate
is deliberately omitted because the model does not currently predict task
success or termination. The evaluator accepts the current portable `.pt`
policy format.

# Policy-training status and PPO recovery plan

## Scope

The supported path is ResNet-MLP behavior cloning followed by skrl PPO in a
live MMBench2 environment. The state-only, tokenizer-based, dense visual, and
ACT policy implementations remain in the repository as experimental work.
They are not part of the current PPO baseline and should not be treated as
stable interfaces.

Generated checkpoints, videos, metrics, and TensorBoard logs belong under an
ignored `policy_training/runs/` directory or in external experiment storage.
The existing local outputs under `training/runs/` are retained in place and
ignored; they are not source code.

## Current diagnosis

The unsuccessful manipulation runs are not directly comparable to ManiSkill's
successful PPO configuration:

- ManiSkill's RGB policy includes simulator state by default and combines a
  NatureCNN visual representation with a learned state representation. The
  current actor is RGB-only, and the actor and critic use separate ResNet-18
  trunks, totaling roughly 24 million parameters.
- ManiSkill's PickCube examples use millions more transitions and substantially
  more parallel environments and PPO updates. The current matched runs used
  four environments, two PPO epochs, and a one-million-transition budget.
- Important coefficient differences include learning rate (`3e-4` versus
  `1e-5`), PPO epochs (`8` versus `2`), ratio clip (`0.2` versus `0.1`), target
  KL (`0.2` versus `0.01`), discount (`0.8` versus `0.99`), GAE lambda (`0.9`
  versus `0.95`), and initial log standard deviation (`-0.5` versus `-1.0`).
- Time-limit bootstrapping is currently unsafe with auto-reset: a timed-out
  transition can bootstrap from the next episode's initial image instead of
  the prior episode's final observation.
- MMBench2 also differs in camera, controller, action repeat, and horizon. These
  differences must be controlled rather than bundled into one comparison.
- Failed pick-task BC checkpoints had zero rollout success, whereas reach
  checkpoints began with nonzero success and were improved by PPO.

A promising deployment-compatible architecture is an RGB-only actor with a
privileged-state critic during training. For BC-initialized PPO, separate
learning rates may also be appropriate for the pretrained actor backbone and
new critic/head parameters.

## Recovery sequence

### 1. Reproduce the reference

Run ManiSkill's unmodified state PPO and then RGB-plus-state PPO on PickCube.
Record the exact ManiSkill revision, environment, command, seed, evaluation
definition, and result. Do not begin broad MMBench2 sweeps until the state
baseline succeeds reliably.

### 2. Repair PPO correctness

Preserve final observations across partial auto-resets and use them for timeout
bootstrap values. Add regression tests for mixed partial resets and timeout
GAE. Verify action/log-probability semantics at action bounds and exact
train/evaluation parity for camera, controller, action repeat, horizon, reward,
object identity, and success aggregation.

### 3. Establish an attainable MMBench2 baseline

Start from ManiSkill's PPO coefficients and update schedule. Compare a compact
or shared visual encoder before training two independent ResNet-18 trunks from
sparse reward. Run scratch and BC-initialized variants separately, use at least
three seeds, and use a transition budget comparable in order of magnitude to
the reference. The gate is repeatable nonzero PickCube success.

### 4. Run controlled ablations

Change one axis at a time: actor/critic state access, NatureCNN versus ResNet-18,
shared versus independent encoders, scratch versus BC initialization, native
control rate versus action repeat, sensor versus render camera, controller
type, and discount/GAE settings. Every run should save its resolved config,
code revision, dependency versions, seed, curves, checkpoints, and both
success-once and success-at-end metrics.

### 5. Expand task coverage

After PickCube is stable, proceed through reach and fixed-object pick tasks,
xArm6 variants, then push, pull, stack, place, and tool tasks. Multi-task and
transfer experiments come after reproducible single-task baselines.

## Deferred work

- Deep refactoring or a unified policy interface.
- Promoting the experimental policy families to supported status.
- Slurm scripts and HPC-specific paths; add these when the target cluster and
  requested jobs are known.
- New configuration frameworks, CI, formatting systems, or broad test suites.

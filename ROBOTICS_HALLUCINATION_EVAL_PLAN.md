# Small robotics hallucination screen

## Goal

Find **5–8 seen robotics tasks** where the local `combined` world-model checkpoint produces useful, physically plausible predictions **most of the time**, but occasionally makes a clear hallucination. Errors may occur at any rollout horizon; a short-good/long-bad pattern is not required. The eight tasks below are screening candidates, **not measured successes**.

## First-pass tasks

| Task | Interaction to inspect |
| --- | --- |
| `ms-reach`, `mw-reach` | Simple arm motion; competence baselines |
| `ms-push-cube`, `mw-push` | Object contact and unsupported motion |
| `mw-button-press` | Activation before contact |
| `mw-drawer-open`, `rd-open-drawer` | Drawer motion and contact/geometry |
| `ms-pick-cube` | Grasp, lift, and object detachment |

If fewer than five qualify, screen `mw-door-open`, `rd-open-slide`, `mw-pick-place`, or `ms-stack-cube` one at a time. Do not assume complex backups will perform better.

## Four analysis rounds

Run the **same recorded-action rollout protocol** for each task and partition: start from a real frame and predict 1, 4, 8, and 16 transitions. The action at observation `t+1` caused `obs_t → obs_{t+1}`; index 0 has NaN action. Log error in **latent space** (predicted state versus the encoded real frame) and **reconstructed image space** (decoded prediction versus the real frame), alongside tokenizer-only reconstruction error as a baseline. Report both errors, the fraction of useful/plausible windows, the fraction with a concrete physical error, and the first error frame **by task, horizon, episode, and partition**. Count unsupported object motion, detachment, or opening before contact as hallucinations only when the real trajectory and actions support that label. Inspect several rollout seeds so one unlucky stochastic sample does not define task quality.

| Round | Data | Purpose |
| --- | --- | --- |
| 1 | `expert` | Check whether the checkpoint follows clean, high-return interactions. |
| 2 | `mixed-small` and `mixed-large` | Check harder behavior and whether error rates change with trajectory quality; sample an equal number of episodes per task/subpartition for the first comparison. |
| 3 | `val` | Screen candidates and calibrate event-label and predictor thresholds using held-out trajectories. |
| 4 | `test` | Confirm the selected tasks and thresholds on separate episodes. Do not revise selection from this final check without reporting it. |

`expert` and mixed data are diagnostic rounds; use `val` for selection and reserve `test` for confirmation. Compare rates within a task rather than pooling the very different reward scales or episode lengths across domains. Keep tasks whose predictions are useful in a clear majority of sampled windows and whose hallucinations occur sometimes across episodes or seeds, **at any tested horizon**. A provisional screen is about 5–30% affected windows, with the actual cutoff set from measured results. Do not select a task solely because its errors increase with horizon.

**Step 2, after task selection:** log metrics that show whether successful predictions cover diverse robot and object states, rather than only a narrow set of easy situations. Choose those diversity metrics after the simple screen; they are not part of the current task filter.

## Current data findings and run status

The [trajectory audit](ROBOTICS_DATASET_ROUND_RESULTS.md) records episode counts and mean returns for all five partitions. Each eight-task `.pt` file and its PNG strips are present under `~/datasets/mmbench2_robotics/`. `expert`, `mixed-small`, `val`, and `test` have 20 episodes per task; `mixed-large` has 200. `test` has expert-like recorded returns, while `val` and mixed data usually have lower returns. These are **dataset results, not checkpoint scores**.

The finetuned checkpoint pair is present at `src/checkpoints/combined/{tokenizer,dynamics}.pt`. `nvidia-smi` still cannot communicate with the driver, so no model rollout, fidelity score, or hallucination rate is available. Once inference works, preprocess only the eight tasks for the partitions above with `src/preprocess_dataset.py` at 224×224, pilot one episode per round/task to verify loading and alignment, then scale the rollout comparison. After the task set is fixed, evaluate `u_r`, `u_f`, and `u_s` from `src/uncertainty.py` against independently labeled event onsets.

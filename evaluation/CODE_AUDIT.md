# Recorded-action evaluation audit

## Model design relevant to interpreting evals

The loaded `combined/dynamics.pt` checkpoint **does not use task text embeddings
to predict spatial latents or frames**. This is how its architecture routes
information, not an embedding lookup error in the evaluator. `evaluation/models.py` passes the
task-specific embedding, and `Dynamics.task_proj` maps distinct embeddings to
distinct agent-token inputs. But `SpaceSelfAttentionModality(mode="wm_agent")`
prevents every non-agent query (including spatial tokens) from reading agent
tokens. Time attention mixes the same token position across timesteps only, and
the spatial prediction head reads spatial tokens. Thus there is no path from
language to a predicted latent. Task-specific initial frames and actions still
affect predictions.

Controlled checkpoint tests held the real starting frame, action, initial
noise, and all model settings fixed:

| Embedding swap | Agent output max change | Direct spatial max change | Sampled latent max change (cache off / on) |
| --- | ---: | ---: | ---: |
| `mw-reach` → `mw-push` | 34.60 | 0.0 | 0.0 / 0.0 |
| `ms-push-cube` → `ms-stack-cube` | 60.99 | 0.0 | 0.0 / 0.0 |

Reproduce with `python evaluation/check_task_conditioning.py` or choose another
same-action-dimension pair with `--task-a` and `--task-b`. Changing the attention
mask only at inference would use an untrained information path; a language-
conditioned spatial model needs an architecture change and retraining.

The validation scores are therefore valid *recorded-frame/action prediction*
errors, but they do not measure task-text understanding. This architecture
behavior does not by itself explain why the validation errors are similar.

In the MetaWorld source metadata, 53.75% of finite validation action rows are
exactly identical at the same row index for `mw-reach` and `mw-push`, versus 0%
of the corresponding expert rows. This suggests the validation action regime
shares substantial structure across those tasks. We have not established the
dataset-generation cause. For the ten MS/MW tasks in the quick comparison, the
16-step validation latent error divided by pooled task-state standard deviation
ranges from 0.86 to 1.14; the similar denominator scales and validation action
structure warrant caution when treating those ratios as evidence of similar
skill competence.

The quick-run image and latent errors are computed as implemented, but raw whole-image RMS is not a task-success or hallucination metric. The original quick runs also sampled a different start frame for each horizon, which confounds horizon comparisons.

## Checks

- The dataset and training code both use the action stored at observation `t+1` for the transition `obs[t] → obs[t+1]` (`wm_dataset.py`, `train_dynamics.py`, and `runner.py`). The evaluation action slice is aligned with this convention.
- An independent source-data comparison found zero action-slice mismatches among all 1,760 saved validation windows (`leo_val_quick` and `leo_more_msmw_val_quick`), with 1,760 unique window identities. This checks the saved inputs, not just the slicing code.
- On an `ms-stack-cube` episode, the batched `rollout_windows` and `rollout_many` predictions were identical (`max_abs=0.0`). A single-sample rollout differed by at most `0.00293` in latent value; this small batch-dependent difference did not change the conclusion.
- Recomputing a saved `ms-stack-cube` endpoint image RMS from its PNG gave `0.07565`, versus `0.07559` in `steps.jsonl` (8-bit image quantization accounts for the small difference). Recomputed per-task aggregate means matched the CSV exactly.
- With the same seed, replacing recorded actions with zeros changed the endpoint decoded image by RMS `0.0861` for one `ms-stack-cube` window and `0.0719` for one `mw-pick-place` window. The model is using its action input.

## Interpretation issues found

- The original `ms-stack-cube` quick run chose starts from frames `1–24` for horizon 1, but only `1–9` for horizon 16. Different horizons therefore sampled different task phases. `sampling.paired_horizons=true` now reuses the same valid start across horizons. The original run remains unchanged for reproducibility.
- On the original expert `ms-stack-cube` horizon-4 sample, model image RMS was `0.0652`; copying the start frame scored `0.0367`. On a *different, matched-start* expert sample, the model scored `0.0612` and copy-start scored `0.0860`. The reversal demonstrates that start selection matters. These are distinct samples, not before/after model improvements.
- A reviewed validation `ms-stack-cube` strip (`leo_more_msmw_val_quick/visuals/val/ms-stack-cube/ep6_s4_h16_seed23.png`) shows the real red cube lifted while the prediction leaves it on the table, despite a whole-image endpoint RMS of about `0.083`. Large static regions and tokenizer reconstruction error dilute object-level mistakes.
- The interactive session uses user actions and a warm start; quick runs use recorded actions and fresh-noise shortcut rollouts. Their behavior should not be equated.

Use `python evaluation/audit_baselines.py --run evaluation/runs/<run>` from `mmbench2` to reproduce the copy-start comparison. The paired-start diagnostic runs are `leo_matched_expert_sanity` and `leo_matched_val_sanity`. Neither metric determines physical hallucination without event review.

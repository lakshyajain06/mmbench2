# Robotics hallucination evaluation

Implements the recorded-action screen in [the plan](../ROBOTICS_HALLUCINATION_EVAL_PLAN.md). The JSON config isolates dataset roots and frame format, task lists and metadata, checkpoint/model adapter, rollout schedule, sample budget, seeds, review thresholds, and output location. Override any value at runtime with `--set 'section.key=JSON_VALUE'`; use another JSON config for a different experiment. New datasets implement `episode_ids` and `load_episode` in `datasets.py` and register in `datasets.ADAPTERS`; new models implement `encode`, `decode`, and `rollout` in `models.py` and register in `models.ADAPTERS`. Set each adapter name in JSON.

**Model design note:** The current combined dynamics checkpoint receives task
text embeddings, but they cannot influence its spatial latent prediction under
the `wm_agent` attention mask. A controlled embedding-swap check produces
identical predicted latents with and without KV cache. Frames and actions still
condition predictions. See [the audit](CODE_AUDIT.md) and reproduce with
`python evaluation/check_task_conditioning.py` from `mmbench2`.

Use the repository Python environment (`environment.yaml`). The default dataset reads the original 224-pixel PNG strips directly. To use preprocessed frames, run `python cli.py --config configs/robotics.json prepare --partitions expert mixed-small mixed-large val test` (or pass specific tasks), then set `dataset.frames="shards"`. Preparation calls `src/preprocess_dataset.py` at 224×224 for only the requested tasks and writes under the configured `shards_root`. Raw strips split at `dataset.strip_size` (4,008 by default); preprocessed shards use their JSON index, so the loader does not assume every last shard has 4,096 frames.

From `mmbench2/evaluation`:

```sh
python cli.py --config configs/robotics.json run --pilot --partitions expert
python cli.py --config configs/robotics.json run --partitions expert mixed-small mixed-large val
python cli.py --config configs/robotics.json review-sheet
python cli.py --config configs/robotics.json select --labels runs/robotics/review.csv
python cli.py --config configs/robotics.json run --partitions test
python cli.py --config configs/robotics.json confirm --labels runs/robotics/review.csv
```

Pilot runs one episode and one start per horizon under the output `pilot/` directory, keeping its sampling separate from the full screen. Each rollout begins at a real observation; `action[start+1]` causes the first predicted transition. Every seed is a separate stochastic sample. `steps.jsonl` records per-transition latent RMS versus encoded real observation, decoded-image RMS versus the real image, and tokenizer-only image RMS. `windows.jsonl` contains review identities and the exact recorded action/reward slices used for each prediction. The review CSV asks for `useful_plausible=yes/no`, a concrete `event`, its first transition, and `real_action_support=yes`. Mark `useful_plausible=no` when a supported physical error is present; leave event blank for no supported physical error. Accepted events are `unsupported_object_motion`, `detachment`, `opening_before_contact`, `activation_before_contact`, and `other_physical_error`. Inspect the real and predicted trajectories before labeling; image or latent error alone does not establish hallucination.

By default, starts are sampled independently for each horizon. This covers more episode phases, but a 1-step versus 16-step curve also reflects different starts. Set `sampling.paired_horizons=true` for matched-horizon comparisons: each sampled start must permit the longest valid configured horizon and is reused for every horizon. Keep paired and unpaired runs in separate output directories. Whole-image RMS can hide small object or contact errors; compare against a copy-the-start-frame baseline and inspect the predicted strips before inferring task competence.

For the 56-task random-start rerun, use `configs/random_start_msmw_val.json`. It sets `start_mode="random"`, `paired_horizons=true`, and `random_seed_scope="task_episode"`: every task/episode draws one start uniformly from the frames that leave room for the 16-step horizon, and that start is reused at horizons 1/4/8/16. Seed 2026 makes the selection reproducible while giving different tasks independent draws. Run the two domains into fresh directories on separate GPUs:

```sh
PYTHON_BIN=/home_shared/grail_lakshya/miniforge3/envs/mmbench2/bin/python \
  bash evaluation/run_random_start_msmw.sh ms 0
PYTHON_BIN=/home_shared/grail_lakshya/miniforge3/envs/mmbench2/bin/python \
  bash evaluation/run_random_start_msmw.sh mw 1
```

The launcher refuses to append when its output directory already contains result files. The default outputs are `runs/random_start_msmw_val_ms` and `runs/random_start_msmw_val_mw`.

For a cross-domain comparison with different episode lengths, set `sampling.phase_fraction` together with `paired_horizons=true` and `windows_per_episode=1`. A value of `0.16` starts a 26-frame ManiSkill episode at frame 4 and a 101-frame MetaWorld episode at frame 16. This matches relative *time in the episode*, not a human-labeled contact or interaction event. The dedicated `configs/matched_phase_val_016.json` uses that rule for the ten MS/MW validation tasks.

The expanded validation-only comparison uses `configs/expanded_msmw_val_016.json` for 16 additional tasks (eight per domain). Download a chosen task set with `python src/download_dataset.py --local_dir ~/datasets/mmbench2_robotics --subset val --tasks <task-name> [<task-name> ...]` from `mmbench2`; the downloader checks that each requested task has metadata and a first PNG strip. Run the config with `python evaluation/cli.py --config evaluation/configs/expanded_msmw_val_016.json run --partitions val` from `mmbench2`. Use a fresh output directory when rerunning. Results and the pooled 26-task comparison are documented in [the expanded validation report](EXPANDED_MSMW_VAL_RESULTS.md).

`configs/more_msmw_manip_val_016.json` extends that validation-only comparison with the remaining 15 ManiSkill manipulation tasks and 15 MetaWorld tasks sampled from its remaining inventory with seed 2026. It keeps the same checkpoint, sampler, starts, horizons, episodes, and seeds. The nine remaining ManiSkill locomotion tasks are outside this manipulation comparison. Results for all 56 evaluated tasks are in [the additional validation report](MORE_MSMW_MANIP_VAL_RESULTS.md). `plot_per_task.py --summary <normalized-summary.csv> --output <directory>` creates one curve plot per evaluated task and a browsable index.

For an existing run, `python audit_baselines.py --run runs/<run-name>` writes `audit_baselines.csv` with endpoint model image RMS, tokenizer-only RMS, and the RMS of copying the real start frame to the endpoint. A model-to-copy ratio above one means the model is worse than that trivial baseline for the sampled windows. This is a fidelity check, not a physical-event label.

For cross-task latent comparisons, `normalize_latent.py --runs <expert-run> <val-run> ... --output <new-directory>` re-encodes the sampled real start and endpoint frames with the run's tokenizer. It writes two dimensionless measures at each task/partition/horizon: `normalized_latent_error` divides mean model latent RMS by mean real start-to-end latent RMS (a copy-start baseline), while `error_over_task_state_std` divides by the standard deviation of real latent states pooled across the supplied runs for that task. The former uses a ratio of group means rather than averaging potentially unstable per-window ratios; below one means the model beats copying the start latent on average. The latter holds one denominator fixed across expert and val. The output includes per-window values, a summary CSV, and plots. Set `--prefix ms- --prefix mw-` to restrict the report to ManiSkill and MetaWorld. Use only the selection partitions to set the state scale when reserving test for confirmation. These ratios compare latent fidelity, not task success or physical hallucination.

For a rerun with different sampled starts, pass `--reference-scales <prior-summary.csv>` to keep each task's state-standard-deviation denominator fixed instead of estimating a new scale from the rerun's endpoints. The new run still recomputes its copy-start baseline from its actual start and endpoint frames.

`report` produces task/horizon/partition summaries plus per-window averages and first-error frames. `select` uses complete val labels and writes `selection.json`; test rollout and confirmation require the locked task list and thresholds. The provisional 5–30% affected-window range is configurable and should be set from measured val results before locking. Compare each task across partitions; do not pool reward scales. If fewer than five tasks qualify, add backup tasks to the config one at a time and rerun diagnostic/val rounds. Task-state diversity and `u_r`/`u_f`/`u_s` calibration are the plan's subsequent step after this screen; their metrics and thresholds require independently labeled event onsets.

The output path defaults to `evaluation/runs/robotics`. Use a fresh output directory for each config or run: non-pilot `run` appends to JSONL files. Keep raw predictions and review decisions together with the manifest for reproducibility.

Each window also saves a three-row PNG under `visuals/`: real frames, decoded prediction, and tokenizer-only reconstruction. The first predicted frame is aligned one column after the real start frame. Set `save_visuals=false` to save disk space after visual review.

To inspect a completed run, generate an offline interactive dashboard and a static summary figure:

```sh
python visualize.py --run runs/leo_expert_quick --summary runs/leo_expert_quick_summary/metrics_summary.json
```

The matching 20-episode validation shortcut run uses `configs/leo_expert_quick.json` with `--set 'output="../runs/leo_val_quick"' run --partitions val`; aggregate its complete output with `python aggregate_metrics.py --runs runs/leo_val_quick --output runs/leo_val_quick_summary --expected-windows 1280`, then pass those two paths to `visualize.py`.

This writes `dashboard.html` and `summary.png` in the run directory. The dashboard filters by task and horizon, compares episode/start/seed windows, and opens their three-row review images and recorded actions. It requires a complete `windows.jsonl`/`steps.jsonl` pair and saved review images. Both visuals show endpoint image RMS and latent RMS by horizon; image RMS also compares the model with tokenizer-only reconstruction at the longest horizon. Latent RMS is the error between the predicted latent and the encoded real frame; it has a different scale from image RMS. These metrics are not human-labeled physical hallucinations. Open the HTML directly in a browser; it needs no server.

After selecting tasks, run `python cli.py --config configs/robotics.json calibrate --labels runs/robotics/review.csv`. This calibrates thresholds for motion-normalized `u_r` and `u_f`, and cross-seed `u_s`, using only independently labeled val event onsets. Steps after an onset are excluded from calibration; pre-onset steps count as negatives. `confirm` reports precision, recall, and F1 on test onset labels using these fixed val thresholds. The threshold rule currently maximizes val F1 and can be changed in `predictors.py` for another calibration objective.

`python cli.py --config configs/robotics.json diversity --partition val` describes the selected tasks' distribution of real latent motion, real image motion, and action magnitude by episode/start-frame coverage. These are descriptive state/interaction coverage checks after selection; they do not affect the initial task filter. Run it on test as well to compare coverage with the held-out trajectories.

For a raw-metric run across the original test candidates before val labels exist, use `run --metrics-only-test --partitions test`; the manifest marks this as exploratory. This does not unlock `confirm` or use test metrics to select tasks. The full split-GPU run can be resumed with `run_full_partitions.sh`; `await_and_aggregate.sh` checks both workers and writes `runs/combined_full/metrics_summary.csv` and `.json` only after all 38,400 configured windows are complete. Human review is still required for usefulness and physical hallucination rates.

While the full split-GPU run is active, `python evaluation/status.py` (from `mmbench2`) prints completed windows by partition. The two workers log to `evaluation/runs/partition_eval_{a,b}/full.log`; the final aggregate appears at `evaluation/runs/combined_full/metrics_summary.csv` after both `full.exit` files report success. The expected full sample is 38,400 windows.

The default screen deliberately uses `model.schedule="finest"` for the combined checkpoint (`k_max=64`), so each transition runs 64 denoising updates. The browser interactive defaults to shortcut `eval_d=0.125` (8 updates, with its `tau_init=0.125` warm start skipping the first), so its apparent per-frame speed is not a timing baseline for this screen. The evaluation also encodes real frames, decodes/re-encodes predictions, scores three seeds, and saves review images. Compare result quality across sampler settings only in separate runs with distinct output directories.

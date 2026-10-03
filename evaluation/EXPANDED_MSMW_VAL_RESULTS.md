# Expanded ManiSkill / MetaWorld validation evaluation

This run adds eight ManiSkill and eight MetaWorld tasks to the earlier matched-
time ten-task screen. **Only the `val` partition was downloaded or evaluated.**
The downloaded 16 metadata files and 16 corresponding PNG strips (32 files)
matched the published MMBench2 SHA-256 checksums. Each task has 20 episodes;
seven new ManiSkill tasks have 25 transitions per episode,
`ms-pull-cube-tool` has 50, and all eight new MetaWorld tasks have 100.
Metadata frame counts equal decoded strip frame counts, and actions follow the
expected leading-NaN/finite-transition convention.

The model is the published `combined` checkpoint pair. The protocol uses
recorded actions, a start at 16% of episode time, paired starts for horizons
1/4/8/16, 20 episodes per task, and two noise seeds. The 16 new tasks produced
2,560 complete windows and 18,560 scored transitions. The earlier ten tasks
and the new tasks together produced 4,160 complete **validation** windows and
30,160 transitions. Normalization below divides each task's model latent RMS
by the standard deviation of encoded real start/end states sampled from these
validation runs; all 26 tasks use the same scale-estimation procedure.

| Cohort | Horizon | ManiSkill normalized RMS | MetaWorld normalized RMS | ManiSkill raw RMS | MetaWorld raw RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 new tasks | 1 | 0.693 | 0.485 | 0.252 | 0.174 |
| 16 new tasks | 4 | 0.862 | 0.781 | 0.313 | 0.279 |
| 16 new tasks | 8 | 0.940 | 0.885 | 0.341 | 0.317 |
| 16 new tasks | 16 | 1.051 | 1.008 | 0.380 | 0.361 |
| All 26 tasks | 1 | 0.658 | 0.471 | 0.247 | 0.170 |
| All 26 tasks | 4 | 0.810 | 0.754 | 0.303 | 0.271 |
| All 26 tasks | 8 | 0.880 | 0.868 | 0.329 | 0.313 |
| All 26 tasks | 16 | 0.991 | 0.988 | 0.370 | 0.356 |

Means give each task equal weight; every task contributes 40 windows per
horizon. The eight new ManiSkill tasks have higher raw and state-normalized
latent error at every horizon, consistent with the user's expected direction.
Pooling all 26 tasks leaves ManiSkill worse at shorter horizons, while the
16-step normalized means are nearly tied. Task choice therefore materially
affects the domain conclusion.

The new tasks' 16-step state-normalized errors range from 0.862
(`ms-pull-cube-tool`) to 1.181 (`ms-poke-cube`) for ManiSkill, and from 0.962
(`mw-button-press-wall`) to 1.069 (`mw-faucet-open`) for MetaWorld.
`ms-pick-cube-so` illustrates why raw and normalized error should both be
shown: raw RMS is 0.238, but its task-state-normalized ratio is 1.139 because
its sampled real-state variation is small.

This is a latent prediction-fidelity comparison, not a task-success or
physically labeled hallucination rate. The 16% start matches relative episode
time, not contact stage. Sixteen prediction steps also cover different
fractions of the domains' episodes; `ms-pull-cube-tool` is longer than the
other new ManiSkill tasks.

Artifacts:

- `runs/expanded_msmw_val_016_report/domain_comparison.png`: domain curves.
- `runs/all_msmw_phase_val_016_per_task/index.html`: browsable graphs for all
  26 validation tasks, with raw, task-state-normalized, and copy-start-normalized
  error curves across horizons. `overview.png` compares the 16-step task values.
- `runs/expanded_msmw_val_016_report/domain_means.csv`: numbers behind the plot.
- `runs/all_msmw_phase_val_016_normalized/summary.csv`: all per-task normalized metrics.
- `runs/expanded_msmw_val_016_summary/metrics_summary.csv`: new-task raw metrics.
- `runs/expanded_msmw_val_016_ms/dashboard.html` and
  `runs/expanded_msmw_val_016_mw/dashboard.html`: per-window visual review.
- `runs/expanded_msmw_val_016/download_audit.json`: selected tasks and file checks.

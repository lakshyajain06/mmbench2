# Validation rerun at a matched episode time

The rerun used the exact published `combined` tokenizer/dynamics checkpoint
pair and the original MMBench2 validation files. It kept the shortcut schedule,
recorded actions, 20 episodes per task, two noise seeds, and horizons 1/4/8/16.
`sampling.phase_fraction=0.16` starts every ManiSkill episode at frame 4 and
every MetaWorld episode at frame 16, approximately 16% through each episode;
the same start is reused across horizons. This is **relative episode time**,
not a manually verified contact/event stage.

The split runs completed 1,600 unique windows and 11,600 transitions across
four ManiSkill and six MetaWorld tasks. The aggregate had no duplicate or
incomplete windows. Normalized errors below use the *same per-task real-latent
standard deviations* as the previous expert+val comparison, so the state-scale
denominator did not change when the starts changed.

| Val horizon | Original ManiSkill | Original MetaWorld | Matched ManiSkill | Matched MetaWorld |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.594 | 0.498 | 0.605 | 0.458 |
| 4 | 0.722 | 0.795 | 0.727 | 0.725 |
| 8 | 0.825 | 0.891 | 0.782 | 0.855 |
| 16 | 0.919 | 1.066 | 0.897 | 0.972 |

At 16 steps, the domain gap in normalized error shrank from 0.147 to 0.075
after changing the starts. Mean **raw** latent RMS in the matched run is
0.350 for ManiSkill and 0.349 for MetaWorld—essentially equal. The mean
real-latent state-standard-deviation denominator is about 0.391 for
ManiSkill and 0.359 for MetaWorld. Thus the remaining normalized ranking is
largely about the task-specific scale of real latent variation, rather than a
large raw prediction-error gap. The model/copy-start latent RMS ratio at 16
steps is 0.719 versus 0.774, respectively.

These data confirm the selected ManiSkill tasks have four times fewer recorded
transitions per task than MetaWorld tasks, but they do **not** confirm worse
ManiSkill latent prediction on this checkpoint. Raw dataset size cannot be
read as effective optimization exposure because the checkpoint records
targeted task sampling. Further, 16 transitions span 64% of a 25-step
ManiSkill episode but only 16% of a 100-step MetaWorld episode. Matching
relative start time removes one confound, but not that horizon difference or
differences in task physics and visual variation.

Artifacts:

- `runs/matched_phase_comparison/comparison.png`: original versus matched plot.
- `runs/matched_phase_comparison/comparison.csv`: per-task and horizon values.
- `runs/matched_phase_comparison/domain_means.json`: domain averages.
- `runs/latent_normalized_matched_phase_val_016/summary.csv`: normalized run.
- `runs/leo_matched_phase_val_016_summary/metrics_summary.csv`: raw run.

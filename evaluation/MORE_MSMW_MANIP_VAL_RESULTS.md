# Additional ManiSkill / MetaWorld validation evaluation

This evaluation adds **30 validation tasks** to the earlier 26-task matched-phase
screen. It includes all 15 remaining ManiSkill *manipulation* tasks in the
published validation files and 15 MetaWorld tasks sampled without replacement
from the 35 remaining names with `random.Random(2026)`. Selection and the
download audit are recorded in
`runs/more_msmw_manip_val_016/download_audit.json`. All 60 downloaded `.pt`
and PNG files matched the published LFS SHA-256 hashes; strip frame counts
matched metadata. The new tasks each have 20 validation episodes, with 25
transitions per ManiSkill episode and 100 per MetaWorld episode. No expert or
test data were downloaded or evaluated for this extension.

The model, checkpoint, shortcut schedule (`eval_d=0.125`), recorded actions,
20 episodes/task, paired 1/4/8/16-step horizons, relative start at 16% of each
episode, and two noise seeds match the earlier validation runs. The two new
workers completed **4,800 windows and 34,800 scored transitions**. The
six-run aggregate checked **8,960 windows and 64,960 transitions** across
all **56 tasks**, with no duplicate or incomplete windows. Every result below
is validation-only and gives each task equal weight; each task contributes
40 windows per horizon.

| Cohort at 16 steps | Tasks (MS/MW) | Raw latent RMS (MS/MW) | RMS / task-state SD (MS/MW) | RMS / copy-start RMS (MS/MW) |
| --- | ---: | ---: | ---: | ---: |
| Previous | 12 / 14 | 0.370 / 0.356 | 0.991 / 0.988 | 0.788 / 0.812 |
| New | 15 / 15 | 0.367 / 0.349 | 0.972 / 0.989 | 0.745 / 0.819 |
| All | 27 / 29 | **0.368 / 0.352** | **0.980 / 0.989** | **0.764 / 0.816** |

For all 56 tasks, task-state-normalized means are 0.635/0.470 (MS/MW) at
one step, 0.797/0.760 at four, 0.870/0.879 at eight, and 0.980/0.989 at
16. Raw latent error remains higher for ManiSkill at every horizon. The
16-step task-state-normalized gap is only −0.008 (MS minus MW); resampling
the evaluated tasks within each domain gives an approximate 95% interval
of **−0.048 to +0.032**, which spans both rankings. This is a sensitivity
check for the chosen tasks, not a population confidence interval: the
ManiSkill set contains many pick-object variants, and the task selection is
not a random sample of all robotics problems.

At 16 steps in the all-task aggregate, decoded-image RMS is 0.083 for
ManiSkill and 0.089 for MetaWorld, while tokenizer-only reconstruction RMS
is 0.032 and 0.021, respectively. Image and latent measures therefore do
not yield one unambiguous domain ranking; the tokenizer baseline also differs.

Normalization divides each task's mean model latent RMS by the standard
deviation of real encoded states sampled at the matched start and endpoints
in these validation runs. The copy-start ratio divides by the mean real
start-to-end latent change. A low-variation task can therefore have a high
state-normalized score despite a low raw error. These measures assess
prediction fidelity, not task success or labeled physical hallucination.
The 16-step horizon covers 64% of a 25-step ManiSkill episode but only 16%
of a 100-step MetaWorld episode; matching relative start time does not fix
that difference.

The published validation split has 36 ManiSkill and 49 MetaWorld tasks.
This report covers 27 ManiSkill manipulation tasks and 29 MetaWorld tasks.
The nine unevaluated ManiSkill tasks are locomotion/control tasks (ant,
anymal, cartpole, hopper); 20 MetaWorld validation tasks remain unevaluated.

Artifacts:

- `runs/more_msmw_manip_val_016_report/domain_comparison.png` and
  `domain_means.csv`: previous/new/all domain curves and numbers.
- `runs/all_msmw_manip_phase_val_016_per_task/index.html`: browsable graphs
  for all 56 tasks; `overview.png` compares 16-step task values.
- `runs/all_msmw_manip_phase_val_016_normalized/summary.csv`: per-task raw,
  state-normalized, and copy-start-normalized latent error.
- `runs/more_msmw_manip_val_016_ms/dashboard.html` and
  `runs/more_msmw_manip_val_016_mw/dashboard.html`: rollout visuals.
- `runs/more_msmw_manip_val_016_summary/run_status.json` and
  `runs/all_msmw_manip_phase_val_016_summary/run_status.json`: completeness checks.

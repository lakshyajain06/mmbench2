# ManiSkill versus MetaWorld dataset-size check

Counts below come from the local `.pt` episode metadata under
`~/datasets/mmbench2_robotics/`. These files contain four evaluated ManiSkill
tasks and six evaluated MetaWorld tasks. This local install is a subset of the
checkpoint's 210-task dataset, so the counts establish local available data,
not the exact number of examples the released checkpoint saw during training.
The [dataset card](https://huggingface.co/datasets/nicklashansen/mmbench2/blob/main/README.md)
describes MetaWorld episodes as 100 steps and ManiSkill episodes as 25–500
steps across the full dataset; the selected local ManiSkill tasks are all at
the 25-step end of that range.

| Per task | ManiSkill | MetaWorld |
| --- | ---: | ---: |
| Observations per episode | 26 | 101 |
| Transitions per episode | 25 | 100 |
| Episodes in each of expert, mixed-small, val, test | 20 | 20 |
| Episodes in mixed-large | 200 | 200 |
| Transitions in expert + mixed-small + mixed-large | 6,000 | 24,000 |
| Transitions in val | 500 | 2,000 |
| Valid 24-transition starts in the three local training partitions | 480 | 18,480 |

The 24-transition counts were checked directly against episode boundaries and
finite action/reward rows. The released dynamics checkpoint records
`seq_len=24` and `task_weighting="targeted"`, but its training data paths are
not present in checkpoint metadata. Targeted sampling also means raw valid
window counts cannot be equated with effective optimization exposure.

For the targeted `ms-push-cube` and `mw-push` pair, each of the five installed
`active_*` collection partitions adds 50 episodes: 1,250 ManiSkill versus
5,000 MetaWorld transitions per partition. Across the three base partitions
and those five active partitions, the local totals are 12,250 versus 49,000
transitions. The [checkpoint card](https://huggingface.co/nicklashansen/mmbench2-models)
describes `combined` as the variant finetuned with all targeted collection
sources. These counts still describe available recordings, not how often each
example was sampled in optimization.

The evaluation config points to the local dataset root and to
`src/checkpoints/combined/{tokenizer,dynamics}.pt`; both checkpoints load with
strict state-dict matching, and both local files have the same SHA-256 as the
published `combined` files. Saved action slices match the source metadata in
all 1,760 validation windows checked. The evaluator uses the same packed
tokenizer and dynamics loading functions as `src/plan_cem.py`.
For file identity, SHA-256 checks matched the published Hugging Face LFS
checksums for all ten evaluated val `.pt` files and all ten corresponding val
PNG frame strips. The `ms-reach`, `mw-reach`, `ms-push-cube`, and `mw-push`
metadata files also matched in each of `expert`, `mixed-small`, and
`mixed-large` (12 more checks). Thus the observed val-return differences do
not come from a local replacement of those checked files.

At 16 transitions, the existing quick validation run samples ManiSkill
starts at frames **1–9** and MetaWorld starts at frames **15–81**. This follows
from sampling one start uniformly among valid starts in each episode: a
26-observation episode permits starts 0–9; a 101-observation episode permits
starts 0–84. It means the domain comparison mixes very different episode
phases. It does not isolate the effect of training dataset size.

The current latent-error results divided by each task's real-latent standard
deviation (pooled across the sampled expert and val endpoints) are:

| Partition / horizon | ManiSkill task mean | MetaWorld task mean |
| --- | ---: | ---: |
| Expert / 1 | 0.493 | 0.331 |
| Expert / 16 | 0.635 | 0.594 |
| Val / 1 | 0.594 | 0.498 |
| Val / 16 | 0.919 | 1.066 |

Each task contributes 40 windows per horizon (20 episodes × 2 seeds); the
table averages four ManiSkill and six MetaWorld task means. The shorter-horizon
results have the expected direction, but the 16-step validation results do
not. At 16 steps, the mean raw val latent RMS is 0.358 for ManiSkill and 0.382
for MetaWorld, while the mean task-state denominator is 0.391 versus 0.359;
both the raw errors and the scaling contribute to the reversed normalized
comparison. A controlled domain comparison needs a common episode-phase sampling
rule, or a fixed absolute start range, followed by fresh rollouts. Dataset
size alone does not establish which domain must have lower prediction error.

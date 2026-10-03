# Robotics dataset rounds: trajectory audit

This is an audit of the **recorded data**, not a world-model evaluation. It covers the eight first-pass tasks in `ROBOTICS_HALLUCINATION_EVAL_PLAN.md` under `~/datasets/mmbench2_robotics/`.

| Task | Expert | Mixed-small | Mixed-large | Val | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ms-reach` | 93.9 | 39.9 | 35.9 | 38.5 | 94.1 |
| `mw-reach` | 1867.2 | 1099.1 | 1412.2 | 851.7 | 1862.0 |
| `ms-push-cube` | 42.7 | 16.3 | 13.8 | 34.3 | 45.2 |
| `mw-push` | 1796.9 | 338.7 | 618.0 | 534.3 | 1795.8 |
| `mw-button-press` | 1663.5 | 323.9 | 639.9 | 339.6 | 1673.0 |
| `mw-drawer-open` | 1827.6 | 821.5 | 1167.3 | 952.7 | 1827.1 |
| `rd-open-drawer` | 752.1 | 291.3 | 354.3 | 338.0 | 734.7 |
| `ms-pick-cube` | 43.4 | 7.2 | 9.6 | 21.6 | 42.6 |

Values are **mean recorded episode returns**, computed as the sum of finite rewards per episode and then averaged within each task/partition. Reward scales differ across tasks, so compare a task across partitions, not one task with another. A high recorded return does not establish world-model competence or low hallucination frequency.

For every task, `expert`, `mixed-small`, `val`, and `test` each contain 20 episodes. `mixed-large` contains 200. Every ManiSkill episode has 26 observations; every Meta-World and RoboDesk episode has 101. No non-finite actions occur after the first observation in the eight tasks. The `episode`, `action`, and `reward` tensors are not exact copies between `expert` and `test`, `expert` and `val`, `mixed-small` and `val`, or `mixed-small` and `test` for any of the eight tasks. This metadata comparison does not prove that no frames or partial action sequences overlap.

The `test` returns resemble `expert` returns, while `val` and the mixed partitions generally have lower returns. This suggests using `expert` for the clean-trajectory check, both mixed partitions for harder behavior, `val` for task screening and calibration, and `test` for the held-out final check. Success labels were not inspected, and no checkpoint rollouts have been run because the NVIDIA driver is currently unavailable.

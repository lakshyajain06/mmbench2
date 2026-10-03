"""Descriptive coverage of selected tasks; never used for initial selection."""
import collections
from report import read_jsonl

METRICS = ('real_motion_rms', 'real_image_motion_rms', 'action_rms')


def describe(steps_path, tasks, partition='val'):
    buckets = collections.defaultdict(list)
    for row in read_jsonl(steps_path):
        if row['partition'] == partition and row['task'] in tasks:
            buckets[row['task']].append(row)
    summary = []
    for task, rows in sorted(buckets.items()):
        result = dict(task=task, partition=partition,
                      episodes=len({r['episode'] for r in rows}),
                      start_frames=len({(r['episode'], r['start']) for r in rows}))
        for metric in METRICS:
            if metric not in rows[0]:
                raise ValueError(f'missing {metric}; rerun rollouts with diversity logging')
            values = sorted(float(r[metric]) for r in rows)
            n = len(values)
            result[metric] = dict(min=values[0], median=values[n // 2],
                                  p90=values[int(.9 * (n - 1))], max=values[-1],
                                  mean=sum(values) / n)
        summary.append(result)
    return summary

"""Compare saved rollout errors with a copy-the-start-frame baseline."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path

from datasets import build_dataset

IDENTITY = ('partition', 'task', 'episode', 'start', 'horizon', 'seed')


def identity(row):
    return tuple(row[field] for field in IDENTITY)


def audit(run_dir):
    import torch

    manifest = json.loads((run_dir / 'manifest.json').read_text())
    dataset = build_dataset(manifest['config']['dataset'])
    terminal = {}
    with (run_dir / 'steps.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            if row['transition'] == row['horizon']:
                key = identity(row)
                if key in terminal:
                    raise ValueError(f'duplicate terminal step: {key}')
                terminal[key] = row

    by_episode = collections.defaultdict(list)
    with (run_dir / 'windows.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            by_episode[(row['partition'], row['task'], row['episode'])].append(row)

    buckets = collections.defaultdict(list)
    seen = set()
    for (partition, task, episode_id), windows in by_episode.items():
        frames = dataset.load_episode(task, partition, episode_id).frames
        for window in windows:
            key = identity(window)
            if key in seen or key not in terminal:
                raise ValueError(f'duplicate or incomplete window: {key}')
            seen.add(key)
            start, horizon = window['start'], window['horizon']
            copy_rms = float((frames[start] - frames[start + horizon]).square().mean().sqrt())
            score = terminal[key]
            buckets[(partition, task, horizon)].append((
                score['image_rms'], score['tokenizer_image_rms'], copy_rms, start))
    if len(seen) != len(terminal):
        raise ValueError('terminal steps do not match windows')

    summary = []
    for (partition, task, horizon), values in sorted(buckets.items()):
        model, tokenizer, copy, starts = zip(*values)
        model_mean = sum(model) / len(model)
        copy_mean = sum(copy) / len(copy)
        row = dict(partition=partition, task=task, horizon=horizon, windows=len(values),
                   model_image_rms=model_mean,
                   tokenizer_image_rms=sum(tokenizer) / len(tokenizer),
                   copy_start_image_rms=copy_mean,
                   model_to_copy_ratio=model_mean / copy_mean if copy_mean else math.inf,
                   mean_start=sum(starts) / len(starts),
                   min_start=min(starts), max_start=max(starts))
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args.run)
    output = args.run / 'audit_baselines.csv'
    with output.open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f'{output}: {len(summary)} task/horizon groups')


if __name__ == '__main__':
    main()

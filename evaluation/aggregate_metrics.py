"""Combine split evaluation runs into per-partition, task, and horizon metrics."""
import argparse
import collections
import csv
import json
import math
import shutil
from pathlib import Path

KEYS = ('partition', 'task', 'episode', 'start', 'horizon', 'seed')
METRICS = ('latent_rms', 'image_rms', 'tokenizer_image_rms',
           'u_r_norm', 'u_f_norm', 'u_s')


def identity(row):
    return tuple(row[k] for k in KEYS)


def aggregate(run_dirs):
    windows = {}
    step_counts = collections.Counter()
    buckets = collections.defaultdict(lambda: dict(steps=0, windows=0, episodes=set(),
                                                   starts=set(), sums=collections.Counter(),
                                                   terminal=collections.Counter()))
    for run_dir in map(Path, run_dirs):
        with (run_dir / 'windows.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                key = identity(row)
                if key in windows:
                    raise ValueError(f'duplicate window {key}')
                windows[key] = True
        with (run_dir / 'steps.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                key = identity(row)
                if key not in windows:
                    raise ValueError(f'step without window {key}')
                step_counts[key] += 1
                group = buckets[(row['partition'], row['task'], row['horizon'])]
                group['steps'] += 1
                group['episodes'].add(row['episode'])
                group['starts'].add((row['episode'], row['start']))
                for name in METRICS:
                    value = row[name]
                    if not math.isfinite(value):
                        raise ValueError(f'nonfinite {name} in {key}')
                    group['sums'][name] += value
                    if row['transition'] == row['horizon']:
                        group['terminal'][name] += value
    incomplete = [(key, step_counts[key]) for key in windows if step_counts[key] != key[4]]
    if incomplete:
        raise ValueError(f'{len(incomplete)} incomplete windows, first {incomplete[:3]}')
    for key in windows:
        buckets[key[:2] + (key[4],)]['windows'] += 1
    summary = []
    for (partition, task, horizon), group in sorted(buckets.items()):
        row = dict(partition=partition, task=task, horizon=horizon,
                   episodes=len(group['episodes']), starts=len(group['starts']),
                   windows=group['windows'], transitions=group['steps'])
        for name in METRICS:
            row[f'mean_{name}'] = group['sums'][name] / group['steps']
            row[f'terminal_{name}'] = group['terminal'][name] / group['windows']
        summary.append(row)
    return summary, dict(windows=len(windows), transitions=sum(step_counts.values()),
                         partitions=sorted({key[0] for key in windows}),
                         tasks=sorted({key[1] for key in windows}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-windows', type=int)
    args = parser.parse_args()
    summary, status = aggregate(args.runs)
    if args.expected_windows is not None and status['windows'] != args.expected_windows:
        raise SystemExit(f"expected {args.expected_windows} windows; got {status['windows']}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((Path(run) / 'manifest.json').read_text()) for run in args.runs]
    first_config = manifests[0]['config']
    for manifest in manifests[1:]:
        config = manifest['config']
        if config['dataset'] != first_config['dataset'] or config['model'] != first_config['model']:
            raise ValueError('runs use different dataset or model configurations')
    combined_manifest = dict(config=first_config, partitions=status['partitions'],
                             tasks=status['tasks'], pilot=False,
                             exploratory_test=False,
                             source_runs=[str(Path(run).resolve()) for run in args.runs])
    (args.output / 'manifest.json').write_text(json.dumps(combined_manifest, indent=2) + '\n')
    (args.output / 'metrics_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'run_status.json').write_text(json.dumps(status, indent=2) + '\n')
    for name in ('windows.jsonl', 'steps.jsonl'):
        with (args.output / name).open('w') as destination:
            for run_dir in map(Path, args.runs):
                with (run_dir / name).open() as source:
                    shutil.copyfileobj(source, destination)
    from report import make_review_sheet
    make_review_sheet(args.output / 'windows.jsonl', args.output / 'review.csv')
    with (args.output / 'metrics_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()

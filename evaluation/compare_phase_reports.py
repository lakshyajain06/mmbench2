"""Compare original and phase-matched normalized latent reports."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_summary(path, partition):
    with Path(path).open(newline='') as source:
        return {(row['task'], int(row['horizon'])): row
                for row in csv.DictReader(source) if row['partition'] == partition}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--matched', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partition', default='val')
    args = parser.parse_args()
    original = read_summary(args.original, args.partition)
    matched = read_summary(args.matched, args.partition)
    if original.keys() != matched.keys():
        raise ValueError('reports must contain the same task/horizon groups')
    rows = []
    for task, horizon in sorted(original):
        a, b = original[(task, horizon)], matched[(task, horizon)]
        if abs(float(a['task_state_latent_std']) -
               float(b['task_state_latent_std'])) > 1e-9:
            raise ValueError(f'{task}: state scales differ; pass --reference-scales to normalizer')
        rows.append(dict(task=task, domain='ManiSkill' if task.startswith('ms-') else 'MetaWorld',
                         horizon=horizon, original=float(a['error_over_task_state_std']),
                         matched=float(b['error_over_task_state_std']),
                         original_raw=float(a['model_latent_rms']),
                         matched_raw=float(b['model_latent_rms']),
                         original_copy_ratio=float(a['normalized_latent_error']),
                         matched_copy_ratio=float(b['normalized_latent_error']),
                         original_windows=int(a['windows']), matched_windows=int(b['windows'])))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'comparison.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['domain'], row['horizon'])].append(row)
    domain = [{
        'domain': key[0], 'horizon': key[1], 'tasks': len(items),
        'original': statistics.mean(row['original'] for row in items),
        'matched': statistics.mean(row['matched'] for row in items),
        'original_raw': statistics.mean(row['original_raw'] for row in items),
        'matched_raw': statistics.mean(row['matched_raw'] for row in items),
        'original_copy_ratio': statistics.mean(row['original_copy_ratio'] for row in items),
        'matched_copy_ratio': statistics.mean(row['matched_copy_ratio'] for row in items),
    } for key, items in sorted(grouped.items())]
    (args.output / 'domain_means.json').write_text(json.dumps(domain, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = {'ManiSkill': '#2563eb', 'MetaWorld': '#ea580c'}
    for domain_name in colors:
        items = sorted((row for row in domain if row['domain'] == domain_name),
                       key=lambda row: row['horizon'])
        for field, style, label in (('original', '--', 'original'), ('matched', '-', 'matched phase')):
            axes[0].plot([row['horizon'] for row in items], [row[field] for row in items],
                         linestyle=style, marker='o', color=colors[domain_name],
                         label=f'{domain_name} · {label}')
    axes[0].set(xlabel='Rollout horizon (transitions)', ylabel='Latent RMS / task-state SD',
                title='Domain mean by horizon', xticks=[1, 4, 8, 16])
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    longest = max(row['horizon'] for row in rows)
    task_rows = [row for row in rows if row['horizon'] == longest]
    for i, row in enumerate(task_rows):
        color = colors[row['domain']]
        axes[1].plot([row['original'], row['matched']], [i, i], color=color, alpha=.55)
        axes[1].scatter(row['original'], i, color=color, marker='x', s=55)
        axes[1].scatter(row['matched'], i, color=color, marker='o', s=38)
    axes[1].set(yticks=range(len(task_rows)), yticklabels=[row['task'] for row in task_rows],
                xlabel='Latent RMS / task-state SD',
                title=f'{longest}-step tasks: × original, ● matched phase')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=.2)
    fig.savefig(args.output / 'comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps(domain, indent=2))


if __name__ == '__main__':
    main()

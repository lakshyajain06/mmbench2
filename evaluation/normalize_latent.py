"""Normalize saved latent errors by real motion and task-state variation."""
import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path

from datasets import build_dataset

IDENTITY = ('partition', 'task', 'episode', 'start', 'horizon', 'seed')


def identity(row):
    return tuple(row[field] for field in IDENTITY)


def load_run(run_dir, prefixes):
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    windows = [json.loads(line) for line in (run_dir / 'windows.jsonl').open()]
    if prefixes:
        windows = [row for row in windows if row['task'].startswith(tuple(prefixes))]
    wanted = {identity(row) for row in windows}
    if len(wanted) != len(windows):
        raise ValueError(f'duplicate windows in {run_dir}')
    terminal = {}
    with (run_dir / 'steps.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            key = identity(row)
            if key in wanted and row['transition'] == row['horizon']:
                if key in terminal:
                    raise ValueError(f'duplicate terminal step: {key}')
                terminal[key] = row
    if len(terminal) != len(windows):
        raise ValueError(f'incomplete windows in {run_dir}')
    return manifest, windows, terminal


def normalize(run_dirs, prefixes):
    import torch

    src = Path(__file__).resolve().parents[1] / 'src'
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from interactive import load_tokenizer_from_ckpt
    from plan_cem import encode_frames_to_packed

    loaded = [load_run(Path(path), prefixes) for path in run_dirs]
    configs = [item[0]['config'] for item in loaded]
    first = configs[0]
    for config in configs[1:]:
        if config['dataset'] != first['dataset']:
            raise ValueError('runs must use the same dataset configuration')
        for field in ('tokenizer_ckpt', 'packing_factor'):
            if config['model'][field] != first['model'][field]:
                raise ValueError(f'runs differ in model.{field}')
    dataset = build_dataset(first['dataset'])
    device = torch.device(first['model']['device'])
    tokenizer, info = load_tokenizer_from_ckpt(first['model']['tokenizer_ckpt'], device)
    packing_factor = first['model']['packing_factor']
    n_spatial = info['n_latents'] // packing_factor
    use_amp = bool(first['model']['amp']) and device.type == 'cuda'

    by_episode = collections.defaultdict(list)
    terminals = {}
    for _manifest, windows, terminal in loaded:
        for window in windows:
            key = identity(window)
            if key in terminals:
                raise ValueError(f'duplicate window across runs: {key}')
            terminals[key] = terminal[key]
            by_episode[(window['partition'], window['task'], window['episode'])].append(window)

    rows = []
    state_sums = {}
    state_squares = {}
    state_counts = collections.Counter()
    for (partition, task, episode_id), windows in by_episode.items():
        episode = dataset.load_episode(task, partition, episode_id)
        indices = sorted({index for window in windows
                          for index in (window['start'], window['start'] + window['horizon'])})
        with torch.inference_mode():
            encoded = encode_frames_to_packed(
                tokenizer.encoder, episode.frames[indices].to(device),
                patch=info['patch'], n_spatial=n_spatial,
                packing_factor=packing_factor, use_amp=use_amp).float().cpu()
        if task not in state_sums:
            state_sums[task] = torch.zeros_like(encoded[0], dtype=torch.float64)
            state_squares[task] = torch.zeros_like(encoded[0], dtype=torch.float64)
        state_sums[task] += encoded.double().sum(dim=0)
        state_squares[task] += encoded.double().square().sum(dim=0)
        state_counts[task] += len(encoded)
        latents = dict(zip(indices, encoded))
        for window in windows:
            start, end = window['start'], window['start'] + window['horizon']
            copy_rms = float((latents[start] - latents[end]).square().mean().sqrt())
            model_rms = float(terminals[identity(window)]['latent_rms'])
            if not math.isfinite(copy_rms) or not math.isfinite(model_rms):
                raise ValueError(f'non-finite latent error: {identity(window)}')
            rows.append(dict(**{field: window[field] for field in IDENTITY},
                             model_latent_rms=model_rms,
                             copy_start_latent_rms=copy_rms,
                             model_beats_copy=model_rms < copy_rms))
    scales = {}
    for task in state_sums:
        mean = state_sums[task] / state_counts[task]
        variance = (state_squares[task] / state_counts[task] - mean.square()).clamp_min(0)
        scales[task] = float(variance.mean().sqrt())
    return rows, scales


def summarize(rows, state_scales=None):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row['partition'], row['task'], row['horizon'])].append(row)
    summary = []
    for (partition, task, horizon), items in sorted(groups.items()):
        model = sum(item['model_latent_rms'] for item in items) / len(items)
        copy = sum(item['copy_start_latent_rms'] for item in items) / len(items)
        row = dict(partition=partition, task=task, horizon=horizon,
                   windows=len(items), model_latent_rms=model,
                   copy_start_latent_rms=copy,
                   normalized_latent_error=model / copy if copy else math.inf,
                   fraction_model_beats_copy=sum(item['model_beats_copy'] for item in items) / len(items))
        if state_scales is not None:
            scale = state_scales[task]
            row['task_state_latent_std'] = scale
            row['error_over_task_state_std'] = model / scale if scale else math.inf
        summary.append(row)
    return summary


def read_reference_scales(path):
    """Read a prior normalization report's fixed per-task state scales."""
    scales = {}
    with Path(path).open(newline='') as source:
        for row in csv.DictReader(source):
            task = row['task']
            value = float(row['task_state_latent_std'])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'invalid reference state scale for {task}: {value}')
            if task in scales and not math.isclose(scales[task], value, rel_tol=1e-9):
                raise ValueError(f'inconsistent reference state scale for {task}')
            scales[task] = value
    if not scales:
        raise ValueError('reference scale file is empty')
    return scales


def write_figure(summary, output, metric, title, xlabel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    horizon = max(row['horizon'] for row in summary)
    partitions = sorted({row['partition'] for row in summary})
    tasks = sorted({row['task'] for row in summary})
    fig, axes = plt.subplots(1, len(partitions), figsize=(6.5 * len(partitions),
                                                        max(5, 0.48 * len(tasks) + 2)),
                             sharex=True, sharey=True, constrained_layout=True)
    if len(partitions) == 1:
        axes = [axes]
    for ax, partition in zip(axes, partitions):
        lookup = {row['task']: row for row in summary
                  if row['partition'] == partition and row['horizon'] == horizon}
        values = [lookup[task][metric] for task in tasks]
        y = np.arange(len(tasks))
        ax.barh(y, values, color=['#2563eb' if task.startswith('ms-') else '#ea580c'
                                   for task in tasks])
        if metric == 'normalized_latent_error':
            ax.axvline(1, color='#475569', linestyle='--', linewidth=1.3)
        ax.set(yticks=y, yticklabels=tasks, xlabel=xlabel,
               title=f'{partition} · {horizon} transitions')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
    fig.suptitle(title, fontsize=15, fontweight='bold')
    fig.savefig(output, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', action='append', default=[])
    parser.add_argument('--reference-scales', type=Path,
                        help='prior summary.csv whose task-state scales stay fixed')
    args = parser.parse_args()
    rows, scales = normalize(args.runs, args.prefix)
    if args.reference_scales is not None:
        scales = read_reference_scales(args.reference_scales)
        missing = sorted({row['task'] for row in rows} - scales.keys())
        if missing:
            raise ValueError(f'missing reference scales for tasks: {missing}')
    summary = summarize(rows, scales)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'windows.jsonl').open('w') as target:
        for row in rows:
            target.write(json.dumps(row) + '\n')
    with (args.output / 'summary.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_figure(summary, args.output / 'summary.png',
                 'normalized_latent_error',
                 'Latent prediction error / real endpoint change',
                 'Model RMS / copy-start RMS')
    write_figure(summary, args.output / 'state_normalized.png',
                 'error_over_task_state_std',
                 'Latent prediction error / pooled real-state variation',
                 'Model RMS / task latent standard deviation')
    print(f'{len(rows)} windows; {len(summary)} groups; {args.output}')


if __name__ == '__main__':
    main()

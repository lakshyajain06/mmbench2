"""Human event labels, screening and held-out summaries."""
import collections
import csv
import json
from pathlib import Path

KEYS = ('partition', 'task', 'episode', 'start', 'horizon', 'seed')
EVENTS = {'unsupported_object_motion', 'detachment', 'opening_before_contact',
          'activation_before_contact', 'other_physical_error'}


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def key(row):
    return tuple(row[k] for k in KEYS)


def make_review_sheet(windows_path, output):
    windows = read_jsonl(windows_path)
    output = Path(output)
    fields = [*KEYS, 'useful_plausible', 'event', 'first_error_transition',
              'real_action_support', 'notes']
    prior = {}
    if output.exists():
        with output.open(newline='') as f:
            for row in csv.DictReader(f):
                for field in ('episode', 'start', 'horizon', 'seed'):
                    row[field] = int(row[field])
                prior[key(row)] = row
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for window in windows:
            identity = key(window)
            row = dict(prior.get(identity, {}))
            row.update({k: window[k] for k in KEYS})
            writer.writerow({field: row.get(field, '') for field in fields})
    return len(windows)


def read_labels(path):
    labels = {}
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            for k in ('episode', 'start', 'horizon', 'seed'):
                row[k] = int(row[k])
            identity = key(row)
            if identity in labels:
                raise ValueError(f'duplicate label: {identity}')
            useful = row['useful_plausible'].strip().lower()
            if useful not in ('yes', 'no'):
                continue
            event = row['event'].strip()
            if event and event not in EVENTS:
                raise ValueError(f'unknown physical event {event}')
            supported = row['real_action_support'].strip().lower()
            if event and useful == 'yes':
                raise ValueError(f'{identity}: physical error cannot be useful/plausible')
            if event and supported != 'yes':
                raise ValueError(f'{identity}: event requires real_action_support=yes')
            onset = row['first_error_transition'].strip()
            if event and (not onset or not 1 <= int(onset) <= row['horizon']):
                raise ValueError(f'{identity}: first error must be within horizon')
            if not event and onset:
                raise ValueError(f'{identity}: onset without event')
            row['useful'] = useful == 'yes'
            row['hallucination'] = bool(event)
            row['onset'] = int(onset) if onset else None
            labels[identity] = row
    return labels


def summarize(steps_path, labels_path, review_config, partition=None):
    steps = read_jsonl(steps_path)
    labels = read_labels(labels_path)
    grouped = collections.defaultdict(list)
    for step in steps:
        if partition and step['partition'] != partition:
            continue
        grouped[key(step)].append(step)
    rows = []
    buckets = collections.defaultdict(list)
    for identity, samples in grouped.items():
        samples.sort(key=lambda x: x['transition'])
        if len(samples) != identity[4]:
            raise ValueError(f'incomplete rollout {identity}')
        label = labels.get(identity)
        terminal = samples[-1]
        row = {k: terminal[k] for k in KEYS}
        row.update(latent_rms=sum(s['latent_rms'] for s in samples) / len(samples),
                   image_rms=sum(s['image_rms'] for s in samples) / len(samples),
                   tokenizer_image_rms=sum(s['tokenizer_image_rms'] for s in samples) / len(samples),
                   useful=label['useful'] if label else None,
                   hallucination=label['hallucination'] if label else None,
                   first_error_frame=(identity[3] + label['onset']) if label and label['onset'] else None,
                   event=label['event'] if label else None)
        rows.append(row)
        buckets[(row['partition'], row['task'], row['horizon'])].append(row)
    summary = []
    for (part, task, horizon), group in sorted(buckets.items()):
        reviewed = [r for r in group if r['useful'] is not None]
        affected = [r for r in reviewed if r['hallucination']]
        episodes = {r['episode'] for r in affected}
        summary.append(dict(partition=part, task=task, horizon=horizon,
                            windows=len(group), reviewed=len(reviewed),
                            latent_rms=sum(r['latent_rms'] for r in group) / len(group),
                            image_rms=sum(r['image_rms'] for r in group) / len(group),
                            tokenizer_image_rms=sum(r['tokenizer_image_rms'] for r in group) / len(group),
                            useful_rate=sum(r['useful'] for r in reviewed) / len(reviewed) if reviewed else None,
                            hallucination_rate=len(affected) / len(reviewed) if reviewed else None,
                            affected_episodes=len(episodes),
                            qualifies=(len(reviewed) == len(group) and
                                       sum(r['useful'] for r in reviewed) / len(reviewed) >= review_config['min_useful_rate'] and
                                       review_config['min_hallucination_rate'] <= len(affected) / len(reviewed) <= review_config['max_hallucination_rate'] and
                                       len(episodes) >= review_config['min_affected_episodes'])))
    task_buckets = collections.defaultdict(list)
    for row in rows:
        task_buckets[(row['partition'], row['task'])].append(row)
    for (part, task), group in sorted(task_buckets.items()):
        reviewed = [r for r in group if r['useful'] is not None]
        affected = [r for r in reviewed if r['hallucination']]
        episodes = {r['episode'] for r in affected}
        complete = len(reviewed) == len(group)
        useful_rate = sum(r['useful'] for r in reviewed) / len(reviewed) if reviewed else None
        hallucination_rate = len(affected) / len(reviewed) if reviewed else None
        summary.append(dict(partition=part, task=task, horizon='all',
                            windows=len(group), reviewed=len(reviewed),
                            latent_rms=sum(r['latent_rms'] for r in group) / len(group),
                            image_rms=sum(r['image_rms'] for r in group) / len(group),
                            tokenizer_image_rms=sum(r['tokenizer_image_rms'] for r in group) / len(group),
                            useful_rate=useful_rate, hallucination_rate=hallucination_rate,
                            affected_episodes=len(episodes),
                            qualifies=(complete and useful_rate >= review_config['min_useful_rate']
                                       and review_config['min_hallucination_rate'] <= hallucination_rate
                                       <= review_config['max_hallucination_rate']
                                       and len(episodes) >= review_config['min_affected_episodes'])))
    return rows, summary


def select_tasks(summary, partition='val'):
    return sorted(row['task'] for row in summary
                  if row['partition'] == partition and row['horizon'] == 'all'
                  and row['qualifies'])


def episode_summary(windows):
    buckets = collections.defaultdict(list)
    for row in windows:
        buckets[(row['partition'], row['task'], row['horizon'], row['episode'])].append(row)
    result = []
    for (partition, task, horizon, episode), group in sorted(buckets.items()):
        reviewed = [r for r in group if r['useful'] is not None]
        result.append(dict(partition=partition, task=task, horizon=horizon, episode=episode,
                           windows=len(group), reviewed=len(reviewed),
                           useful_rate=sum(r['useful'] for r in reviewed) / len(reviewed) if reviewed else None,
                           hallucination_rate=sum(r['hallucination'] for r in reviewed) / len(reviewed) if reviewed else None,
                           first_error_frames=sorted({r['first_error_frame'] for r in reviewed
                                                      if r['first_error_frame'] is not None}),
                           latent_rms=sum(r['latent_rms'] for r in group) / len(group),
                           image_rms=sum(r['image_rms'] for r in group) / len(group),
                           tokenizer_image_rms=sum(r['tokenizer_image_rms'] for r in group) / len(group)))
    return result

"""Recorded-action evaluation with explicit seed and window identity."""
import json
from pathlib import Path

from config import sample_episode_ids, sample_horizon_starts, stable_seed
from datasets import build_dataset
from models import build_model


def rms(a, b):
    return float((a.float() - b.float()).square().mean().sqrt().item())


def run(config, partitions=None, tasks=None, pilot=False, exploratory_test=False):
    import torch
    dataset = build_dataset(config['dataset'])
    model = build_model(config['model'], config['tasks'])
    sampling = config['sampling']
    partitions = partitions or config['dataset']['partitions']
    tasks = tasks or config['tasks']['first_pass']
    output = Path(config['output']) / 'pilot' if pilot else Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'config': config, 'partitions': partitions, 'tasks': tasks,
                'pilot': pilot, 'exploratory_test': exploratory_test}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    result_path = output / 'steps.jsonl'
    windows_path = output / 'windows.jsonl'
    mode = 'w' if pilot else 'a'
    existing = set()
    if not pilot and windows_path.exists():
        existing = recover_complete_windows(windows_path, result_path)
    with result_path.open(mode) as steps_file, windows_path.open(mode) as windows_file:
        for task in tasks:
            for partition in partitions:
                ids = sample_episode_ids(dataset.episode_ids(task, partition),
                                         1 if pilot else sampling['episodes_per_task'],
                                         sampling['seed'])
                for episode_id in ids:
                    episode = dataset.load_episode(task, partition, episode_id)
                    seed_scope = sampling.get('random_seed_scope', 'episode')
                    if seed_scope == 'episode':
                        start_seed = sampling['seed'] + episode_id
                    elif seed_scope == 'task_episode':
                        start_seed = stable_seed(sampling['seed'], partition, task,
                                                 episode_id)
                    else:
                        raise ValueError("random_seed_scope must be 'episode' or 'task_episode'")
                    starts_by_horizon = sample_horizon_starts(
                        len(episode.frames), sampling['horizons'],
                        1 if pilot else sampling['windows_per_episode'],
                        start_seed,
                        paired=sampling.get('paired_horizons', False),
                        phase_fraction=sampling.get('phase_fraction'),
                        start_mode=sampling.get('start_mode'))
                    for horizon in sampling['horizons']:
                        starts = starts_by_horizon[horizon]
                        entries = []
                        with torch.inference_mode():
                            for start in starts:
                                if all((partition, task, episode_id, start, horizon, seed) in existing
                                       for seed in sampling['seeds']):
                                    continue
                                real = episode.frames[start:start + horizon + 1].to(model.device)
                                encoded = model.encode(real)
                                baseline = model.decode(encoded[1:])
                                action_slice = episode.actions[start + 1:start + horizon + 1]
                                entries.append((start, real, encoded, baseline, action_slice))
                            if not entries:
                                continue
                            if hasattr(model, 'rollout_windows'):
                                all_predictions, all_instabilities = model.rollout_windows(
                                    torch.stack([entry[2][0] for entry in entries]),
                                    torch.stack([entry[4] for entry in entries]),
                                    task, sampling['seeds'])
                            else:
                                all_predictions = all_instabilities = None
                            for index, (start, real, encoded, baseline, action_slice) in enumerate(entries):
                                samples = []
                                if all_predictions is not None:
                                    outputs = zip(sampling['seeds'], all_predictions[index],
                                                  all_instabilities[index])
                                elif hasattr(model, 'rollout_many'):
                                    predictions, instabilities = model.rollout_many(
                                        encoded[0], action_slice, task, sampling['seeds'])
                                    outputs = zip(sampling['seeds'], predictions, instabilities)
                                else:
                                    outputs = ((seed, *model.rollout(encoded[0], action_slice, task, seed))
                                               for seed in sampling['seeds'])
                                for seed, predicted, u_f in outputs:
                                    decoded = model.decode(predicted)
                                    reencoded = model.encode(decoded)
                                    samples.append((seed, predicted, decoded, reencoded, u_f))
                                stack = torch.stack([s[1] for s in samples])
                                u_s = stack.float().var(dim=0, unbiased=False).mean(dim=(1, 2))
                                for seed, predicted, decoded, reencoded, u_f in samples:
                                    identity = (partition, task, episode_id, start, horizon, seed)
                                    if identity in existing:
                                        continue
                                    existing.add(identity)
                                    key = dict(partition=partition, task=task,
                                               episode=episode_id, start=start,
                                               horizon=horizon, seed=seed)
                                    window = dict(key,
                                                  actions=action_slice.float().tolist(),
                                                  rewards=[float(x) if torch.isfinite(x) else None
                                                           for x in episode.rewards[start + 1:start + horizon + 1]])
                                    windows_file.write(json.dumps(window, allow_nan=False) + '\n')
                                    for step in range(horizon):
                                        previous = encoded[0] if step == 0 else predicted[step - 1]
                                        motion = rms(predicted[step], previous)
                                        u_r = rms(predicted[step], reencoded[step])
                                        real_motion = rms(encoded[step + 1], encoded[step])
                                        real_image_motion = rms(real[step + 1], real[step])
                                        action_rms = float(action_slice[step].float().square().mean().sqrt().item())
                                        row = dict(**key, transition=step + 1,
                                                   frame=start + step + 1,
                                                   latent_rms=rms(predicted[step], encoded[step + 1]),
                                                   image_rms=rms(decoded[step], real[step + 1]),
                                                   tokenizer_image_rms=rms(baseline[step], real[step + 1]),
                                                   motion_rms=motion, real_motion_rms=real_motion,
                                                   real_image_motion_rms=real_image_motion,
                                                   action_rms=action_rms, u_r=u_r,
                                                   u_r_norm=u_r / max(motion, 1e-3),
                                                   u_f=float(u_f[step]),
                                                   u_f_norm=float(u_f[step]) / max(motion, 1e-3),
                                                   u_s=float(u_s[step].item()))
                                        steps_file.write(json.dumps(row) + '\n')
                                    if config.get('save_visuals', True):
                                        save_visual(output, key, real, decoded, baseline)
                                    steps_file.flush()
                                    windows_file.flush()
                    print(f'{partition}/{task}: episode {episode_id} complete', flush=True)
    return result_path


def save_visual(output, identity, real, predicted, baseline):
    from PIL import Image, ImageDraw
    import numpy as np
    task = identity['task']
    directory = output / 'visuals' / identity['partition'] / task
    directory.mkdir(parents=True, exist_ok=True)
    name = (f"ep{identity['episode']}_s{identity['start']}_"
            f"h{identity['horizon']}_seed{identity['seed']}.png")
    sequences = [real, predicted, baseline]
    n = identity['horizon']
    height, width = real.shape[-2:]
    row_height = height + 20
    canvas = Image.new('RGB', ((n + 1) * width, 3 * row_height), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (label, frames) in enumerate(zip(('real', 'predicted', 'tokenizer'), sequences)):
        draw.text((2, row * row_height), label, fill='black')
        for i, frame in enumerate(frames):
            rgb = (frame.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            column = i if row == 0 else i + 1
            canvas.paste(Image.fromarray(rgb), (column * width, row * row_height + 20))
    canvas.save(directory / name)


def recover_complete_windows(windows_path, steps_path):
    """Drop only interrupted windows so a long run can safely resume."""
    from collections import Counter
    from report import key as review_key, read_jsonl
    windows = read_jsonl(windows_path)
    steps = read_jsonl(steps_path)
    counts = Counter(review_key(row) for row in steps)
    complete = {review_key(row) for row in windows
                if counts[review_key(row)] == row['horizon']}
    if len(complete) != len(windows) or len(steps) != sum(counts[k] for k in complete):
        for path, rows in ((windows_path, windows), (steps_path, steps)):
            tmp = path.with_suffix(path.suffix + '.recovered')
            with tmp.open('w') as f:
                for row in rows:
                    if review_key(row) in complete:
                        f.write(json.dumps(row, allow_nan=False) + '\n')
            tmp.replace(path)
    return complete

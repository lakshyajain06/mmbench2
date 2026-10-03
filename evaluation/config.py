"""JSON configuration and deterministic experiment sampling."""
import copy
import hashlib
import json
import random
from pathlib import Path


def read_config(path, overrides=()):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    for override in overrides:
        key, separator, value = override.partition('=')
        if not separator:
            raise ValueError(f'override must be KEY=JSON_VALUE: {override}')
        target = config
        parts = key.split('.')
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(key)
        target[parts[-1]] = json.loads(value)
    base = path.parent
    for section, keys in {'dataset': ['root', 'shards_root'], 'tasks': ['metadata'],
                          'model': ['tokenizer_ckpt', 'dynamics_ckpt']}.items():
        for key in keys:
            if key in config[section] and config[section][key]:
                p = Path(config[section][key]).expanduser()
                config[section][key] = str((base / p).resolve() if not p.is_absolute() else p)
    output = Path(config['output']).expanduser()
    config['output'] = str((base / output).resolve() if not output.is_absolute() else output)
    return config


def sample_episode_ids(ids, limit, seed):
    ids = sorted(ids)
    if limit is None or limit >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, limit))


def stable_seed(seed, *parts):
    """Derive a reproducible integer seed without Python's randomized hash()."""
    payload = '\0'.join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


def sample_starts(length, horizon, limit, seed):
    eligible = list(range(length - horizon))
    if not eligible:
        return []
    if limit is None or limit >= len(eligible):
        return eligible
    return sorted(random.Random(seed).sample(eligible, limit))


def sample_horizon_starts(length, horizons, limit, seed, paired=False,
                          phase_fraction=None, start_mode=None):
    """Choose valid starts per horizon using explicit random or fixed-phase sampling."""
    if start_mode is None:
        # Preserve old configs: a supplied phase means fixed phase; otherwise random.
        start_mode = 'fixed_phase' if phase_fraction is not None else 'random'
    if start_mode not in ('random', 'fixed_phase'):
        raise ValueError("start_mode must be 'random' or 'fixed_phase'")
    if start_mode == 'random' and phase_fraction is not None:
        raise ValueError('random start_mode cannot set phase_fraction')
    if start_mode == 'fixed_phase':
        if phase_fraction is None:
            raise ValueError('fixed_phase start_mode requires phase_fraction')
        if not paired:
            raise ValueError('phase_fraction requires paired_horizons=true')
        if limit != 1:
            raise ValueError('phase_fraction currently requires windows_per_episode=1')
        if not 0 <= phase_fraction <= 1:
            raise ValueError('phase_fraction must be in [0, 1]')
        start = round(phase_fraction * (length - 1))
        return {horizon: [start] if start + horizon < length else []
                for horizon in horizons}
    if not paired:
        return {horizon: sample_starts(length, horizon, limit, seed + horizon)
                for horizon in horizons}
    eligible_horizons = [horizon for horizon in horizons if horizon < length]
    common = (sample_starts(length, max(eligible_horizons), limit, seed)
              if eligible_horizons else [])
    return {horizon: common if horizon in eligible_horizons else []
            for horizon in horizons}

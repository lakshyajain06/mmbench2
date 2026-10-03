"""Recorded robotics episodes; observation k is preceded by action[k]."""
import bisect
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Episode:
    episode_id: int
    frames: object
    actions: object
    rewards: object


class RoboticsDataset:
    def __init__(self, config):
        self.root = Path(config['root'])
        self.format = config['frames']
        self.shards_root = Path(config.get('shards_root', ''))
        self.strip_size = int(config.get('strip_size', 4008))
        self.strip_cache_size = int(config.get('strip_cache_size', 2))
        self._strip_cache = OrderedDict()
        if self.format not in ('strips', 'shards'):
            raise ValueError('dataset.frames must be strips or shards')

    def metadata(self, task, partition):
        import torch
        path = self.root / partition / f'{task}.pt'
        if not path.exists():
            raise FileNotFoundError(path)
        td = torch.load(path, map_location='cpu', weights_only=False)
        for field in ('episode', 'action', 'reward'):
            if field not in td:
                raise KeyError(f'{path}: missing {field}')
        return td

    def episode_ids(self, task, partition):
        return [int(x) for x in self.metadata(task, partition)['episode'].unique().tolist()]

    def load_episode(self, task, partition, episode_id):
        import torch
        td = self.metadata(task, partition)
        indices = (td['episode'].to(torch.int64) == episode_id).nonzero().flatten().tolist()
        if not indices:
            raise ValueError(f'{partition}/{task}: episode {episode_id} missing')
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f'{partition}/{task}: episode {episode_id} is not contiguous')
        frames = self._frames(task, partition, indices)
        actions = td['action'][indices].float()
        rewards = td['reward'][indices].float()
        if frames.shape[0] != len(indices):
            raise ValueError('frame/action alignment failed')
        if not torch.isnan(actions[0]).all() or not torch.isfinite(actions[1:]).all():
            raise ValueError('expected NaN action[0] and finite transition actions')
        return Episode(episode_id, frames, actions, rewards)

    def _frames(self, task, partition, indices):
        import torch
        if self.format == 'strips':
            from PIL import Image
            # Trusted dataset strips can be 4,008 frames wide; Pillow's generic
            # decompression-bomb limit is below their expected pixel count.
            Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS or 0, 224 * 224 * self.strip_size)
            root = self.root / partition
            result = []
            for index in indices:
                strip_num, offset = divmod(index, self.strip_size)
                cache_key = (partition, task, strip_num)
                if cache_key not in self._strip_cache:
                    path = root / f'{task}-{strip_num}.png'
                    self._strip_cache[cache_key] = Image.open(path).convert('RGB')
                    while len(self._strip_cache) > self.strip_cache_size:
                        self._strip_cache.popitem(last=False)
                self._strip_cache.move_to_end(cache_key)
                strip = self._strip_cache[cache_key]
                if strip.height != 224 or (offset + 1) * 224 > strip.width:
                    raise ValueError(f'bad strip alignment for {task}: index {index}')
                crop = strip.crop((offset * 224, 0, (offset + 1) * 224, 224))
                result.append(torch.from_numpy(__import__('numpy').array(crop).copy()).permute(2, 0, 1))
            return torch.stack(result).float() / 255.0
        root = self.shards_root / partition / task
        index_path = root / f'{task}_index.json'
        shard_meta = json.loads(index_path.read_text())
        names = sorted(shard_meta)
        ends = []
        for name in names:
            ends.append((ends[-1] if ends else 0) + int(shard_meta[name]))
        loaded = {}
        result = []
        for index in indices:
            shard = bisect.bisect_right(ends, index)
            if shard >= len(names):
                raise IndexError(f'frame {index} exceeds shard index')
            if shard not in loaded:
                loaded[shard] = torch.load(root / names[shard], map_location='cpu', weights_only=False)['frames']
            start = ends[shard - 1] if shard else 0
            result.append(loaded[shard][index - start])
        frames = torch.stack(result)
        if frames.shape[-1] == 3 and frames.shape[1] != 3:
            frames = frames.permute(0, 3, 1, 2)
        return frames.float() / 255.0


ADAPTERS = {'robotics': RoboticsDataset}

def build_dataset(config):
    name = config.get('adapter', 'robotics')
    if name not in ADAPTERS:
        raise ValueError(f'unknown dataset adapter {name}; register it in datasets.ADAPTERS')
    return ADAPTERS[name](config)

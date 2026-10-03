"""Model adapter contract: encode, decode, rollout."""
import json
import sys
from pathlib import Path


class CombinedModel:
    def __init__(self, config, tasks_config):
        import torch
        src = Path(__file__).resolve().parents[1] / 'src'
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from plan_cem import build_models
        from types import SimpleNamespace
        self.torch = torch
        self.config = config
        self.device = torch.device(config['device'])
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; override model.device="cpu" for a CPU pilot')
        args = SimpleNamespace(**config)
        (self.encoder, self.decoder, self.dyn, _, _, self.schedule,
         self.info) = build_models(args, self.device)
        self.task_meta = json.loads(Path(tasks_config['metadata']).read_text())
        self.use_amp = bool(config['amp']) and self.device.type == 'cuda'

    def encode(self, frames):
        from plan_cem import encode_frames_to_packed
        i = self.info
        return encode_frames_to_packed(self.encoder, frames.to(self.device),
                                       patch=i['patch'], n_spatial=i['n_spatial'],
                                       packing_factor=i['packing_factor'], use_amp=self.use_amp)

    def decode(self, latents):
        from plan_cem import decode_packed_sequence
        i = self.info
        return decode_packed_sequence(self.decoder, latents,
                                      H=i['H'], W=i['W'], C=i['C'], patch=i['patch'],
                                      packing_factor=i['packing_factor'],
                                      d_bottleneck=i['d_bottleneck'], use_amp=self.use_amp)

    def rollout(self, start, actions, task, seed):
        from interactive import sample_one_timestep_packed
        torch = self.torch
        meta = self.task_meta.get(task, {})
        act_dim = int(meta.get('action_dim', 16))
        if not 1 <= act_dim <= 16 or actions.shape[1] < act_dim:
            raise ValueError(f'{task}: invalid action dimension {act_dim}')
        if not torch.isfinite(actions[:, :act_dim]).all():
            raise ValueError('non-finite transition action')
        plan = torch.zeros((1, len(actions) + 1, 16), device=self.device)
        plan[0, 1:, :act_dim] = actions[:, :act_dim].to(self.device).clamp(-1, 1)
        mask = torch.zeros(16, device=self.device)
        mask[:act_dim] = 1
        lang = None
        if self.info['lang_dim']:
            embedding = meta.get('text_embedding')
            if embedding is None or len(embedding) != self.info['lang_dim']:
                raise ValueError(f'{task}: missing matching task text embedding')
            lang = torch.tensor(embedding, device=self.device, dtype=torch.float32).view(1, -1)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if self.device.type == 'cuda' else []):
            torch.manual_seed(int(seed))
            past = start.unsqueeze(0).unsqueeze(1)
            predicted, instability = [], []
            for t in range(len(actions)):
                next_z, u_f = sample_one_timestep_packed(
                    self.dyn, past_packed=past, k_max=self.info['k_max'],
                    sched=self.schedule, actions=plan[:, :t + 2], act_mask=mask,
                    use_amp=self.use_amp, tau_ctx=0.0, lang_emb=lang,
                    use_kv_cache=bool(self.config['use_kv_cache']))
                predicted.append(next_z[0].float())
                instability.append(float(u_f))
                past = torch.cat([past, next_z.unsqueeze(1)], dim=1)
            return torch.stack(predicted), instability

    def rollout_many(self, start, actions, task, seeds):
        """Batch independent seeded samples while preserving each seed's noise stream."""
        from interactive import sample_one_timestep_packed
        torch = self.torch
        meta = self.task_meta.get(task, {})
        act_dim = int(meta.get('action_dim', 16))
        if not seeds or not 1 <= act_dim <= 16 or actions.shape[1] < act_dim:
            raise ValueError(f'{task}: invalid seeds or action dimension')
        if not torch.isfinite(actions[:, :act_dim]).all():
            raise ValueError('non-finite transition action')
        batch = len(seeds)
        plan = torch.zeros((batch, len(actions) + 1, 16), device=self.device)
        plan[:, 1:, :act_dim] = actions[:, :act_dim].to(self.device).clamp(-1, 1)
        mask = torch.zeros(16, device=self.device)
        mask[:act_dim] = 1
        lang = None
        if self.info['lang_dim']:
            embedding = meta.get('text_embedding')
            if embedding is None or len(embedding) != self.info['lang_dim']:
                raise ValueError(f'{task}: missing matching task text embedding')
            lang = torch.tensor(embedding, device=self.device, dtype=torch.float32).view(1, -1).expand(batch, -1)
        generators = [torch.Generator(device=self.device).manual_seed(int(seed)) for seed in seeds]
        past = start.unsqueeze(0).unsqueeze(1).expand(batch, -1, -1, -1).contiguous()
        predicted, instability = [], []
        for t in range(len(actions)):
            noise = torch.cat([torch.randn((1, 1, *start.shape), generator=generator,
                                            device=self.device, dtype=start.dtype)
                               for generator in generators], dim=0)
            next_z, u_f = sample_one_timestep_packed(
                self.dyn, past_packed=past, k_max=self.info['k_max'],
                sched=self.schedule, actions=plan[:, :t + 2], act_mask=mask,
                use_amp=self.use_amp, tau_ctx=0.0, lang_emb=lang,
                use_kv_cache=bool(self.config['use_kv_cache']),
                initial_noise=noise, per_sample_instability=True)
            predicted.append(next_z.float())
            instability.append(u_f.float())
            past = torch.cat([past, next_z.unsqueeze(1)], dim=1)
        return torch.stack(predicted, dim=1), torch.stack(instability, dim=1)

    def rollout_windows(self, starts, action_windows, task, seeds):
        """Evaluate W recorded windows and N independent seeds in one model batch."""
        from interactive import sample_one_timestep_packed
        torch = self.torch
        windows, horizon = action_windows.shape[:2]
        n_seeds = len(seeds)
        if windows < 1 or n_seeds < 1 or starts.shape[0] != windows:
            raise ValueError('invalid window/seed batch')
        meta = self.task_meta.get(task, {})
        act_dim = int(meta.get('action_dim', 16))
        if not 1 <= act_dim <= 16 or action_windows.shape[2] < act_dim:
            raise ValueError(f'{task}: invalid action dimension')
        if not torch.isfinite(action_windows[:, :, :act_dim]).all():
            raise ValueError('non-finite transition action')
        batch = windows * n_seeds
        plan = torch.zeros((batch, horizon + 1, 16), device=self.device)
        actions = action_windows.to(self.device)[:, None, :, :act_dim]
        plan[:, 1:, :act_dim] = actions.expand(windows, n_seeds, horizon, act_dim).reshape(batch, horizon, act_dim).clamp(-1, 1)
        mask = torch.zeros(16, device=self.device)
        mask[:act_dim] = 1
        lang = None
        if self.info['lang_dim']:
            embedding = meta.get('text_embedding')
            if embedding is None or len(embedding) != self.info['lang_dim']:
                raise ValueError(f'{task}: missing matching task text embedding')
            lang = torch.tensor(embedding, device=self.device, dtype=torch.float32).view(1, -1).expand(batch, -1)
        generators = [torch.Generator(device=self.device).manual_seed(int(seed)) for seed in seeds]
        start_batch = starts.to(self.device)[:, None].expand(windows, n_seeds, *starts.shape[1:])
        past = start_batch.reshape(batch, *starts.shape[1:]).unsqueeze(1).contiguous()
        predicted, instability = [], []
        for t in range(horizon):
            noise_base = torch.cat([torch.randn((1, 1, *starts.shape[1:]), generator=generator,
                                                device=self.device, dtype=starts.dtype)
                                    for generator in generators], dim=0)
            noise = noise_base.repeat(windows, 1, 1, 1)
            next_z, u_f = sample_one_timestep_packed(
                self.dyn, past_packed=past, k_max=self.info['k_max'],
                sched=self.schedule, actions=plan[:, :t + 2], act_mask=mask,
                use_amp=self.use_amp, tau_ctx=0.0, lang_emb=lang,
                use_kv_cache=bool(self.config['use_kv_cache']),
                initial_noise=noise, per_sample_instability=True)
            predicted.append(next_z.float())
            instability.append(u_f.float())
            past = torch.cat([past, next_z.unsqueeze(1)], dim=1)
        latents = torch.stack(predicted, dim=1).reshape(windows, n_seeds, horizon, *starts.shape[1:])
        scores = torch.stack(instability, dim=1).reshape(windows, n_seeds, horizon)
        return latents, scores


ADAPTERS = {'combined': CombinedModel}


def build_model(config, tasks_config):
    name = config['adapter']
    if name not in ADAPTERS:
        raise ValueError(f'unknown model adapter {name}; register it in models.ADAPTERS')
    return ADAPTERS[name](config, tasks_config)

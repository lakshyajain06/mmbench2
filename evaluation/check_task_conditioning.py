"""Controlled task-embedding swap for the loaded robotics world model.

Keeps the frame, action, diffusion noise, and checkpoint fixed. Only the task
text embedding changes. This diagnoses whether language reaches latent rollout.
"""
import argparse
import json
import math
from pathlib import Path

from config import read_config
from datasets import build_dataset
from models import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='evaluation/configs/leo_expert_quick.json')
    parser.add_argument('--partition', default='expert')
    parser.add_argument('--task-a', default='mw-reach')
    parser.add_argument('--task-b', default='mw-push')
    parser.add_argument('--episode', type=int)
    args = parser.parse_args()

    config = read_config(args.config)
    dataset = build_dataset(config['dataset'])
    model = build_model(config['model'], config['tasks'])
    torch = model.torch
    if not model.info['lang_dim'] or model.dyn.task_proj is None:
        raise ValueError('this checkpoint has no task-language projection')
    episode_id = args.episode
    if episode_id is None:
        episode_id = dataset.episode_ids(args.task_a, args.partition)[0]
    episode = dataset.load_episode(args.task_a, args.partition, episode_id)
    if len(episode.frames) < 2:
        raise ValueError('episode needs at least two observations')
    metadata = json.loads(Path(config['tasks']['metadata']).read_text())
    if metadata[args.task_a]['action_dim'] != metadata[args.task_b]['action_dim']:
        raise ValueError('tasks must have the same action dimension for a controlled swap')
    embeddings = [torch.tensor(metadata[task]['text_embedding'], device=model.device,
                               dtype=torch.float32).view(1, -1)
                  for task in (args.task_a, args.task_b)]
    act_dim = int(metadata[args.task_a]['action_dim'])
    mask = torch.zeros(16, device=model.device)
    mask[:act_dim] = 1
    actions = torch.zeros((1, 2, 16), device=model.device)
    actions[0, 1, :act_dim] = episode.actions[1, :act_dim].to(model.device).clamp(-1, 1)

    with torch.inference_mode():
        start = model.encode(episode.frames[:1])[0]
        noise = torch.randn((1, 1, *start.shape), device=model.device, dtype=start.dtype,
                            generator=torch.Generator(device=model.device).manual_seed(12345))
        packed = torch.cat([start[None, None], noise], dim=1)
        k_max = int(model.info['k_max'])
        e = int(model.schedule['e'])
        step = torch.tensor([[int(round(math.log2(k_max))), e]], device=model.device)
        signal = torch.tensor([[k_max, int(model.schedule['tau_idx'][0])]], device=model.device)
        with torch.autocast(device_type=model.device.type, enabled=model.use_amp):
            direct = [model.dyn(actions, step, signal, packed, act_mask=mask, lang_emb=emb)
                      for emb in embeddings]
            projected = [model.dyn.task_proj(emb) for emb in embeddings]
        direct_spatial_diff = (direct[0][0] - direct[1][0]).float().abs().max().item()
        direct_agent_diff = (direct[0][1] - direct[1][1]).float().abs().max().item()
        projection_diff = (projected[0] - projected[1]).float().abs().max().item()

        from interactive import sample_one_timestep_packed
        rollout_diffs = {}
        for use_cache in (False, True):
            predictions = [sample_one_timestep_packed(
                model.dyn, past_packed=start[None, None], k_max=k_max,
                sched=model.schedule, actions=actions, act_mask=mask,
                use_amp=model.use_amp, tau_ctx=0.0, lang_emb=emb,
                use_kv_cache=use_cache, initial_noise=noise,
            )[0] for emb in embeddings]
            rollout_diffs['cache_on' if use_cache else 'cache_off'] = (
                predictions[0] - predictions[1]).float().abs().max().item()

    result = {
        'checkpoint': config['model']['dynamics_ckpt'],
        'source': f'{args.partition}/{args.task_a}/episode_{episode_id}',
        'task_swap': [args.task_a, args.task_b],
        'embedding_max_abs_diff': (embeddings[0] - embeddings[1]).abs().max().item(),
        'projected_agent_input_max_abs_diff': projection_diff,
        'direct_agent_output_max_abs_diff': direct_agent_diff,
        'direct_spatial_prediction_max_abs_diff': direct_spatial_diff,
        'sampled_latent_max_abs_diff': rollout_diffs,
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

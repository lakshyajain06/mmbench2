"""Gymnasium environment backed by an MMBench2 generative world model.

Only ``reset`` touches a real simulator, and only when ``seed_from_env=True``.
Every subsequent transition, reward, and RGB observation is produced by the
tokenizer/dynamics checkpoint pair.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import gymnasium as gym
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def _add_source_root() -> None:
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


class WorldModelRuntimeProtocol(Protocol):
    """Small injectable boundary used by :class:`WorldModelEnv`."""

    height: int
    width: int
    channels: int
    n_spatial: int
    d_spatial: int
    lang_dim: int
    has_reward_head: bool

    def encode(self, frame_chw_01: torch.Tensor) -> torch.Tensor: ...

    def decode(self, latent: torch.Tensor) -> torch.Tensor: ...

    def transition(
        self,
        *,
        past: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        language_embedding: Optional[torch.Tensor],
        previous_latent: Optional[torch.Tensor],
        seed: int,
    ) -> tuple[torch.Tensor, float, float]: ...


class WorldModelRuntime:
    """Checkpoint loading and one-step inference for a single Gym environment."""

    def __init__(
        self,
        *,
        tokenizer_checkpoint: str | Path,
        dynamics_checkpoint: str | Path,
        device: str | torch.device = "auto",
        packing_factor: int = 2,
        schedule: str = "shortcut",
        eval_d: float = 0.125,
        tau_ctx: float = 0.01,
        tau_init: float = 0.125,
        use_amp: bool = True,
        use_kv_cache: bool = False,
        compile_models: bool = False,
    ):
        _add_source_root()
        from interactive import (
            decode_single_packed_frame,
            load_dynamics_from_ckpt,
            load_tokenizer_from_ckpt,
            make_tau_schedule,
            reward_from_reward_head_output,
            sample_one_timestep_packed,
        )
        from model import pack_bottleneck_to_spatial, temporal_patchify

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.packing_factor = int(packing_factor)
        self.tau_ctx = float(tau_ctx)
        self.tau_init = float(tau_init)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.use_kv_cache = bool(use_kv_cache)

        tokenizer, tokenizer_info = load_tokenizer_from_ckpt(
            str(tokenizer_checkpoint), self.device
        )
        self.encoder = tokenizer.encoder
        self.decoder = tokenizer.decoder
        self.dynamics, self.reward_head, _policy_head, dynamics_info = (
            load_dynamics_from_ckpt(
                str(dynamics_checkpoint),
                device=self.device,
                d_bottleneck=int(tokenizer_info["d_bottleneck"]),
                n_latents=int(tokenizer_info["n_latents"]),
                packing_factor=self.packing_factor,
            )
        )
        self.height = int(tokenizer_info["H"])
        self.width = int(tokenizer_info["W"])
        self.channels = int(tokenizer_info["C"])
        self.patch = int(tokenizer_info["patch"])
        self.d_bottleneck = int(tokenizer_info["d_bottleneck"])
        self.n_spatial = int(dynamics_info["n_spatial"])
        self.d_spatial = int(dynamics_info["d_spatial"])
        self.k_max = int(dynamics_info["k_max"])
        self.lang_dim = int(dynamics_info["lang_dim"])
        self.schedule = make_tau_schedule(
            k_max=self.k_max,
            schedule=schedule,
            d=(float(eval_d) if schedule == "shortcut" else None),
        )
        self.has_reward_head = self.reward_head is not None

        if compile_models:
            self.dynamics = torch.compile(self.dynamics, mode="default")
            self.decoder = torch.compile(self.decoder, mode="default")

        self._decode_single = decode_single_packed_frame
        self._reward_from_output = reward_from_reward_head_output
        self._sample_one = sample_one_timestep_packed
        self._pack = pack_bottleneck_to_spatial
        self._patchify = temporal_patchify

    @torch.inference_mode()
    def encode(self, frame_chw_01: torch.Tensor) -> torch.Tensor:
        frame = frame_chw_01.to(self.device, dtype=torch.float32)
        patches = self._patchify(
            frame.view(1, 1, self.channels, self.height, self.width), self.patch
        )
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            encoded, _ = self.encoder(patches)
        return self._pack(
            encoded, n_spatial=self.n_spatial, k=self.packing_factor
        )[0, 0].float().detach()

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            frame = self._decode_single(
                self.decoder,
                z_packed=latent,
                H=self.height,
                W=self.width,
                C=self.channels,
                patch=self.patch,
                packing_factor=self.packing_factor,
                d_bottleneck=self.d_bottleneck,
            )
        return frame.float()

    @torch.inference_mode()
    def transition(
        self,
        *,
        past: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        language_embedding: Optional[torch.Tensor],
        previous_latent: Optional[torch.Tensor],
        seed: int,
    ) -> tuple[torch.Tensor, float, float]:
        # Isolate each environment's model randomness from global PyTorch RNG state.
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [self.device.index if self.device.index is not None else 0]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            need_reward = self.reward_head is not None
            result = self._sample_one(
                self.dynamics,
                past_packed=past.to(self.device),
                k_max=self.k_max,
                sched=self.schedule,
                actions=actions.to(self.device),
                act_mask=action_mask.to(self.device),
                use_amp=self.use_amp,
                return_h=need_reward,
                tau_ctx=self.tau_ctx,
                lang_emb=(
                    None
                    if language_embedding is None
                    else language_embedding.to(self.device)
                ),
                z_prev=(
                    None
                    if previous_latent is None
                    else previous_latent.to(self.device)
                ),
                tau_init=self.tau_init,
                use_kv_cache=self.use_kv_cache,
            )
        if need_reward:
            next_latent, hidden, instability = result
            logits, centers = self.reward_head(hidden[:, -1:])
            reward = self._reward_from_output(logits[0, 0], centers)
        else:
            next_latent, instability = result
            reward = 0.0
        return next_latent[0].float().detach(), float(reward), float(instability)


def _frame_to_chw01(observation: Any, *, height: int, width: int) -> torch.Tensor:
    """Convert an RGB-like observation to CHW float32 in ``[0, 1]``."""
    _add_source_root()
    from interactive import env_obs_to_frame_chw01

    return env_obs_to_frame_chw01(observation, H=height, W=width)


class WorldModelEnv(gym.Env):
    """Single-task Gymnasium environment whose transitions come from a world model.

    The model does not predict terminal states. Episodes therefore end only via
    Gymnasium truncation at ``max_episode_steps``. By default, the checkpoint's
    reward head supplies rewards. A custom ``reward_fn`` can replace it.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        *,
        task: str,
        tokenizer_checkpoint: str | Path | None = None,
        dynamics_checkpoint: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        tasks_json: str | Path = REPOSITORY_ROOT / "tasks.json",
        device: str | torch.device = "auto",
        observation_mode: str = "rgb",
        reward_mode: str = "predicted",
        reward_fn: Optional[Callable[[np.ndarray, dict[str, Any]], float]] = None,
        initial_observation: Any = None,
        initial_observation_fn: Optional[Callable[[Optional[int], dict], Any]] = None,
        seed_from_env: bool = True,
        max_episode_steps: int = 100,
        context_window: int = 24,
        action_smooth_beta: float = 0.0,
        packing_factor: int = 2,
        schedule: str = "shortcut",
        eval_d: float = 0.125,
        tau_ctx: float = 0.01,
        tau_init: float = 0.125,
        use_amp: bool = True,
        use_kv_cache: bool = False,
        compile_models: bool = False,
        render_mode: str = "rgb_array",
        runtime: Optional[WorldModelRuntimeProtocol] = None,
    ):
        super().__init__()
        if observation_mode not in {"rgb", "latent", "both"}:
            raise ValueError("observation_mode must be rgb, latent, or both")
        if reward_mode not in {"predicted", "zero", "custom"}:
            raise ValueError("reward_mode must be predicted, zero, or custom")
        if reward_mode == "custom" and reward_fn is None:
            raise ValueError("reward_mode='custom' requires reward_fn")
        if max_episode_steps < 1 or context_window < 1:
            raise ValueError("max_episode_steps and context_window must be positive")
        if not 0.0 <= action_smooth_beta < 1.0:
            raise ValueError("action_smooth_beta must be in [0, 1)")
        if render_mode != "rgb_array":
            raise ValueError("only render_mode='rgb_array' is supported")

        if runtime is None:
            if checkpoint_dir is not None:
                checkpoint_dir = Path(checkpoint_dir)
                tokenizer_checkpoint = checkpoint_dir / "tokenizer.pt"
                dynamics_checkpoint = checkpoint_dir / "dynamics.pt"
            if tokenizer_checkpoint is None or dynamics_checkpoint is None:
                raise ValueError(
                    "provide checkpoint_dir or both tokenizer_checkpoint and dynamics_checkpoint"
                )
            runtime = WorldModelRuntime(
                tokenizer_checkpoint=tokenizer_checkpoint,
                dynamics_checkpoint=dynamics_checkpoint,
                device=device,
                packing_factor=packing_factor,
                schedule=schedule,
                eval_d=eval_d,
                tau_ctx=tau_ctx,
                tau_init=tau_init,
                use_amp=use_amp,
                use_kv_cache=use_kv_cache,
                compile_models=compile_models,
            )
        self.runtime = runtime
        if reward_mode == "predicted" and not runtime.has_reward_head:
            raise ValueError(
                "reward_mode='predicted' requires a dynamics checkpoint with a reward head"
            )

        self.task = str(task)
        self.observation_mode = observation_mode
        self.reward_mode = reward_mode
        self.reward_fn = reward_fn
        self.initial_observation = initial_observation
        self.initial_observation_fn = initial_observation_fn
        self.seed_from_env = bool(seed_from_env)
        self.max_episode_steps = int(max_episode_steps)
        self.context_window = int(context_window)
        self.action_smooth_beta = float(action_smooth_beta)
        self.render_mode = render_mode
        self._seed_env = None
        self._step = 0
        self._cumulative_reward = 0.0
        self._cumulative_predicted_reward = 0.0
        self._latent_history: list[torch.Tensor] = []
        self._action_history: list[torch.Tensor] = []
        self._smoothed_action = torch.zeros(16, dtype=torch.float32)
        self._last_rgb = np.zeros(
            (runtime.channels, runtime.height, runtime.width), dtype=np.uint8
        )

        metadata = json.loads(Path(tasks_json).read_text())
        task_info = metadata.get(self.task)
        if task_info is None:
            _add_source_root()
            from interactive import TEST_TASK_SET

            task_info = metadata.get(TEST_TASK_SET.get(self.task, ""), {})
        self.action_dim = max(0, min(16, int(task_info.get("action_dim", 16))))
        self._action_mask = torch.zeros(16, dtype=torch.float32)
        self._action_mask[: self.action_dim] = 1.0
        embedding = task_info.get("text_embedding")
        if embedding is None or runtime.lang_dim <= 0:
            self._language_embedding = None
        else:
            vector = torch.tensor(embedding, dtype=torch.float32)
            if vector.numel() != runtime.lang_dim:
                raise ValueError(
                    f"task embedding has {vector.numel()} values; runtime expects {runtime.lang_dim}"
                )
            self._language_embedding = vector.view(1, -1)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        rgb_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(runtime.channels, runtime.height, runtime.width),
            dtype=np.uint8,
        )
        latent_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(runtime.n_spatial, runtime.d_spatial),
            dtype=np.float32,
        )
        if observation_mode == "rgb":
            self.observation_space = rgb_space
        elif observation_mode == "latent":
            self.observation_space = latent_space
        else:
            self.observation_space = gym.spaces.Dict(
                {"rgb": rgb_space, "latent": latent_space}
            )

    def _make_seed_env(self, seed: Optional[int]):
        if self._seed_env is None:
            os.environ.setdefault("MUJOCO_GL", "egl")
            _add_source_root()
            from env_wrapper import _EnvCfg
            from envs import make_env

            self._seed_env = make_env(
                _EnvCfg(self.task, img_size=self.runtime.width, seed=int(seed or 0))
            )
        return self._seed_env

    def _initial_frame(self, seed: Optional[int], options: dict) -> torch.Tensor:
        if "initial_observation" in options:
            observation = options["initial_observation"]
        elif self.initial_observation_fn is not None:
            observation = self.initial_observation_fn(seed, options)
        elif self.initial_observation is not None:
            observation = self.initial_observation
        elif self.seed_from_env:
            env = self._make_seed_env(seed)
            try:
                observation, _info = env.reset(seed=seed, options=options or None)
            except TypeError:
                observation, _info = env.reset()
        else:
            raise ValueError(
                "reset needs an initial observation; pass one in options, configure "
                "initial_observation/initial_observation_fn, or enable seed_from_env"
            )
        return _frame_to_chw01(
            observation, height=self.runtime.height, width=self.runtime.width
        )

    def _observation(self) -> np.ndarray | dict[str, np.ndarray]:
        latent = self._latent_history[-1].detach().float().cpu().numpy()
        if self.observation_mode == "rgb":
            return self._last_rgb.copy()
        if self.observation_mode == "latent":
            return latent.astype(np.float32, copy=False)
        return {
            "rgb": self._last_rgb.copy(),
            "latent": latent.astype(np.float32, copy=False),
        }

    def _model_seed(self) -> int:
        return int(self.np_random.integers(0, 2**31 - 1))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.action_space.seed(seed)
        options = {} if options is None else dict(options)
        frame = self._initial_frame(seed, options)
        latent = self.runtime.encode(frame).detach().float()
        self._latent_history = [latent]
        self._action_history = [torch.zeros(16, dtype=torch.float32)]
        self._smoothed_action.zero_()
        self._step = 0
        self._cumulative_reward = 0.0
        self._cumulative_predicted_reward = 0.0
        self._last_rgb = (
            frame.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
        )
        info = {
            "task": self.task,
            "model_step": 0,
            "cumulative_reward": 0.0,
            "cumulative_predicted_reward": 0.0,
            "seed_source": (
                "options"
                if "initial_observation" in options
                else "callable"
                if self.initial_observation_fn is not None
                else "fixed"
                if self.initial_observation is not None
                else "live_env"
            ),
        }
        return self._observation(), info

    def _build_model_inputs(
        self, padded_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._action_history.append(padded_action)
        end = len(self._latent_history)
        start = max(0, end - self.context_window)
        past_items = self._latent_history[start:end]
        past = torch.stack(past_items).unsqueeze(0)
        length = len(past_items)
        actions = torch.zeros((1, length + 1, 16), dtype=torch.float32)
        actions[0, :length] = torch.stack(
            self._action_history[start : start + length]
        )
        actions[0, length] = self._action_history[-1]
        mask = self._action_mask.view(1, 1, 16).expand(1, length + 1, 16).clone()
        return past, actions, mask

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self._latent_history:
            raise RuntimeError("reset() must be called before step()")
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"action shape {action_array.shape} does not match {self.action_space.shape}"
            )
        padded = torch.zeros(16, dtype=torch.float32)
        padded[: self.action_dim] = torch.from_numpy(
            np.clip(action_array, -1.0, 1.0)
        )
        if self.action_smooth_beta > 0:
            beta = self.action_smooth_beta
            self._smoothed_action = beta * self._smoothed_action + (1.0 - beta) * padded
            padded = self._smoothed_action.clone()
        padded *= self._action_mask

        past, actions, mask = self._build_model_inputs(padded)
        previous = (
            self._latent_history[-1].unsqueeze(0)
            if getattr(self.runtime, "tau_init", 0.0) > 0
            else None
        )
        next_latent, predicted_reward, instability = self.runtime.transition(
            past=past,
            actions=actions,
            action_mask=mask,
            language_embedding=self._language_embedding,
            previous_latent=previous,
            seed=self._model_seed(),
        )
        self._latent_history.append(next_latent.detach().float())
        cap = self.context_window + 1
        if len(self._latent_history) > cap:
            self._latent_history = self._latent_history[-cap:]
            self._action_history = self._action_history[-cap:]

        decoded = self.runtime.decode(next_latent)
        self._last_rgb = (
            decoded.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
        )
        self._step += 1
        self._cumulative_predicted_reward += float(predicted_reward)
        terminated = False
        truncated = self._step >= self.max_episode_steps
        info = {
            "task": self.task,
            "model_step": self._step,
            "predicted_reward": float(predicted_reward),
            "flow_instability": float(instability),
            "action_padded": padded.numpy().copy(),
            "is_success": False,
            "termination_is_modeled": False,
        }
        observation = self._observation()
        if self.reward_mode == "predicted":
            reward = float(predicted_reward)
        elif self.reward_mode == "custom":
            assert self.reward_fn is not None
            reward = float(self.reward_fn(self._last_rgb.copy(), dict(info)))
        else:
            reward = 0.0
        self._cumulative_reward += reward
        info["cumulative_reward"] = self._cumulative_reward
        info["cumulative_predicted_reward"] = self._cumulative_predicted_reward
        if truncated:
            info["TimeLimit.truncated"] = True
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Return the current generated frame in Gymnasium HWC format."""
        return np.transpose(self._last_rgb, (1, 2, 0)).copy()

    def close(self) -> None:
        if self._seed_env is not None and hasattr(self._seed_env, "close"):
            self._seed_env.close()
        self._seed_env = None


def make_world_model_env(
    task: str,
    checkpoint_dir: str | Path,
    **kwargs,
) -> WorldModelEnv:
    """Convenience factory using ``checkpoint_dir/{tokenizer,dynamics}.pt``."""
    return WorldModelEnv(
        task=task,
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )

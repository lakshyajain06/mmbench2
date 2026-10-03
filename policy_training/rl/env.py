"""MMBench environment adapters for visual policy learning."""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


def _add_source_root() -> None:
	# Keep simulator imports lazy so offline model/tests do not require them.
	repository_root = Path(__file__).resolve().parents[2]
	source_root = str(repository_root / "src")
	if source_root not in sys.path:
		sys.path.insert(0, source_root)


class RGBObservation(gym.ObservationWrapper):
    """Expose only CHW uint8 RGB and standardize the success info key."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        rgb_space = env.observation_space.spaces["rgb"]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=rgb_space.shape, dtype=np.uint8
        )

    def observation(self, observation):
        rgb = np.asarray(observation["rgb"])
        return rgb.astype(np.uint8, copy=False)

    def reset(self, *, seed=None, options=None):
        # Some repository task wrappers predate Gymnasium's seeded-reset
        # signature. Keep the standard interface at this boundary without
        # modifying the shared environment implementation.
        if seed is not None:
            self.action_space.seed(seed)
            np.random.seed(seed)
        try:
            observation, info = self.env.reset(seed=seed, options=options)
        except TypeError as error:
            if "unexpected keyword argument" not in str(error):
                raise
            observation, info = self.env.reset()
        return self.observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["is_success"] = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
        return self.observation(observation), float(reward), terminated, truncated, info


class RGBHistory(gym.Wrapper):
    """Stack the current and previous RGB observations, padding resets in place."""

    def __init__(self, env: gym.Env, context_length: int):
        super().__init__(env)
        if context_length < 1:
            raise ValueError("context_length must be positive")
        self.context_length = int(context_length)
        shape = (self.context_length,) + tuple(env.observation_space.shape)
        self.observation_space = gym.spaces.Box(0, 255, shape, dtype=np.uint8)
        self._history = deque(maxlen=self.context_length)

    def _stack(self):
        return np.stack(tuple(self._history), axis=0)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._history.clear()
        self._history.extend(observation.copy() for _ in range(self.context_length))
        return self._stack(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._history.append(observation)
        return self._stack(), reward, terminated, truncated, info


def make_rgb_env(task: str, *, seed: int = 0, image_size: int = 224) -> gym.Env:
    """Create one CPU-compatible MMBench task for evaluation and videos."""
    _add_source_root()
    from env_wrapper import _EnvCfg
    from envs import make_env

    env = make_env(_EnvCfg(task, img_size=image_size, seed=seed))
    env = RGBObservation(env)
    env.action_space.seed(seed)
    return env


class _RLEnvCfg:
    """Configuration expected by MMBench's native ManiSkill constructor."""

    def __init__(
        self,
        task: str,
        num_envs: int,
        image_size: int,
        context_length: int,
        seed: int,
        sim_backend: str,
    ):
        self.task = task
        self.obs = "rgb"
        self.num_envs = int(num_envs)
        self.render_size = int(image_size)
        self.context_length = int(context_length)
        self.seed = int(seed)
        self.sim_backend = sim_backend

    def get(self, key, default=None):
        return getattr(self, key, default)


class MMBenchManiSkillEnv:
    """Adapt native ManiSkill tensors to skrl without losing terminal frames."""

    def __init__(self, env):
        self._env = env
        self._observations = None
        self._reset_once = True

    @property
    def device(self):
        return torch.device(self._env.device)

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def observation_space(self):
        return self._env.single_observation_space

    @property
    def action_space(self):
        return self._env.single_action_space

    @property
    def state_space(self):
        return None

    def state(self):
        return None

    @staticmethod
    def _flatten(space, value):
        from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space

        return flatten_tensorized_space(tensorize_space(space, value))

    def reset(self):
        if self._reset_once:
            observation, info = self._env.reset(seed=self._env.cfg.seed)
            self._observations = self._flatten(self.observation_space, observation)
            self._reset_once = False
            return self._observations, info
        return self._observations, {}

    def step(self, actions):
        from skrl.utils.spaces.torch import unflatten_tensorized_space

        actions = unflatten_tensorized_space(self.action_space, actions)
        low = torch.as_tensor(self.action_space.low, device=actions.device)
        high = torch.as_tensor(self.action_space.high, device=actions.device)
        environment_actions = actions.clamp(low, high)
        with torch.no_grad():
            observation, reward, terminated, truncated, info = self._env.step(
                environment_actions
            )
            flattened = self._flatten(self.observation_space, observation)
            done = (terminated | truncated).flatten()
            if done.any():
                final_observation = flattened.clone()
                final_info = info
                env_idx = torch.arange(self.num_envs, device=done.device)[done]
                observation, reset_info = self._env.reset(options={"env_idx": env_idx})
                flattened = self._flatten(self.observation_space, observation)
                info = dict(reset_info)
                info.update(
                    {
                        "final_observation": final_observation,
                        "final_info": final_info,
                        "_final_observation": done,
                        "_final_info": done,
                    }
                )
            self._observations = flattened
        return (
            flattened,
            reward.view(-1, 1),
            terminated.view(-1, 1),
            truncated.view(-1, 1),
            info,
        )

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        self._env.close()


def make_training_env(
    task: str,
    *,
    num_envs: int,
    seed: int = 0,
    image_size: int = 64,
    context_length: int = 1,
    sim_backend: str = "physx_cuda",
):
    """Create and skrl-wrap a vectorized MMBench environment.

    ManiSkill tasks use MMBench's native GPU-vectorized constructor. Other
    MMBench domains retain their normal environment constructor and are
    vectorized through Gymnasium, which keeps the learner environment-agnostic.
    """
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    _add_source_root()
    from skrl.envs.wrappers.torch import wrap_env

    if task.startswith("ms-"):
        from envs.maniskill import make_gpu_env

        cfg = _RLEnvCfg(
            task, num_envs, image_size, context_length, seed, sim_backend
        )
        env = make_gpu_env(cfg)
        return MMBenchManiSkillEnv(env)

    def factory(index):
        env = make_rgb_env(task, seed=seed + index, image_size=image_size)
        return RGBHistory(env, context_length) if context_length > 1 else env

    factories = [lambda index=index: factory(index) for index in range(num_envs)]
    env = gym.vector.SyncVectorEnv(factories)
    return wrap_env(env, wrapper="gymnasium")

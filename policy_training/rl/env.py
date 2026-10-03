"""MMBench environment adapters for visual policy learning."""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np


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
        # The explicit tag ensures skrl adds ManiSkillVectorEnv/autoreset while
        # retaining torch tensors on the simulator device.
        return wrap_env(env, wrapper="mani-skill")

    def factory(index):
        env = make_rgb_env(task, seed=seed + index, image_size=image_size)
        return RGBHistory(env, context_length) if context_length > 1 else env

    factories = [lambda index=index: factory(index) for index in range(num_envs)]
    env = gym.vector.SyncVectorEnv(factories)
    return wrap_env(env, wrapper="gymnasium")

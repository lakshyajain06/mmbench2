import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from gymnasium.utils.env_checker import check_env

from policy_training.rl.world_model_env import WorldModelEnv


class FakeRuntime:
    height = 8
    width = 8
    channels = 3
    n_spatial = 2
    d_spatial = 4
    lang_dim = 3
    has_reward_head = True
    tau_init = 0.0

    def __init__(self):
        self.last_actions = None
        self.last_mask = None
        self.last_language = None
        self.last_seed = None

    def encode(self, frame):
        value = frame.float().mean()
        return torch.full((self.n_spatial, self.d_spatial), value)

    def decode(self, latent):
        value = latent.float().mean().clamp(0, 1)
        return torch.full((self.channels, self.height, self.width), value)

    def transition(
        self,
        *,
        past,
        actions,
        action_mask,
        language_embedding,
        previous_latent,
        seed,
    ):
        self.last_actions = actions.clone()
        self.last_mask = action_mask.clone()
        self.last_language = None if language_embedding is None else language_embedding.clone()
        self.last_seed = seed
        increment = float(actions[0, -1, 0]) * 0.1
        return past[0, -1] + increment, float(actions[0, -1, 0]), 0.25


class WorldModelEnvTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.tasks_json = Path(self.temp.name) / "tasks.json"
        self.tasks_json.write_text(
            json.dumps(
                {
                    "test-task": {
                        "action_dim": 2,
                        "text_embedding": [1.0, 2.0, 3.0],
                    }
                }
            )
        )

    def tearDown(self):
        self.temp.cleanup()

    def make_env(self, **kwargs):
        return WorldModelEnv(
            task="test-task",
            tasks_json=self.tasks_json,
            runtime=kwargs.pop("runtime", FakeRuntime()),
            initial_observation=np.zeros((3, 8, 8), dtype=np.uint8),
            seed_from_env=False,
            max_episode_steps=2,
            **kwargs,
        )

    def test_gymnasium_contract_and_predicted_reward(self):
        runtime = FakeRuntime()
        env = self.make_env(runtime=runtime, observation_mode="rgb")
        observation, info = env.reset(seed=7)
        self.assertEqual(observation.shape, (3, 8, 8))
        self.assertEqual(observation.dtype, np.uint8)
        self.assertEqual(info["seed_source"], "fixed")

        observation, reward, terminated, truncated, info = env.step(
            np.array([0.5, -0.25], dtype=np.float32)
        )
        self.assertAlmostEqual(reward, 0.5)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(info["flow_instability"], 0.25)
        self.assertEqual(runtime.last_actions.shape, (1, 2, 16))
        np.testing.assert_allclose(runtime.last_actions[0, -1, :2], [0.5, -0.25])
        np.testing.assert_allclose(runtime.last_actions[0, -1, 2:], 0.0)
        torch.testing.assert_close(runtime.last_language, torch.tensor([[1.0, 2.0, 3.0]]))

        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["TimeLimit.truncated"])
        self.assertEqual(env.render().shape, (8, 8, 3))
        env.close()

    def test_latent_and_dict_observation_modes(self):
        latent_env = self.make_env(observation_mode="latent", reward_mode="zero")
        latent, _ = latent_env.reset(seed=1)
        self.assertEqual(latent.shape, (2, 4))
        self.assertEqual(latent.dtype, np.float32)
        _, reward, _, _, _ = latent_env.step(np.zeros(2, dtype=np.float32))
        self.assertEqual(reward, 0.0)

        both_env = self.make_env(observation_mode="both")
        observation, _ = both_env.reset(seed=1)
        self.assertEqual(set(observation), {"rgb", "latent"})
        self.assertTrue(both_env.observation_space.contains(observation))

    def test_reset_option_overrides_fixed_initial_observation(self):
        env = self.make_env(observation_mode="rgb")
        white = np.full((8, 8, 3), 255, dtype=np.uint8)
        observation, info = env.reset(
            seed=2, options={"initial_observation": white}
        )
        self.assertEqual(info["seed_source"], "options")
        self.assertTrue(np.all(observation == 255))

    def test_seed_reproducibility(self):
        runtime = FakeRuntime()
        env = self.make_env(runtime=runtime)
        env.reset(seed=123)
        env.step(np.zeros(2, dtype=np.float32))
        first_seed = runtime.last_seed
        env.reset(seed=123)
        env.step(np.zeros(2, dtype=np.float32))
        self.assertEqual(first_seed, runtime.last_seed)

    def test_checker(self):
        check_env(self.make_env(), skip_render_check=False)


if __name__ == "__main__":
    unittest.main()

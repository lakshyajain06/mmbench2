import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from policy_training.bc.resnet_mlp_model import ResNetMLPPolicy
from policy_training.rl.env import MMBenchManiSkillEnv, RGBObservation
from policy_training.rl.ppo import (
    ResNetGaussianPolicy,
    StateValue,
    evaluate_ppo,
    final_observations_for_bootstrap,
    final_states_for_bootstrap,
    initialize_policy_from_bc,
    load_portable_policy,
    save_portable_checkpoint,
)
from policy_training.rl.train_ppo import _apply_preset, _validate, parser


class DummyRGBEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8),
                "state": gym.spaces.Box(-np.inf, np.inf, (5,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        return {
            "rgb": np.zeros((3, 64, 64), dtype=np.uint8),
            "state": np.zeros(5, dtype=np.float32),
        }, {}

    def step(self, action):
        self.steps += 1
        done = self.steps >= 2
        observation = {
            "rgb": np.zeros((3, 64, 64), dtype=np.uint8),
            "state": np.zeros(5, dtype=np.float32),
        }
        return observation, 1.0, False, done, {"success": float(done)}


class PPOPipelineTest(unittest.TestCase):
    def test_maniskill_and_legacy_presets(self):
        reference = parser().parse_args(["--task", "ms-pick-cube"])
        _apply_preset(reference)
        _validate(reference)
        self.assertEqual(reference.num_envs, 1024)
        self.assertEqual(reference.n_steps, 16)
        self.assertEqual(reference.batch_size, 512)
        self.assertEqual(reference.n_epochs, 8)
        self.assertEqual(reference.learning_rate, 3e-4)
        self.assertEqual(reference.value_clip, 0.0)

        legacy = parser().parse_args(
            ["--task", "ms-pick-cube", "--preset", "legacy"]
        )
        _apply_preset(legacy)
        _validate(legacy)
        self.assertEqual(legacy.num_envs, 64)
        self.assertEqual(legacy.n_steps, 32)
        self.assertEqual(legacy.batch_size, 256)
        self.assertEqual(legacy.learning_rate, 1e-5)

        override = parser().parse_args(
            [
                "--task",
                "ms-pick-cube",
                "--num-envs",
                "8",
                "--n-steps",
                "4",
                "--num-minibatches",
                "4",
            ]
        )
        _apply_preset(override)
        _validate(override)
        self.assertEqual(override.batch_size, 8)

    def test_rgb_wrapper(self):
        env = RGBObservation(DummyRGBEnv())
        observation, _ = env.reset()
        self.assertEqual(observation.shape, (3, 64, 64))
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertIn("is_success", info)

    def test_evaluation_distinguishes_success_once_and_at_end(self):
        class TransientSuccessEnv(DummyRGBEnv):
            def step(self, action):
                self.steps += 1
                done = self.steps >= 2
                observation = {
                    "rgb": np.zeros((3, 64, 64), dtype=np.uint8),
                    "state": np.zeros(5, dtype=np.float32),
                }
                return observation, 1.0, False, done, {
                    "success": float(self.steps == 1)
                }

        class Policy:
            context_length = 1

            def eval(self):
                return self

            def predict(self, observation):
                return np.zeros(4, dtype=np.float32)

        with patch(
            "policy_training.rl.ppo.make_rgb_env",
            return_value=RGBObservation(TransientSuccessEnv()),
        ):
            metrics = evaluate_ppo(
                Policy(), "ms-test", episodes=1, seed=0, image_size=64
            )
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertEqual(metrics["success_once_rate"], 1.0)
        self.assertEqual(metrics["success_at_end_rate"], 0.0)

    def test_full_bc_actor_initialization_and_portable_checkpoint(self):
        bc = ResNetMLPPolicy(
            num_tasks=1,
            context_length=2,
            chunk_size=3,
            action_dim=16,
            hidden_dim=32,
            task_embedding_dim=8,
            mlp_layers=1,
            dropout=0.0,
            pretrained_backbone=False,
        )
        checkpoint = {
            "policy_type": "resnet_mlp_visual_bc",
            "tasks": ["ms-test"],
            "model": bc.state_dict(),
            "model_config": {
                "context_length": 2,
                "hidden_dim": 32,
                "task_embedding_dim": 8,
                "mlp_layers": 1,
                "dropout": 0.0,
            },
        }
        observation_space = gym.spaces.Box(0, 255, (2, 3, 64, 64), dtype=np.uint8)
        action_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        policy = ResNetGaussianPolicy(
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            image_size=64,
            context_length=2,
            hidden_dim=32,
            task_embedding_dim=8,
            mlp_layers=1,
            pretrained_backbone=False,
        ).to("cpu")
        initialize_policy_from_bc(policy, checkpoint, 0)
        torch.testing.assert_close(policy.backbone.conv1.weight, bc.backbone.conv1.weight)
        torch.testing.assert_close(policy.mlp[0].weight, bc.mlp[0].weight)
        torch.testing.assert_close(policy.mlp[4].weight, bc.mlp[4].weight[:4])
        observation = np.zeros((2, 3, 64, 64), dtype=np.uint8)
        bc.eval()
        policy.eval()
        with torch.no_grad():
            expected = bc(
                torch.from_numpy(observation)[None], torch.tensor([0])
            )["actions"][0, 0, :4].numpy()
        np.testing.assert_allclose(policy.predict(observation), expected, atol=1e-6)

        metadata = {
            "task": "ms-test",
            "action_dim": 4,
            "image_size": 64,
            "context_length": 2,
            "hidden_dim": 32,
            "task_embedding_dim": 8,
            "mlp_layers": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            save_portable_checkpoint(path, policy, None, metadata)
            loaded, loaded_metadata = load_portable_policy(path, device="cpu")
            self.assertEqual(loaded_metadata["task"], "ms-test")
            np.testing.assert_allclose(loaded.predict(observation), expected, atol=1e-6)

    def test_partial_autoreset_preserves_final_observation(self):
        class TensorEnv:
            num_envs = 2
            device = torch.device("cpu")
            cfg = type("Cfg", (), {"seed": 7})()
            single_observation_space = gym.spaces.Box(
                -1000, 1000, (1,), dtype=np.float32
            )
            single_state_space = gym.spaces.Box(-1000, 1000, (1,), dtype=np.float32)
            single_action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

            def __init__(self):
                self.observation = torch.tensor([[1.0], [2.0]])
                self.privileged_state = torch.tensor([[3.0], [4.0]])
                self.last_action = None

            def reset(self, *, seed=None, options=None):
                if options is not None:
                    self.observation[options["env_idx"]] = 100.0
                    self.privileged_state[options["env_idx"]] = 1000.0
                return self.observation.clone(), {"reset": True}

            def state(self):
                return self.privileged_state.clone()

            def step(self, action):
                self.last_action = action.clone()
                self.observation = torch.tensor([[10.0], [20.0]])
                self.privileged_state = torch.tensor([[30.0], [40.0]])
                return (
                    self.observation.clone(),
                    torch.ones(2),
                    torch.zeros(2, dtype=torch.bool),
                    torch.tensor([True, False]),
                    {"terminal": True},
                )

            def close(self):
                pass

        tensor_env = TensorEnv()
        env = MMBenchManiSkillEnv(tensor_env)
        env.reset()
        observation, _, _, truncated, info = env.step(
            torch.tensor([[2.0], [-3.0]])
        )
        torch.testing.assert_close(tensor_env.last_action, torch.tensor([[1.0], [-1.0]]))
        torch.testing.assert_close(observation, torch.tensor([[100.0], [20.0]]))
        torch.testing.assert_close(
            info["final_observation"], torch.tensor([[10.0], [20.0]])
        )
        torch.testing.assert_close(info["final_state"], torch.tensor([[30.0], [40.0]]))
        torch.testing.assert_close(truncated, torch.tensor([[True], [False]]))

    def test_timeout_bootstrap_uses_final_not_reset_observation(self):
        reset_observation = torch.tensor([[100.0], [200.0]])
        final_observation = torch.tensor([[10.0], [20.0]])
        truncated = torch.tensor([[True], [False]])
        selected = final_observations_for_bootstrap(
            reset_observation,
            truncated,
            {
                "final_observation": final_observation,
                "_final_observation": torch.tensor([True, False]),
            },
        )
        torch.testing.assert_close(selected, torch.tensor([[10.0], [200.0]]))

        selected_states = final_states_for_bootstrap(
            torch.tensor([[1000.0], [2000.0]]),
            truncated,
            {
                "final_state": torch.tensor([[30.0], [40.0]]),
                "_final_observation": torch.tensor([True, False]),
            },
        )
        torch.testing.assert_close(selected_states, torch.tensor([[30.0], [2000.0]]))

    def test_privileged_state_value(self):
        value = StateValue(
            state_space=gym.spaces.Box(-1, 1, (5,), dtype=np.float32),
            action_space=gym.spaces.Box(-1, 1, (4,), dtype=np.float32),
            device="cpu",
            hidden_dim=16,
        )
        result, _ = value.compute({"states": torch.zeros((3, 5))})
        self.assertEqual(result.shape, (3, 1))

    def test_policy_log_probability_uses_unclipped_sample(self):
        observation_space = gym.spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)
        action_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        policy = ResNetGaussianPolicy(
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            image_size=64,
            hidden_dim=16,
            task_embedding_dim=4,
            mlp_layers=1,
            initial_log_std=1.0,
            pretrained_backbone=False,
        ).eval()
        observations = torch.zeros((16, 3, 64, 64), dtype=torch.uint8)
        with torch.no_grad():
            actions, rollout = policy.act({"observations": observations})
            _, update = policy.act(
                {"observations": observations, "taken_actions": actions}
            )
        self.assertTrue(torch.any(actions.abs() > 1))
        torch.testing.assert_close(rollout["log_prob"], update["log_prob"])


if __name__ == "__main__":
    unittest.main()

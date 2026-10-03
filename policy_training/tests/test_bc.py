import tempfile
import unittest
from pathlib import Path

import torch

from policy_training.bc.dataset import MMBenchStateBCDataset
from policy_training.bc.model import ChunkedBCPolicy, masked_action_mse
from policy_training.bc.visual_model import DenseVisualChunkPolicy, block_causal_mask
from policy_training.bc.act_model import SimpleACTPolicy, act_loss
from policy_training.bc.eval_act import evaluate_policy, observation_to_rgb
from policy_training.bc.resnet_mlp_model import ResNetMLPPolicy, resnet_mlp_loss
from policy_training.bc.train_act import rollout_metrics_for_logging


class BehaviorCloningTest(unittest.TestCase):
    def test_action_alignment_and_episode_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "expert").mkdir()
            data = {
                "obs": torch.arange(30, dtype=torch.float32).view(6, 5),
                "action": torch.tensor(
                    [
                        [float("nan"), float("nan")],
                        [0.1, 0.2],
                        [0.3, 0.4],
                        [float("nan"), float("nan")],
                        [0.5, 0.6],
                        [0.7, 0.8],
                    ]
                ),
                "episode": torch.tensor([0, 0, 0, 1, 1, 1]),
            }
            torch.save(data, root / "expert" / "ms-test.pt")
            metadata = root / "tasks.json"
            metadata.write_text('{"ms-test": {"action_dim": 2}}')
            dataset = MMBenchStateBCDataset(
                root,
                ["expert"],
                ["ms-test"],
                tasks_json=metadata,
                context_length=2,
                chunk_size=2,
                max_action_dim=2,
            )
            self.assertEqual(len(dataset), 2)
            first = dataset[0]
            torch.testing.assert_close(
                first["action"], torch.tensor([[0.1, 0.2], [0.3, 0.4]])
            )
            torch.testing.assert_close(first["observation"][0], first["observation"][1])

    def test_policy_and_masked_loss(self):
        policy = ChunkedBCPolicy(
            5,
            num_tasks=2,
            context_length=2,
            chunk_size=3,
            action_dim=4,
            d_model=32,
            n_heads=4,
            n_layers=1,
        )
        prediction = policy(torch.randn(2, 2, 5), torch.tensor([0, 1]))
        self.assertEqual(prediction.shape, (2, 3, 4))
        self.assertTrue((prediction.abs() <= 1).all())
        mask = torch.zeros_like(prediction, dtype=torch.bool)
        mask[..., :2] = True
        loss = masked_action_mse(prediction, torch.zeros_like(prediction), mask)
        self.assertTrue(torch.isfinite(loss))

    def test_dense_visual_policy_and_block_causality(self):
        mask = block_causal_mask(context_length=2, tokens_per_frame=3)
        self.assertFalse(mask[:3, :3].any())
        self.assertTrue(mask[:3, 3:].all())
        self.assertFalse(mask[3:, :].any())
        policy = DenseVisualChunkPolicy(
            visual_dim=8,
            tokens_per_frame=3,
            num_tasks=2,
            context_length=2,
            chunk_size=4,
            action_dim=5,
            d_model=32,
            n_heads=4,
            n_layers=1,
        )
        output = policy(torch.randn(2, 2, 3, 8), torch.tensor([0, 1]))
        self.assertEqual(output.shape, (2, 4, 5))
        self.assertTrue((output.abs() <= 1).all())

    def test_simple_act_training_and_inference(self):
        policy = SimpleACTPolicy(
            num_tasks=2,
            chunk_size=3,
            action_dim=4,
            d_model=32,
            n_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            latent_dim=8,
            pretrained_backbone=False,
        )
        frames = torch.randint(0, 256, (2, 1, 3, 64, 64), dtype=torch.uint8)
        tasks = torch.tensor([0, 1])
        target = torch.randn(2, 3, 4).clamp(-1, 1)
        mask = torch.ones_like(target, dtype=torch.bool)
        training_output = policy(frames, tasks, target)
        loss, components = act_loss(training_output, target, mask, kl_weight=1.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(components["kl"]), 0.0)
        inference_output = policy(frames, tasks)
        self.assertEqual(inference_output["actions"].shape, target.shape)
        self.assertIsNone(inference_output["posterior_mean"])

    def test_resnet_mlp_training_and_inference(self):
        policy = ResNetMLPPolicy(
            num_tasks=2,
            context_length=3,
            chunk_size=3,
            action_dim=4,
            hidden_dim=32,
            task_embedding_dim=8,
            mlp_layers=2,
            pretrained_backbone=False,
        )
        frames = torch.randint(0, 256, (2, 3, 3, 64, 64), dtype=torch.uint8)
        target = torch.randn(2, 3, 4).clamp(-1, 1)
        mask = torch.ones_like(target, dtype=torch.bool)
        output = policy(frames, torch.tensor([0, 1]))
        loss, components = resnet_mlp_loss(output, target, mask)
        self.assertEqual(output["actions"].shape, target.shape)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(components["kl"]), 0.0)
        with self.assertRaisesRegex(ValueError, "Expected 3 observation frames"):
            policy(frames[:, :2], torch.tensor([0, 1]))

    def test_rollout_evaluator(self):
        class DummyPolicy(torch.nn.Module):
            num_tasks = 1
            chunk_size = 2

            def forward(self, frames, task_index):
                self.seen_shape = tuple(frames.shape)
                return {"actions": torch.zeros(len(frames), 2, 4)}

        class DummyEnv:
            class ActionSpace:
                shape = (2,)

            action_space = ActionSpace()

            def reset(self):
                self.step_count = 0
                return {"rgb": torch.zeros(24, 32, 3, dtype=torch.uint8)}, {}

            def step(self, action):
                self.step_count += 1
                done = self.step_count == 3
                info = {"success": float(done)}
                return torch.zeros(3, 24, 32, dtype=torch.uint8), 1.0, False, done, info

            def close(self):
                pass

        policy = DummyPolicy()
        metrics = evaluate_policy(
            policy,
            ["dummy"],
            device="cpu",
            episodes=2,
            image_size=32,
            env_factory=lambda task, seed, image_size: DummyEnv(),
        )
        self.assertEqual(policy.seen_shape, (1, 1, 3, 32, 32))
        self.assertEqual(metrics["mean_success_rate"], 1.0)
        self.assertEqual(metrics["tasks"]["dummy"]["mean_return"], 3.0)
        self.assertEqual(tuple(observation_to_rgb(torch.zeros(10, 12, 3), 16).shape), (3, 16, 16))

    def test_rollout_metrics_are_flattened_for_wandb(self):
        logged = rollout_metrics_for_logging(
            {
                "mean_success_rate": 0.5,
                "mean_return": 4.0,
                "tasks": {
                    "ms-reach": {
                        "success_rate": 0.5,
                        "mean_return": 4.0,
                        "return_std": 1.0,
                        "mean_length": 25.0,
                    }
                },
            }
        )
        self.assertEqual(logged["eval/mean_success_rate"], 0.5)
        self.assertEqual(logged["eval/task/ms-reach/mean_length"], 25.0)


if __name__ == "__main__":
    unittest.main()

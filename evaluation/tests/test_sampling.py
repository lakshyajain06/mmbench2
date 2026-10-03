"""Checks that horizon comparisons can use identical starting frames."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import sample_horizon_starts, sample_starts, stable_seed


class HorizonSamplingTests(unittest.TestCase):
    def test_explicit_random_mode_is_reproducible_and_paired(self):
        first = sample_horizon_starts(101, [1, 4, 8, 16], 3, 2026,
                                      paired=True, start_mode='random')
        second = sample_horizon_starts(101, [1, 4, 8, 16], 3, 2026,
                                       paired=True, start_mode='random')
        self.assertEqual(first, second)
        self.assertEqual(first[1], first[16])
        self.assertEqual(len(first[16]), 3)
        self.assertTrue(all(0 <= start < 85 for start in first[16]))

    def test_random_mode_rejects_fixed_phase(self):
        with self.assertRaises(ValueError):
            sample_horizon_starts(101, [16], 1, 2026, paired=True,
                                  phase_fraction=0.16, start_mode='random')

    def test_task_scoped_seed_is_stable_and_task_specific(self):
        first = stable_seed(2026, 'val', 'ms-pick-cube', 3)
        self.assertEqual(first, stable_seed(2026, 'val', 'ms-pick-cube', 3))
        self.assertNotEqual(first, stable_seed(2026, 'val', 'ms-push-cube', 3))

    def test_fixed_phase_is_valid_and_paired_across_episode_lengths(self):
        for length, expected in ((26, 4), (101, 16)):
            starts = sample_horizon_starts(length, [1, 4, 8, 16], 1, 2026,
                                           paired=True, phase_fraction=0.16)
            self.assertEqual(starts, {horizon: [expected] for horizon in [1, 4, 8, 16]})

    def test_fixed_phase_requires_paired_single_window(self):
        with self.assertRaises(ValueError):
            sample_horizon_starts(26, [16], 1, 2026, phase_fraction=0.16)
        with self.assertRaises(ValueError):
            sample_horizon_starts(26, [16], 2, 2026, paired=True, phase_fraction=0.16)

    def test_paired_starts_are_identical_and_valid(self):
        starts = sample_horizon_starts(26, [1, 4, 8, 16], 4, 2026, paired=True)
        self.assertEqual(starts[1], starts[4])
        self.assertEqual(starts[4], starts[8])
        self.assertEqual(starts[8], starts[16])
        self.assertTrue(all(0 <= start < 10 for start in starts[16]))

    def test_default_sampling_preserves_existing_seeds(self):
        starts = sample_horizon_starts(26, [1, 4, 8, 16], 1, 2026)
        for horizon in starts:
            self.assertEqual(starts[horizon], sample_starts(26, horizon, 1,
                                                            2026 + horizon))


if __name__ == '__main__':
    unittest.main()

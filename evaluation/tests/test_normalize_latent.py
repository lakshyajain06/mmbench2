"""Normalized latent comparisons use a group baseline, not unstable window ratios."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from normalize_latent import read_reference_scales, summarize


class NormalizeLatentTests(unittest.TestCase):
    def test_reference_scales_reuse_prior_task_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'summary.csv'
            path.write_text('task,task_state_latent_std\nms-reach,0.37\nms-reach,0.37\nmw-reach,0.36\n')
            self.assertEqual(read_reference_scales(path),
                             {'ms-reach': 0.37, 'mw-reach': 0.36})

    def test_ratio_of_group_means_and_fraction_beating_copy(self):
        rows = [
            dict(partition='expert', task='ms-stack-cube', horizon=4,
                 model_latent_rms=.2, copy_start_latent_rms=.1, model_beats_copy=False),
            dict(partition='expert', task='ms-stack-cube', horizon=4,
                 model_latent_rms=.2, copy_start_latent_rms=.3, model_beats_copy=True),
        ]
        result = summarize(rows)[0]
        self.assertAlmostEqual(result['normalized_latent_error'], 1.0)
        self.assertAlmostEqual(result['fraction_model_beats_copy'], .5)
        self.assertEqual(result['windows'], 2)


if __name__ == '__main__':
    unittest.main()

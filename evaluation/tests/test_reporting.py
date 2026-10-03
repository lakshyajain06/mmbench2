import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import read_config, sample_starts
from report import make_review_sheet, read_labels, summarize


class ReportingTest(unittest.TestCase):
    def test_config_and_sampling(self):
        config = read_config(Path(__file__).parents[1] / 'configs/robotics.json',
                             ['sampling.horizons=[1,2]'])
        self.assertEqual(config['sampling']['horizons'], [1, 2])
        self.assertTrue(Path(config['model']['tokenizer_ckpt']).exists())
        self.assertEqual(sample_starts(3, 4, 1, 0), [])

    def test_review_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window = dict(partition='val', task='mw-push', episode=1, start=3,
                          horizon=2, seed=11)
            (root / 'windows.jsonl').write_text(json.dumps(window) + '\n')
            steps = [dict(window, transition=i, frame=3+i, latent_rms=float(i),
                          image_rms=0.2, tokenizer_image_rms=0.1) for i in (1, 2)]
            (root / 'steps.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in steps))
            make_review_sheet(root / 'windows.jsonl', root / 'review.csv')
            with (root / 'review.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            rows[0].update(useful_plausible='no', event='unsupported_object_motion',
                           first_error_transition='2', real_action_support='yes')
            with (root / 'review.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            make_review_sheet(root / 'windows.jsonl', root / 'review.csv')
            self.assertEqual(len(read_labels(root / 'review.csv')), 1)
            windows, summary = summarize(root / 'steps.jsonl', root / 'review.csv',
                                         dict(min_useful_rate=.6, min_hallucination_rate=.05,
                                              max_hallucination_rate=.3, min_affected_episodes=1))
            self.assertEqual(windows[0]['first_error_frame'], 5)
            self.assertEqual(windows[0]['latent_rms'], 1.5)
            self.assertEqual(summary[0]['hallucination_rate'], 1.0)


if __name__ == '__main__':
    unittest.main()

class PredictorTest(unittest.TestCase):
    def test_val_thresholds_and_test_counts(self):
        from predictors import calibrate, evaluate
        examples = [({'u_r_norm': .1, 'u_f_norm': .2, 'u_s': .1}, False),
                    ({'u_r_norm': .9, 'u_f_norm': .8, 'u_s': .7}, True)]
        thresholds = calibrate(examples)
        result = evaluate(examples, thresholds)
        self.assertEqual(result['u_r_norm']['tp'], 1)
        self.assertEqual(result['u_r_norm']['fp'], 0)

class AggregationTest(unittest.TestCase):
    def test_partition_metrics_and_completeness(self):
        from aggregate_metrics import aggregate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = dict(partition='expert', task='ms-reach', episode=1,
                       start=2, horizon=2, seed=11)
            (root / 'windows.jsonl').write_text(json.dumps(key) + '\n')
            rows = [dict(key, transition=i, latent_rms=float(i), image_rms=.2,
                         tokenizer_image_rms=.1, u_r_norm=.3, u_f_norm=.4,
                         u_s=.5) for i in (1, 2)]
            (root / 'steps.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            summary, status = aggregate([root])
            self.assertEqual(status['windows'], 1)
            self.assertEqual(summary[0]['mean_latent_rms'], 1.5)
            self.assertEqual(summary[0]['terminal_latent_rms'], 2.0)

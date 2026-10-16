from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import json
from unittest import TestCase

CLOUD = Path(__file__).resolve().parents[1] / 'scripts' / 'cloud'
sys.path.insert(0, str(CLOUD))
import run_horizon_diagnostic_257m as runner


class HorizonDiagnosticCampaignTest(TestCase):
    def test_matched_data_and_scheduler_with_hard_pilot_stop(self):
        for variant in runner.VARIANTS:
            args = runner.pilot_args('test', variant, Path('/tmp/metrics.jsonl'))
            value = lambda key: args[args.index(key) + 1]
            self.assertEqual(value('--iterations'), '39250')
            self.assertEqual(value('--early-stop-iteration'), '5000')
            self.assertEqual(value('--lr'), '5e-4')
            self.assertEqual(value('--weight-decay'), '1e-1')
            self.assertEqual(value('--fineweb-replay-layout'), 'serial')
            self.assertEqual(value('--model-parameter-dtype'), 'float32')
            self.assertEqual(value('--warmup-steps'), '2000')
            self.assertEqual(value('--wandb-group'), runner.GROUP)
            self.assertEqual(value('--latest-ckpt-interval'), '0')
            self.assertIn('--no-local-save', args)
            self.assertNotIn('--resume-from', args)
        args = runner.pilot_args('test', 'horizon16', Path('/tmp/metrics.jsonl'))
        self.assertEqual(args[args.index('--horizon-adam-bits') + 1], '16')
        args = runner.pilot_args('test', 'identity8', Path('/tmp/metrics.jsonl'))
        self.assertEqual(args[args.index('--horizon-adam-codec') + 1], 'identity')
        self.assertLessEqual(sum(runner.WALL_SECONDS.values()), 20 * 3600)

    def test_final_metric_requires_exact_step_budget_and_finite_loss(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.jsonl'
            row = dict(iter=5000, consumed_tokens=655360000, **{'val/loss': 3.5})
            path.write_text(json.dumps(row) + '\n')
            self.assertEqual(runner.verify_metrics(path, 5000)['step'], 5000)
            for change in (dict(iter=4999), dict(consumed_tokens=1), {'val/loss': float('nan')}):
                path.write_text(json.dumps({**row, **change}) + '\n')
                with self.assertRaises(RuntimeError):
                    runner.verify_metrics(path, 5000)

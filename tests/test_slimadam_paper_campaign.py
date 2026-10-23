import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from argparse import ArgumentParser
from config.base import parse_args


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('slimadam_paper_campaign', ROOT / 'scripts/cloud/run_slimadam_paper.py')
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


class SlimAdamPaperCampaignTest(TestCase):
    def test_all_four_commands_preserve_budget_and_compression(self):
        for lr in ('2e-3', '1e-3', '5e-4', '1e-4'):
            command, metrics, name = CAMPAIGN.build_command(lr, Path('/tmp/results'), '20261023')
            args = parse_args(ArgumentParser(), command[1:], None)
            self.assertEqual(args.opt, 'slim_adam')
            self.assertEqual(args.lr, float(lr))
            self.assertEqual(args.weight_decay, 0.1)
            self.assertEqual((args.beta1, args.beta2), (0.9, 0.99))
            self.assertEqual((args.n_layer, args.n_embd, args.n_head), (12, 1024, 8))
            self.assertEqual(args.batch_size * args.acc_steps, 128)
            self.assertEqual(args.iterations * 128 * args.sequence_length, 5144576000)
            self.assertEqual((args.warmup_steps, args.scheduler, args.wsd_fract_decay), (2000, 'wsd', 0.1))
            self.assertEqual(args.fineweb_replay_world_size, 1)
            self.assertEqual(args.fineweb_replay_layout, 'concat')
            self.assertEqual(Path(args.slim_adam_layer_map).name, 'layer_map_llama_paper.json')
            self.assertIsNone(args.slim_adam_rules_json)
            self.assertFalse(args.fp8_optim)
            self.assertFalse(args.fp8)
            self.assertTrue(args.no_local_save)
            self.assertEqual(args.latest_ckpt_interval, 0)
            self.assertEqual(args.permanent_ckpt_interval, 0)
            self.assertEqual(metrics.parent.name, name)

    def test_reject_unrequested_lr(self):
        with self.assertRaises(ValueError):
            CAMPAIGN.build_command('3e-3', Path('/tmp/results'), '20261023')

    def test_final_requires_native_metric_and_full_token_budget(self):
        with TemporaryDirectory() as directory:
            metrics = Path(directory) / 'metrics.jsonl'
            record = dict(event='validation', iter=39250, consumed_tokens=5144576000,
                          **{'final-val/loss': 3.0})
            metrics.write_text(json.dumps(record) + '\n')
            self.assertEqual(CAMPAIGN.verify_final(metrics), record)
            for invalid in (
                {**record, 'iter': 39200},
                {**record, 'consumed_tokens': 1},
                {**record, 'final-val/loss': float('nan')},
                dict(event='validation', iter=39250, consumed_tokens=5144576000,
                     **{'val/loss': 3.0}),
            ):
                metrics.write_text(json.dumps(invalid) + '\n')
                with self.assertRaises(RuntimeError):
                    CAMPAIGN.verify_final(metrics)

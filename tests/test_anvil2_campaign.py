"""Submission safety and the actual ANVIL II Cloud training command."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch


cloud = Path(__file__).resolve().parents[1] / 'scripts/cloud'
spec = importlib.util.spec_from_file_location('anvil2_controller', cloud / 'anvil2_controller.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class Anvil2CampaignTest(TestCase):
    def test_500m_submission_safety(self):
        with patch.object(sys, 'path', [str(cloud), *sys.path]):
            import anvil2_500m_wd_controller as wd_controller
        self.assertEqual(wd_controller.WDS, ('1e-1', '1e-2', '1e-3', '1e-4'))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(wd_controller, 'GATEWAY', root), patch.object(wd_controller, 'check_policy'), \
                 patch.object(wd_controller, 'command') as command:
                with self.assertRaises(ValueError):
                    wd_controller.submit('a' * 40, '0')
                (root / 'wd1e-1.json').write_text(json.dumps(dict(state='ambiguous_submission')))
                with self.assertRaises(RuntimeError):
                    wd_controller.submit('a' * 40, '1e-1')
                for age, free in ((601, 10_000_000_000), (0, 5_000_000_000)):
                    (root / 'preflight_verified.json').write_text(json.dumps(dict(
                        code_revision='a' * 40, timestamp=time.time() - age,
                        available_bytes={'nfs3': free})))
                    with self.assertRaises(RuntimeError):
                        wd_controller.submit('a' * 40, '1e-4')
                command.assert_not_called()

    def test_exact_six_authorized_rates(self):
        self.assertEqual(controller.LRS, ('1e-2', '5e-3', '2e-3', '1e-3', '5e-4', '1e-4'))
        with patch.object(controller, 'check_policy'):
            with self.assertRaises(ValueError):
                controller.submit('a' * 40, '0.1')

    def test_existing_ambiguous_receipt_is_never_resubmitted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'lr1e-2.json').write_text(json.dumps(dict(state='ambiguous_submission')))
            with patch.object(controller, 'GATEWAY', root), patch.object(controller, 'check_policy'), \
                 patch.object(controller, 'command') as command:
                with self.assertRaises(RuntimeError):
                    controller.submit('a' * 40, '1e-2')
                command.assert_not_called()

    def test_stale_or_low_space_preflight_blocks_submission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for age, free in ((601, 10_000_000_000), (0, 5_000_000_000)):
                (root / 'preflight_verified.json').write_text(json.dumps(dict(
                    code_revision='a' * 40, timestamp=time.time() - age,
                    available_bytes={'nfs3': free})))
                with patch.object(controller, 'GATEWAY', root), patch.object(controller, 'check_policy'), \
                     patch.object(controller, 'command') as command:
                    with self.assertRaises(RuntimeError):
                        controller.submit('a' * 40, '1e-2')
                    command.assert_not_called()

    def test_dry_run_timeout_does_not_expose_credentials(self):
        import subprocess

        with tempfile.TemporaryDirectory() as folder:
            key = 'test-private-token'
            error = subprocess.TimeoutExpired(['mlsub', '--env', 'WANDB_API_KEY=' + key], 40)
            with patch.object(controller, 'GATEWAY', Path(folder)), patch.object(controller, 'check_policy'), \
                 patch.object(controller.netrc, 'netrc', return_value=SimpleNamespace(
                     authenticators=lambda host: ('api', None, key))), \
                 patch.object(controller, 'command', side_effect=[dict(returncode=0, stdout=''), error]):
                with self.assertRaises(RuntimeError) as raised:
                    controller.submit('a' * 40)
                self.assertNotIn(key, str(raised.exception))
                self.assertTrue(raised.exception.__suppress_context__)
                self.assertFalse((Path(folder) / 'preflight.json').exists())

    def test_real_training_command_and_final_metadata(self):
        with patch.object(sys, 'path', [str(cloud), *sys.path]):
            import run_adaptive_muon as campaign
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def launch(command, **kwargs):
                args = command[command.index('src/main.py'):]
                def value(flag):
                    return args[args.index(flag) + 1]
                expected = {'--opt': 'anvil2', '--lr': '1e-2', '--weight-decay': '1e-1',
                            '--iterations': '39250', '--scheduler': 'wsd', '--warmup-steps': '2000',
                            '--wsd-fract-decay': '0.1', '--wsd-final-lr-scale': '0', '--decay-type': 'cosine',
                            '--batch-size': '16', '--acc-steps': '8', '--sequence-length': '1024',
                            '--n-layer': '12', '--n-embd': '1024', '--n-head': '8',
                            '--latest-ckpt-interval': '0', '--permanent-ckpt-interval': '0',
                            '--model-parameter-dtype': 'float32', '--anvil2-engage-fraction': '0.5'}
                for flag, expected_value in expected.items():
                    self.assertEqual(value(flag), expected_value)
                self.assertIn('--no-local-save', args)
                self.assertIn('--wandb', args)
                self.assertNotIn('--inter-ckpts', args)
                self.assertNotIn('--muon_adamw_lr_scale', args)
                self.assertEqual(kwargs['env']['WORLD_SIZE'], '2')
                self.assertEqual(kwargs['env']['WANDB_MODE'], 'online')
                metrics = Path(value('--metrics-jsonl'))
                metrics.parent.mkdir(parents=True, exist_ok=True)
                metrics.write_text(json.dumps(dict(event='validation', iter=39250,
                    consumed_tokens=5144576000, **{'final-val/loss': 3.0})) + '\n')
                return SimpleNamespace(returncode=0, poll=lambda: 0)
            with patch.object(campaign, 'ROOT', root), patch.object(campaign, 'MODE', 'matched'), \
                 patch.object(campaign, 'RANK', 0), patch.object(campaign, 'WORLD', 2), \
                 patch.object(campaign, 'TRAIN_WORLD', 2), patch.object(campaign, 'CAMPAIGN_DATE', '20261106'), \
                 patch.dict(os.environ, {'WANDB_MODE': 'online', 'CLOUD_DISABLE_CHECKPOINTS': '1'}), \
                 patch.object(campaign.subprocess, 'Popen', side_effect=launch):
                campaign.train('anvil2', '1e-2', 257, '1xC', 39250, weight_decay='1e-1')
            final = json.loads(next((root / 'finals').glob('*.json')).read_text())
            self.assertEqual(final['auxiliary_lr'], 0.01)
            self.assertEqual(final['engage_step'], 19625)
            self.assertEqual(final['ns_steps'], 6)
            self.assertFalse(final['checkpoint_saving'])

    def test_500m_weight_decay_training_command_and_final_metadata(self):
        with patch.object(sys, 'path', [str(cloud), *sys.path]):
            import run_adaptive_muon as campaign
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def launch(command, **kwargs):
                args = command[command.index('src/main.py'):]
                def value(flag):
                    return args[args.index(flag) + 1]
                expected = {'--opt': 'anvil2', '--lr': '5e-3', '--weight-decay': '1e-4',
                            '--iterations': '75457', '--scheduler': 'wsd', '--warmup-steps': '2000',
                            '--wsd-fract-decay': '0.1', '--wsd-final-lr-scale': '0', '--decay-type': 'cosine',
                            '--batch-size': '16', '--acc-steps': '8', '--sequence-length': '1024',
                            '--n-layer': '18', '--n-embd': '1280', '--n-head': '20',
                            '--latest-ckpt-interval': '0', '--permanent-ckpt-interval': '0',
                            '--model-parameter-dtype': 'float32', '--anvil2-engage-fraction': '0.5'}
                for flag, expected_value in expected.items():
                    self.assertEqual(value(flag), expected_value)
                self.assertIn('--no-local-save', args)
                self.assertIn('--wandb', args)
                self.assertNotIn('--inter-ckpts', args)
                self.assertNotIn('--muon_adamw_lr_scale', args)
                self.assertEqual(kwargs['env']['WORLD_SIZE'], '2')
                self.assertEqual(kwargs['env']['WANDB_MODE'], 'online')
                metrics = Path(value('--metrics-jsonl'))
                metrics.parent.mkdir(parents=True, exist_ok=True)
                metrics.write_text(json.dumps(dict(event='validation', iter=75457,
                    consumed_tokens=9890299904, **{'final-val/loss': 3.0})) + '\n')
                return SimpleNamespace(returncode=0, poll=lambda: 0)
            with patch.object(campaign, 'ROOT', root), patch.object(campaign, 'MODE', 'matched'), \
                 patch.object(campaign, 'RANK', 0), patch.object(campaign, 'WORLD', 2), \
                 patch.object(campaign, 'TRAIN_WORLD', 2), patch.object(campaign, 'CAMPAIGN_DATE', '20261106'), \
                 patch.dict(os.environ, {'WANDB_MODE': 'online', 'CLOUD_DISABLE_CHECKPOINTS': '1'}), \
                 patch.object(campaign.subprocess, 'Popen', side_effect=launch):
                campaign.train('anvil2', '5e-3', 500, '1xC', 75457, weight_decay='1e-4')
            final = json.loads(next((root / 'finals').glob('*.json')).read_text())
            self.assertEqual(final['auxiliary_lr'], 0.005)
            self.assertEqual(final['engage_step'], 37729)
            self.assertEqual(final['ns_steps'], 6)
            self.assertFalse(final['checkpoint_saving'])
            self.assertEqual(final['weight_decay'], 1e-4)
            self.assertEqual(final['auxiliary_weight_decay'], 1e-4)

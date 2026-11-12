"""Ablation launch, storage, and completed-sweep selection contracts."""

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
with patch.object(sys, 'path', [str(cloud), *sys.path]):
    import anvil2_no_norm_controller as controller
    import run_adaptive_muon as campaign


class Anvil2NoNormCampaignTest(TestCase):
    def test_actual_two_and_four_gpu_commands(self):
        for size, world, target, wd, batch in ((257, 2, 39250, '1e-1', 16),
                                               (500, 4, 75457, '1e-4', 8)):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                def launch(command, **kwargs):
                    def value(flag):return command[command.index(flag) + 1]
                    self.assertEqual(value('--batch-size'), str(batch))
                    self.assertEqual(value('--acc-steps'), str(4 * world))
                    self.assertEqual(value('--iterations'), str(target))
                    self.assertEqual(value('--weight-decay'), wd)
                    self.assertIn('--anvil2-no-normalization', command)
                    self.assertIn('--no-local-save', command)
                    self.assertNotIn('--inter-ckpts', command)
                    self.assertNotIn('--muon_adamw_lr_scale', command)
                    self.assertEqual(value('--scheduler'), 'wsd')
                    self.assertEqual(value('--warmup-steps'), '2000')
                    self.assertEqual(kwargs['env']['WORLD_SIZE'], str(world))
                    self.assertEqual(kwargs['env']['WANDB_MODE'], 'online')
                    self.assertIn('anvil2_no_norm_', kwargs['env']['WANDB_RUN_ID'])
                    metrics = Path(value('--metrics-jsonl'))
                    metrics.parent.mkdir(parents=True)
                    metrics.write_text(json.dumps(dict(event='validation', iter=target,
                        consumed_tokens=target * 128 * 1024, **{'final-val/loss': 3.0})) + '\n')
                    return SimpleNamespace(returncode=0, poll=lambda: 0)
                with patch.object(campaign, 'ROOT', root), patch.object(campaign, 'MODE', 'matched'), \
                     patch.object(campaign, 'RANK', 0), patch.object(campaign, 'WORLD', world), \
                     patch.object(campaign, 'TRAIN_WORLD', world), patch.object(campaign, 'CAMPAIGN_DATE', '20261112'), \
                     patch.dict(os.environ, {'WANDB_MODE': 'online', 'CLOUD_DISABLE_CHECKPOINTS': '1',
                                            'ANVIL2_NO_NORMALIZATION': '1'}), \
                     patch.object(campaign.subprocess, 'Popen', side_effect=launch):
                    campaign.train('anvil2', '5e-3', size, '1xC', target, weight_decay=wd)
                final = json.loads(next((root / 'finals').glob('*.json')).read_text())
                self.assertEqual(final['algorithm'], 'ANVIL_II_NO_NORM')
                self.assertFalse(final['matrix_normalization'])
                self.assertFalse(final['checkpoint_saving'])
                self.assertEqual(final['physical_world_size'], world)
                self.assertEqual(final['global_batch_size'], 128)
                self.assertEqual(final['engage_step'], (target + 1) // 2)

    def test_submission_blocks_stale_storage_and_ambiguous_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(controller, 'GATEWAY', root), patch.object(controller, 'check_policy'), \
                 patch.object(controller, 'command') as command:
                with self.assertRaises(ValueError):controller.submit('a' * 40, '257m_lr0', lr='0')
                (root / '257m_lr1e-2.json').write_text(json.dumps(dict(state='ambiguous_submission')))
                with self.assertRaises(RuntimeError):controller.submit('a' * 40, '257m_lr1e-2', lr='1e-2')
                for age, free in ((601, 10_000_000_000), (0, 5_000_000_000)):
                    (root / 'preflight257_verified.json').write_text(json.dumps(dict(
                        code_revision='a' * 40, timestamp=time.time() - age, available_bytes={'nfs3': free})))
                    with self.assertRaises(RuntimeError):controller.submit('a' * 40, '257m_lr1e-4', lr='1e-4')
                command.assert_not_called()

    def test_500m_rejects_unselected_lr_before_any_submission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'preflight500_verified.json').write_text(json.dumps(dict(
                code_revision='a' * 40, timestamp=time.time(), available_bytes={'nfs3': 10_000_000_000})))
            (root / 'selection.json').write_text(json.dumps(dict(code_revision='a' * 40, lr='5e-3')))
            with patch.object(controller, 'GATEWAY', root), patch.object(controller, 'check_policy'), \
                 patch.object(controller, 'command') as command:
                with self.assertRaises(RuntimeError):controller.submit('a' * 40, '500m', size=500, lr='1e-3')
                command.assert_not_called()

    def test_winner_requires_all_six_verified_final_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = dict(code_revision='a' * 40, events={'FINAL': []})
            records = [dict(size=257, lr=lr) for lr in controller.LRS]
            (root / 'latest_inspection.json').write_text(json.dumps(raw))
            losses = dict(zip(controller.LRS, (3.1, 3.0, 3.01, 3.03, 3.05, 3.2)))
            with patch.object(controller, 'GATEWAY', root), \
                 patch.object(controller, 'snapshot', return_value=dict(trials=records)), \
                 patch.object(controller, 'validate_final', side_effect=lambda r, raw: losses[r['lr']]):
                self.assertEqual(controller.select_winner('a' * 40)['lr'], '5e-3')
                records.pop()
                with self.assertRaises(RuntimeError):controller.select_winner('a' * 40)
            with self.assertRaises(RuntimeError):
                controller.validate_final(dict(size=257, lr='5e-3', status=dict(status='running')), raw)

    def test_final_validation_rejects_normalized_or_incomplete_artifacts(self):
        from argparse import ArgumentParser
        from copy import deepcopy
        from config.base import parse_args

        args = parse_args(ArgumentParser(), ['--experiment-name', 'test', '--opt', 'anvil2',
            '--lr', '5e-3', '--weight-decay', '0.1', '--iterations', '39250', '--n-layer', '12',
            '--n-embd', '1024', '--n-head', '8', '--batch-size', '16', '--acc-steps', '4',
            '--sequence-length', '1024', '--seed', '0', '--data-seed', '1337', '--dtype', 'bfloat16',
            '--scheduler', 'wsd', '--warmup-steps', '2000', '--wsd-fract-decay', '0.1',
            '--wsd-final-lr-scale', '0', '--decay-type', 'cosine', '--no-local-save',
            '--latest-ckpt-interval', '0', '--permanent-ckpt-interval', '0',
            '--anvil2-no-normalization'], None)
        args.inter_ckpts = []  # main normalizes the absent checkpoint list before W&B init.
        record = dict(size=257, lr='5e-3', status=dict(status='completed', error_code=0),
            wandb=dict(state='finished', config=vars(args), summary=dict(iter=39250,
                consumed_tokens=5144576000, **{'final-val/loss': 3.0})))
        final = dict(phase='1xC', size=257, lr='5e-3', iter=39250, consumed_tokens=5144576000,
            algorithm='ANVIL_II_NO_NORM', matrix_normalization=False, weight_decay=0.1,
            auxiliary_lr=0.005, auxiliary_weight_decay=0.1, physical_world_size=2,
            checkpoint_saving=False, engage_step=19625, ns_steps=6, **{'final-val/loss': 3.0})
        raw = dict(events={'FINAL': [final]})
        self.assertEqual(controller.validate_final(record, raw), 3.0)
        for key, value in (('matrix_normalization', True), ('consumed_tokens', 5144575999),
                           ('final-val/loss', 2.9)):
            broken = deepcopy(raw)
            broken['events']['FINAL'][0][key] = value
            with self.assertRaises(RuntimeError):controller.validate_final(record, broken)
        record['wandb']['config']['anvil2_no_normalization'] = False
        with self.assertRaises(RuntimeError):controller.validate_final(record, raw)

    def test_preflight_cannot_relabel_another_revision(self):
        with patch.object(controller, 'inspection', return_value=dict(ready=True, code_revision='b' * 40)):
            with self.assertRaises(RuntimeError):controller.verify_preflight(257, 'a' * 40)

    def test_existing_receipt_cannot_be_reused_for_another_recipe(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / '500m.json').write_text(json.dumps(dict(job_id='tracked-job', code_revision='a' * 40,
                size=500, lr='1e-3', preflight=False, label='500m')))
            with patch.object(controller, 'GATEWAY', root), patch.object(controller, 'check_policy'), \
                 patch.object(controller, 'command') as command:
                with self.assertRaises(RuntimeError):controller.submit('a' * 40, '500m', size=500, lr='5e-3')
                command.assert_not_called()

"""Per-period LR selection cannot mix p2/p5 or incomplete grid finals."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import TestCase
from unittest.mock import patch


class SlimLionCampaignTest(TestCase):
    def load(self, period, stage):
        cloud = Path(__file__).resolve().parents[1] / 'scripts/cloud'
        with patch.dict(os.environ, {'SLIM_LION_PERIOD': str(period), 'SLIM_LION_STAGE': stage}), patch.object(sys, 'path', [str(cloud), *sys.path]):
            spec = importlib.util.spec_from_file_location('lion_campaign_test', cloud / 'run_slim_normuon_lion.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return module

    def test_each_period_selects_its_own_complete_grid(self):
        for period in (2, 5):
            module = self.load(period, '500m')
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                with self.assertRaises(FileNotFoundError):
                    module.select_winner(root, period)
                for index, lr in enumerate(module.campaign.LRS):
                    record = dict(optimizer='slim_normuon_lion', size=257, phase='1xC', lr=lr,
                                  period=period, iter=39250, consumed_tokens=5144576000,
                                  weight_decay=0.1, auxiliary_lr=0.001, auxiliary_weight_decay=0.0,
                                  ns_steps=5, slim_update_rms=0.2, lion_lr_ratio=0.01,
                                  matrix_momentum=0.99, direction_beta=0.95, matrix_beta2=0.95,
                                  second_moment_clock='slim_steps_only', checkpoint_saving=False,
                                  physical_world_size=1, compression_map='layer_map_llama_paper.json',
                                  **{'final-val/loss': 3.0 + abs(index - (1 if period == 5 else 2))})
                    path = root / f'lr{lr}' / 'finals' / f'slim_normuon_lion_257m_bf16_lr{lr}_wd1e-1_1xC_1gpu_p{period}.json'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(record))
                winner = module.select_winner(root, period)
                self.assertEqual(winner['lr'], '5e-4' if period == 5 else '1e-3')
                with patch.dict(os.environ, {'SLIM_LION_SWEEP_ROOT': str(root)}), patch.object(module, 'preflight'), patch.object(module.campaign, 'train') as train:
                    module.main()
                self.assertEqual([c.args for c in train.call_args_list], [
                    ('slim_normuon_lion', winner['lr'], 500, 'matched_smoke', 75457),
                    ('slim_normuon_lion', winner['lr'], 500, '1xC', 75457)])
                self.assertTrue(all(c.kwargs == {'weight_decay': '1e-4'} for c in train.call_args_list))
                winner['period'] = 7
                with self.assertRaises(RuntimeError):
                    module.validate_final(winner, winner['lr'], period)

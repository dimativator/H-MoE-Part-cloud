import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import call, patch

from scripts.cloud import run_adaptive_muon as campaign


class CampaignVerificationTest(TestCase):
    def test_auxiliary_lr_257m_mode_uses_weight_decay_point_one(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'packed_metadata.json').write_text(json.dumps(dict(
                format='packed_fineweb_h200_v3', iterations=150914, world_size=2, batch_size=16)))
            with patch.object(campaign, 'ROOT', root), patch.object(campaign, 'DATA', root), \
                 patch.object(campaign, 'MODE', '257m_aux_lr'), patch.object(campaign, 'RECOVERY', None), \
                 patch.object(campaign.subprocess, 'run'), patch.object(campaign, 'train') as train:
                campaign.main()
            self.assertEqual(train.call_args_list, [
                call('normuon', '1e-3', 257, 'aux_smoke', 39250, weight_decay='1e-1'),
                call('normuon', '1e-3', 257, '1xC', 39250, weight_decay='1e-1'),
            ])

    def test_final_metric_uses_native_final_validation_key(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'metrics.jsonl'
            record = dict(event='validation', iter=39250, consumed_tokens=39250 * 128 * 1024)
            record['final-val/loss'] = 2.9
            path.write_text(json.dumps(record) + '\n')
            self.assertEqual(campaign.final_metric(path, 39250)['val_loss'], 2.9)
            record['consumed_tokens'] -= 1
            path.write_text(json.dumps(record) + '\n')
            with self.assertRaises(RuntimeError):
                campaign.final_metric(path, 39250)

    def test_selects_best_lr_only_after_four_completed_finals(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for lr, loss in zip(campaign.LRS, (3.1, 2.9, 2.8, 3.0)):
                path = root / 'finals' / f'muon2_257m_bf16_lr{lr}_wd1e-1_1xC_2gpu.json'
                campaign.atomic_json(path, dict(iter=39250, val_loss=loss, lr=lr))
            with patch.object(campaign, 'SOURCE_ROOT', root):
                self.assertEqual(campaign.best_lr('muon2'), '1e-3')
                path.write_text(json.dumps(dict(iter=39000, val_loss=2.7, lr='2e-3')))
                with self.assertRaises(RuntimeError):
                    campaign.best_lr('muon2')

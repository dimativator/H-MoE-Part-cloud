import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts.cloud import run_adaptive_muon as campaign


class CampaignVerificationTest(TestCase):
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
            with patch.object(campaign, 'ROOT', root):
                self.assertEqual(campaign.best_lr('muon2'), '1e-3')
                path.write_text(json.dumps(dict(iter=39000, val_loss=2.7, lr='2e-3')))
                with self.assertRaises(RuntimeError):
                    campaign.best_lr('muon2')

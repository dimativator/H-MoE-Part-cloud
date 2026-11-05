"""The follow-up must use all four verified final losses, never interim minima."""

import copy
import importlib.util
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import tempfile

path = Path(__file__).resolve().parents[1] / 'scripts/cloud/soft_lion_muon_controller.py'
spec = importlib.util.spec_from_file_location('controller', path)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def final_record(lr: str, loss: float) -> dict:
    config = dict(opt='soft_lion_muon', lr=float(lr), weight_decay=0.1,
        iterations=39250, seed=0, data_seed=1337, dtype='bfloat16', model_parameter_dtype='float32',
        batch_size=16, acc_steps=4, sequence_length=1024, no_local_save=True,
        latest_ckpt_interval=0, permanent_ckpt_interval=0, inter_ckpts=[], n_layer=12, n_embd=1024,
        soft_lion_muon_period=5, soft_lion_muon_beta1=0.95, soft_lion_muon_beta2=0.99,
        soft_lion_muon_lr_ratio=0.01, soft_lion_muon_start_fraction=0.9,
        soft_lion_muon_eps=1e-4, soft_lion_muon_ns_steps=6, soft_lion_muon_soft_ns_steps=30,
        soft_lion_muon_aux_lr=0.001)
    return dict(lr=lr, size=257, status=dict(status='completed', error_code=0),
        wandb=dict(state='finished', config=config,
            summary={'iter': 39250, 'consumed_tokens': 5144576000, 'final-val/loss': loss,
                     'val/loss': loss - 1}))


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.records = [final_record(lr, loss) for lr, loss in zip(controller.LRS, (3.1, 3.0, 3.2, 3.3))]

    def test_select_lowest_final_loss_and_require_all_four(self):
        self.assertEqual(controller.select_winner(self.records)['lr'], '1e-3')
        with self.assertRaises(RuntimeError):
            controller.select_winner(self.records[:-1])

    def test_reject_incomplete_failed_or_checkpointed_run(self):
        for key, value in (('no_local_save', False), ('iterations', 39249),
                           ('weight_decay', 1e-4), ('soft_lion_muon_period', 2)):
            records = copy.deepcopy(self.records)
            records[0]['wandb']['config'][key] = value
            with self.assertRaises(RuntimeError):
                controller.select_winner(records)
        for key, value in (('iter', 39249), ('consumed_tokens', 0), ('final-val/loss', float('nan'))):
            records = copy.deepcopy(self.records)
            records[0]['wandb']['summary'][key] = value
            with self.assertRaises(RuntimeError):
                controller.select_winner(records)
        records = copy.deepcopy(self.records)
        records[0]['status']['error_code'] = 1
        with self.assertRaises(ValueError):
            controller.select_winner(records)

    def test_500m_configuration_uses_four_gpus_and_changed_weight_decay(self):
        record = final_record('1e-3', 2.8)
        record['wandb']['config'].update(weight_decay=1e-4, iterations=75457,
                                        n_layer=18, n_embd=1280, batch_size=8)
        record['wandb']['summary'].update(iter=75457, consumed_tokens=75457 * 128 * 1024)
        self.assertEqual(controller.validate_final(record, 500, '1e-3'), 2.8)
        self.assertIn('wd1e-4_1xC_4gpu', controller.run_name(500, '1e-3'))

    def test_inventory_timeout_does_not_create_a_submission_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'preflight257_verified.json').write_text(json.dumps(dict(
                timestamp=time.time(), code_revision='a' * 40,
                available_bytes={'output': 10_000_000_000})))
            with patch.object(controller, 'GATEWAY', root), \
                 patch.object(controller, 'check_policy'), \
                 patch.object(controller, 'command', return_value={'error': 'timeout'}):
                with self.assertRaises(ConnectionError):
                    controller.submit('257m_lr2e-3', 'a' * 40, lr='2e-3')
            self.assertFalse((root / '257m_lr2e-3.json').exists())


if __name__ == '__main__':
    unittest.main()

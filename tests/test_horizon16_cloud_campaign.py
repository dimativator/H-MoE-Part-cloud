from argparse import ArgumentParser
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from types import SimpleNamespace

import torch

from config.base import parse_args
from scripts.cloud.run_horizon_adam16_257m import GROUP, LRS, SCIENCE, TARGET, storage_report, train_args, validate_resume


class HorizonCloudCampaignTest(TestCase):
    def test_checkpoint_budget_counts_remaining_space_not_full_campaign_twice(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            disk = SimpleNamespace(total=100_000_000_000, used=91_310_000_000, free=20_000_000_000)
            with patch('scripts.cloud.run_horizon_adam16_257m.ROOT', root), patch('shutil.disk_usage', return_value=disk):
                self.assertLess(storage_report()['required_free_bytes'], disk.free)
                run = root / GROUP / 'horizon_adam_257m_b16_h32_lr1e-4_wd1e-1_s0_attempt1'
                checkpoint = run / 'ckpts' / 'latest'
                checkpoint.mkdir(parents=True)
                (run / 'campaign.json').write_text(json.dumps(dict(attempt=1)))
                with (checkpoint / 'main.pt').open('wb') as stream:
                    stream.truncate(1_300_000_000)
                disk.free -= 1_300_000_000
                self.assertLess(storage_report()['required_free_bytes'], disk.free)
                disk.free = 100_000_000
                with self.assertRaisesRegex(RuntimeError, 'Insufficient space'):
                    storage_report()

    def test_all_lr_commands_preserve_scientific_settings(self):
        for lr in LRS:
            for smoke in (True, False):
                command = train_args('test', lr, Path('/tmp/test/metrics.jsonl'), smoke, None)
                args = parse_args(ArgumentParser(), command[1:], None)
                self.assertEqual(args.lr, float(lr))
                self.assertEqual(args.weight_decay, 0.1)
                self.assertEqual(args.iterations, TARGET)
                self.assertEqual(args.batch_size * args.acc_steps, 128)
                self.assertEqual(args.fineweb_replay_world_size, 2)
                self.assertEqual(args.fineweb_replay_layout, 'serial')
                self.assertEqual(args.horizon_adam_horizon, 32)
                self.assertEqual(args.horizon_adam_bits, 16)
                self.assertEqual(args.horizon_adam_codec, 'learned')
                self.assertEqual(args.model_parameter_dtype, 'float32')
                self.assertEqual(args.iterations * args.batch_size * args.acc_steps * args.sequence_length, 5144576000)
                self.assertEqual(args.latest_ckpt_interval, 0 if smoke else 2000)

    def test_checkpoint_consistency_before_resume(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = run / 'ckpts' / 'latest'
            checkpoint.mkdir(parents=True)
            (run / 'campaign.json').write_text(json.dumps(dict(science=SCIENCE, lr='1e-3')))
            main = dict(itr=2000, scheduler={'last_epoch': 2000},
                        optimizer={'state': {0: {'codec_version': 1}}})
            worker = dict(train_reader_state=dict(reader_type='fineweb_serial_replay_train_reader_v1',
                          batch_size=16, sequence_length=1024, source_world_size=2, step=16000,
                          source_states=[dict(rank=rank, step=8000) for rank in (0, 1)]),
                          rng_python=(), rng_np=(), rng_torch_cpu=torch.zeros(1), rng_torch_gpu=torch.zeros(1))
            torch.save(main, checkpoint / 'main.pt')
            torch.save(worker, checkpoint / 'worker_0.pt')
            self.assertEqual(validate_resume(checkpoint, '1e-3')['iteration'], 2000)
            inconsistent = deepcopy(worker)
            inconsistent['train_reader_state']['step'] = 8000
            torch.save(inconsistent, checkpoint / 'worker_0.pt')
            with self.assertRaisesRegex(RuntimeError, 'Model/data checkpoint mismatch'):
                validate_resume(checkpoint, '1e-3')
            inconsistent = deepcopy(worker)
            inconsistent['train_reader_state']['source_states'][1]['step'] -= 1
            torch.save(inconsistent, checkpoint / 'worker_0.pt')
            with self.assertRaisesRegex(RuntimeError, 'source-reader positions'):
                validate_resume(checkpoint, '1e-3')
            torch.save(worker, checkpoint / 'worker_0.pt')
            with self.assertRaisesRegex(RuntimeError, 'scientific configuration'):
                validate_resume(checkpoint, '2e-3')
            incomplete = deepcopy(worker)
            del incomplete['rng_torch_gpu']
            torch.save(incomplete, checkpoint / 'worker_0.pt')
            with self.assertRaisesRegex(RuntimeError, 'complete RNG'):
                validate_resume(checkpoint, '1e-3')

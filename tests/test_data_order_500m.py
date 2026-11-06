"""The distributed shuffle must change order without changing the token population."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from src.data.fineweb_replay import PermutedPackedFineWebTrainReader


class Source:
    batch_size = 2
    sequence_length = 2
    _num_steps = 19

    def __init__(self, rank: int):
        self.rank = rank

    def set_step(self, step: int) -> None:
        self.step = step

    def sample_batch(self):
        x = torch.tensor([[self.rank * 1000 + self.step * 2, 7],
                          [self.rank * 1000 + self.step * 2 + 1, 8]])
        return x, x + 1


class DataOrderTest(unittest.TestCase):
    def test_ddp_exact_coverage_and_paired_rank_order(self):
        orders = []
        for seed in (1338, 1339):
            readers = [PermutedPackedFineWebTrainReader(Source(rank), steps=13, seed=seed)
                       for rank in range(2)]
            np.testing.assert_array_equal(readers[0].order, readers[1].order)
            orders.append(readers[0].order.copy())
            batches = [[reader.sample_batch()[0][:, 0].tolist() for _ in range(13)]
                       for reader in readers]
            for rank, values in enumerate(batches):
                self.assertEqual(sorted(value for batch in values for value in batch),
                                 list(range(rank * 1000, rank * 1000 + 26)))
            self.assertEqual([[value - 1000 for value in batch] for batch in batches[1]], batches[0])
            with self.assertRaises(RuntimeError):
                readers[0].sample_batch()
        self.assertFalse(np.array_equal(*orders))

    def test_seek_and_rng_independence(self):
        np.random.seed(42)
        expected = np.random.random(4)
        np.random.seed(42)
        reader = PermutedPackedFineWebTrainReader(Source(0), steps=13, seed=1338)
        np.testing.assert_array_equal(np.random.random(4), expected)
        batch = reader.sample_batch()
        reader.set_step(0)
        torch.testing.assert_close(reader.sample_batch(), batch)
        with self.assertRaises(ValueError):
            PermutedPackedFineWebTrainReader(Source(0), steps=20, seed=1338)

    def test_authorized_launch_matrix(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/cloud/run_data_order_500m.py'
        spec = importlib.util.spec_from_file_location('order_launcher', path)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        self.assertEqual(len(launcher.TRIALS), 8)
        for trial, record in launcher.TRIALS.items():
            command = launcher.arguments(trial)
            self.assertIn('--fineweb-packed-shuffle-steps', command)
            self.assertIn('--no-local-save', command)
            self.assertNotIn('--resume-from', command)
            self.assertNotIn('--fp8', command)
            self.assertEqual(command[command.index('--seed') + 1], '0')
            self.assertEqual(command[command.index('--data-seed') + 1], trial.rsplit('_data', 1)[1])
            self.assertEqual(command[command.index('--latest-ckpt-interval') + 1], '0')
            self.assertEqual(command[command.index('--permanent-ckpt-interval') + 1], '0')
            self.assertEqual(record[-2:], (75457, 2))
            if trial.startswith('slimadam_'):
                self.assertEqual(command[command.index('--weight-decay') + 1], '1e-4')
                self.assertEqual(command[command.index('--warmup-steps') + 1], '2000')


if __name__ == '__main__':
    unittest.main()

"""Verify that incomplete or incompatible sweeps cannot trigger the follow-up."""

import json
from pathlib import Path
import tempfile
import unittest

from softmuon_selection import select


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.lrs = ('1e-4', '5e-4', '1e-3', '2e-3', '3e-3', '4e-3', '5e-3', '6e-3')
        self.root = Path(self.temp.name)
        self.sweeps = ((str(self.root), self.lrs),)
        self.paths = []
        for index, lr in enumerate(self.lrs):
            record = dict(optimizer='softmuon', algorithm='Algorithm3', lr=lr, size=257,
                          phase='1xC', iter=39250, consumed_tokens=5144576000,
                          weight_decay=0.1, auxiliary_weight_decay=0.1, auxiliary_lr=0.001,
                          physical_world_size=2, global_batch_size=128, checkpoint_saving=False,
                          seed=0, data_seed=1337, update_scale='original_shape_factor',
                          update_rms=None, nesterov=False, temperature_spectrum='updated_momentum',
                          sign_iters=35325, transition_iters=3925, **{'final-val/loss': 4-index/10})
            path = self.root / f'lr{lr}/finals/softmuon_257m_bf16_lr{lr}_wd1e-1_1xC_2gpu.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(record))
            self.paths.append(path)

    def test_selects_across_all_eight(self):
        self.assertEqual(select(self.sweeps)['best_matrix_lr'], '6e-3')

    def test_incomplete_sweep_is_rejected(self):
        self.paths[-1].unlink()
        with self.assertRaises(FileNotFoundError):
            select(self.sweeps)

    def test_incompatible_or_nonfinite_final_is_rejected(self):
        path = self.paths[-1]
        original = json.loads(path.read_text())
        for change in ({'iter': 39000}, {'checkpoint_saving': True}, {'auxiliary_lr': 0.006},
                       {'final-val/loss': float('nan')}, {'smoke': True}):
            with self.subTest(change=change):
                path.write_text(json.dumps({**original, **change}))
                with self.assertRaises(ValueError):
                    select(self.sweeps)


if __name__ == '__main__':
    unittest.main()

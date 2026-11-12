"""The cosine comparison uses one shared schedule and verified full finals."""

from types import SimpleNamespace
import unittest

import torch

from optim.utils import build_scheduler


class CosineScheduleTest(unittest.TestCase):
    def test_matrix_and_auxiliary_reach_zero_together(self):
        p, q = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))
        opt = torch.optim.SGD([dict(params=[p], lr=0.002), dict(params=[q], lr=0.001)])
        args = SimpleNamespace(iterations=39250, warmup_steps=2000,
                               scheduler='cos_warmup_zero')
        scheduler = build_scheduler(opt, args)
        observed = {}
        for step in range(args.iterations + 1):
            if step in (0, 2000, 20625, 39250):
                observed[step] = [g['lr'] for g in opt.param_groups]
            if step < args.iterations:
                opt.step()
                scheduler.step()
        self.assertEqual(observed[0], [0.0, 0.0])
        self.assertEqual(observed[2000], [0.002, 0.001])
        self.assertAlmostEqual(observed[20625][0], 0.001)
        self.assertAlmostEqual(observed[20625][1], 0.0005)
        self.assertEqual(observed[39250], [0.0, 0.0])


class FinalBudgetTest(unittest.TestCase):
    def test_partial_or_wrong_scheduler_cannot_win(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location('cosine_controller',
            Path(__file__).resolve().parents[1] / 'scripts/cloud/cosine_257m_controller.py')
        controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller)
        record = dict(optimizer='muon_reference', lr='1e-3',
            status=dict(status='Completed', error_code=0),
            wandb=dict(state='finished', summary=dict(iter=39250,
                       consumed_tokens=5144576000, **{'final-val/loss': 3.0}), config={}))
        with self.assertRaises(RuntimeError):
            controller.validate_final(record)
        record['status']['status'] = 'Running'
        with self.assertRaises(ValueError):
            controller.validate_final(record)


if __name__ == '__main__':
    unittest.main()

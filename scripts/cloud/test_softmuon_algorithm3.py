"""Regression tests for momentum calibration and non-Nesterov updates."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from third_party.softsign import softmuon


class Algorithm3Test(unittest.TestCase):
    def test_transition_uses_updated_momentum_once(self):
        p = torch.nn.Parameter(torch.ones(2, 3))
        opt = softmuon.SingleDeviceSoftMuonWithAuxAdam([
            dict(params=[p], use_muon=True, lr=0.01, momentum=0.95,
                 sign_iters=1, transition_iters=3),
        ])
        identity = lambda matrix, **kwargs: matrix.clone()
        with patch.object(softmuon, 'zeropower_via_newtonschulz5', identity):
            p.grad = torch.full_like(p, 2.0)
            opt.step()
        torch.testing.assert_close(opt.state[p]['momentum_buffer'],
                                   torch.full_like(p, 0.1))
        before = p.detach().clone()
        old_momentum = opt.state[p]['momentum_buffer'].clone()
        p.grad = torch.full_like(p, 8.0)
        expected = old_momentum.lerp(p.grad, 0.05)
        captured = []

        def svd(matrix):
            captured.append(matrix.clone())
            return torch.linalg.svd(matrix, full_matrices=False)

        with patch.object(softmuon, 'cans_svd', svd), \
             patch.object(softmuon, 'muon_regularized_update', identity):
            opt.step()
        self.assertEqual(len(captured), 1)
        torch.testing.assert_close(captured[0], expected)
        torch.testing.assert_close(opt.state[p]['momentum_buffer'], expected)
        torch.testing.assert_close(p, before - 0.01 * expected)
        torch.testing.assert_close(p.grad, torch.full_like(p, 8.0))
        with patch.object(softmuon, 'cans_svd') as svd_mock, \
             patch.object(softmuon, 'muon_regularized_update', identity):
            p.grad = torch.ones_like(p)
            opt.step()
        svd_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()

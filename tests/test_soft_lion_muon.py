"""Numerical references and ordering regressions for SoftLionMuon."""

import copy
import math
import unittest
from unittest.mock import patch

import torch

from optim.soft_lion_muon import (
    SoftLionMuon, elementwise_softsign, magnitude_quantile, spectral_softsign, temperature,
)


class SoftLionMuonTest(unittest.TestCase):
    def test_spectral_map_against_svd(self):
        torch.manual_seed(17)
        for dtype in (torch.float32, torch.float64):
            for shape in ((4, 9), (9, 4), (5, 5)):
                left, _ = torch.linalg.qr(torch.randn(shape[0], min(shape), dtype=dtype))
                right, _ = torch.linalg.qr(torch.randn(shape[1], min(shape), dtype=dtype))
                values = torch.logspace(-4, 0, min(shape), dtype=dtype)
                matrix = (left * values) @ right.mT
                for tau in (1.0, 20.0, 1e3, 1e6):
                    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
                    expected = (u * elementwise_softsign(s, tau)) @ vh
                    actual = spectral_softsign(matrix, tau)
                    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(spectral_softsign(torch.zeros(3, 7), 1e9), torch.zeros(3, 7))

    def test_quantile_and_algebraic_saturation(self):
        for stats in ((0.0, 2.0), (3.0, 0.1), (1e-7, 1e-8)):
            taus = []
            for progress in (0.0, 0.1, 0.5, 0.99):
                q = magnitude_quantile(progress, stats)
                mu, scale = stats
                cdf = (math.atan((q - mu) / scale) + math.atan((q + mu) / scale)) / math.pi
                self.assertAlmostEqual(cdf, progress, places=10)
                tau = temperature(stats, progress, 1e-4)
                taus.append(tau)
                if math.isfinite(tau) and 1 < tau < 1e9:
                    self.assertAlmostEqual(float(elementwise_softsign(torch.tensor(q, dtype=torch.float64), tau)),
                                           1 - 1e-4, places=10)
            self.assertEqual(taus, sorted(taus, reverse=True))
        self.assertTrue(math.isinf(temperature((0.0, 0.0), 0.5, 1e-4)))

    def test_lion_ordering_calibration_and_single_momentum_update(self):
        p = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float64))
        opt = SoftLionMuon([p], total_steps=5, start_fraction=0.2, period=5,
                          lr=0.02, weight_decay=0.1)
        p.grad = torch.full_like(p, 2)
        with patch('optim.soft_lion_muon.zeropower_via_newtonschulz5', lambda x, steps: x):
            opt.step()
        expected_m = torch.full_like(p, 0.02)
        torch.testing.assert_close(opt.state[p]['momentum_buffer'], expected_m)
        before = p.detach().clone()
        p.grad = torch.tensor([[-3., -2., -1.], [1., 2., 4.]], dtype=p.dtype)
        expected_h = 0.95 * expected_m + 0.05 * p.grad
        captured = []
        original = torch.linalg.svdvals
        with patch('optim.soft_lion_muon.torch.linalg.svdvals',
                   lambda x: (captured.append(x.clone()), original(x))[1]):
            opt.step()
        self.assertEqual(len(captured), 1)
        torch.testing.assert_close(captured[0], expected_h)
        eta = 0.02 * 0.01
        torch.testing.assert_close(p, (1 - eta * 0.1) * before - eta * expected_h.sign())
        torch.testing.assert_close(opt.state[p]['momentum_buffer'], 0.99 * expected_m + 0.01 * p.grad)
        with patch('optim.soft_lion_muon.torch.linalg.svdvals', side_effect=AssertionError('Refitted')):
            opt.step()

    def test_soft_steps_and_resume_match_exact_reference(self):
        torch.manual_seed(5)
        p = torch.nn.Parameter(torch.randn(4, 7, dtype=torch.float64))
        opt = SoftLionMuon([p], total_steps=8, start_fraction=0, period=2)
        opt.statistics = None
        for step in range(8):
            p.grad = torch.randn_like(p)
            before = p.detach().clone()
            old_m = opt.state[p].get('momentum_buffer', torch.zeros_like(p)).clone()
            h = 0.95 * old_m + 0.05 * p.grad
            opt.step()
            if step:
                tau = opt.last_temperatures[1 if step % 2 == 0 else 0]
                if step % 2 == 0:
                    u, s, vh = torch.linalg.svd(h, full_matrices=False)
                    d = (u * elementwise_softsign(s, tau)) @ vh * (0.2 * math.sqrt(7))
                    eta = 1e-3
                else:
                    d, eta = elementwise_softsign(h, tau), 1e-5
                torch.testing.assert_close(p, (1 - eta * 0.1) * before - eta * d, atol=1e-8, rtol=1e-7)
            torch.testing.assert_close(opt.state[p]['momentum_buffer'], 0.99 * old_m + 0.01 * p.grad)
            if step == 3:
                clone = torch.nn.Parameter(p.detach().clone())
                resumed = SoftLionMuon([clone], total_steps=8, start_fraction=0, period=2)
                resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
            elif step > 3:
                clone.grad = p.grad.clone()
                resumed.step()
                torch.testing.assert_close(clone, p, atol=0, rtol=0)

    def test_auxiliary_matches_adamw_and_bf16_states_remain_fp32(self):
        p = torch.nn.Parameter(torch.ones(3))
        other = torch.nn.Parameter(p.detach().clone())
        opt = SoftLionMuon([dict(params=[p], use_soft_lion_muon=False)], total_steps=5)
        reference = torch.optim.AdamW([other], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        for _ in range(3):
            p.grad = other.grad = torch.tensor([1., -2., 3.])
            opt.step()
            reference.step()
            torch.testing.assert_close(p, other)
        matrix = torch.nn.Parameter(torch.ones(3, 4, dtype=torch.bfloat16))
        low = SoftLionMuon([matrix], total_steps=5)
        matrix.grad = torch.ones_like(matrix)
        low.step()
        self.assertEqual(low.state[matrix]['momentum_buffer'].dtype, torch.float32)


if __name__ == '__main__':
    unittest.main()

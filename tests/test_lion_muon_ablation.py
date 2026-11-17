import os
import unittest
from unittest.mock import patch

import torch

from optim.lion_muon_ablation import LionMuonAblation
from third_party.lion_muon.lion_muon import LionMuon


class AblationTest(unittest.TestCase):
    def optimizer(self, variant, weight=None):
        weight = torch.nn.Parameter(torch.zeros(7, 5)) if weight is None else weight
        return LionMuonAblation([weight], variant=variant, total_steps=101,
                               lr=0.001, lion_lr=0.00001), weight

    def test_normalized_magnitude_zero_and_rectangular_matrices(self):
        for variant in ('rms', 'neuron_half', 'neuron_full'):
            opt, _ = self.optimizer(variant)
            for shape in ((9, 5), (5, 9)):
                state = {}
                for _ in range(3):
                    update = opt._muon_update(torch.randn(shape), state, opt.param_groups[0])
                    self.assertAlmostEqual(update.square().mean().sqrt().item(), 0.2, places=5)
                    self.assertTrue(torch.isfinite(update).all())
                if variant != 'rms':
                    self.assertEqual(state['neuron_variance'].shape, (shape[0], 1))
                zero = opt._muon_update(torch.zeros(shape), {}, opt.param_groups[0])
                self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))

    def test_fractional_power_and_original_row_orientation(self):
        q = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        for variant, alpha in (('neuron_half', 0.5), ('neuron_full', 1.0)):
            opt, _ = self.optimizer(variant)
            with patch('optim.lion_muon_ablation.zeropower_via_newtonschulz5', return_value=q):
                state = {}
                actual = opt._muon_update(q, state, opt.param_groups[0])
            variance = 0.05 * q.square().mean(1, keepdim=True)
            expected = q / (variance + 1e-8).pow(alpha / 2)
            expected *= 0.2 / (expected.square().mean().sqrt() + 1e-8)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(state['neuron_variance'], variance)

    def test_probability_schedule_endpoints_and_rank_agreement(self):
        first, _ = self.optimizer('probability_down')
        second, _ = self.optimizer('probability_down')
        for step in range(101):
            first._step_count = second._step_count = step
            self.assertEqual(first._is_muon_step(), second._is_muon_step())
            self.assertAlmostEqual(first.last_probability, 0.35 - 0.3 * step / 100)
        direction = torch.randn(7, 5)
        original = LionMuon([torch.nn.Parameter(torch.zeros(7, 5))])
        torch.testing.assert_close(first._muon_update(direction, {}, first.param_groups[0]),
                                   original._muon_update(direction, {}, original.param_groups[0]), rtol=0, atol=0)

    def test_sign_momentum_auxiliary_and_period_unchanged(self):
        for variant in ('rms', 'neuron_half', 'neuron_full'):
            opt, p = self.optimizer(variant)
            base_p = torch.nn.Parameter(p.detach().clone())
            base = LionMuon([base_p], lr=0.001, lion_lr=0.00001)
            for step in range(11):
                gradient = torch.randn_like(p)
                p.grad = gradient.clone()
                base_p.grad = gradient.clone()
                before, base_before = p.detach().clone(), base_p.detach().clone()
                opt.step()
                base.step()
                self.assertEqual(opt.last_muon, step % 5 == 0)
                torch.testing.assert_close(opt.state[p]['exp_avg'], base.state[base_p]['exp_avg'], rtol=0, atol=0)
                if step % 5:
                    torch.testing.assert_close(p-before, base_p-base_before, rtol=1e-4, atol=1e-8)
            self.assertEqual(opt.param_groups[0]['lion_lr_ratio'], 0.01)

    def test_launcher_budget_and_isolated_variants(self):
        from scripts.cloud.run_lionmuon_normalization import arguments, TRIALS
        for variant in TRIALS:
            command = arguments(variant, 'lion_muon', 75457, 7546, False)
            get = lambda flag: command[command.index(flag) + 1]
            self.assertEqual(get('--lion-ablation'), variant)
            self.assertEqual(int(get('--batch-size')) * int(get('--acc-steps')), 128)
            self.assertEqual(int(get('--iterations')) * 128 * 1024, 9890299904)
            self.assertEqual(get('--latest-ckpt-interval'), '0')
            self.assertEqual(get('--permanent-ckpt-interval'), '0')
            self.assertIn('--no-local-save', command)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

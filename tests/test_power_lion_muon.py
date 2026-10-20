"""Iterative powers against SVD references, schedules, state restore and CPU DDP."""
import argparse
import copy
import math
import sys
import tempfile
import unittest
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from optim.power_lion_muon import (
    PowerLionMuon,
    elementwise_power,
    power_exponent,
    spectral_power,
)


def make_optimizer(param: torch.Tensor, **kwargs) -> PowerLionMuon:
    return PowerLionMuon([{'params': [param], 'use_power': True}], **dict(total_steps=20, **kwargs))


def ddp_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group('gloo', rank=rank, world_size=2,
                            init_method=rendezvous, timeout=timedelta(seconds=60))
    try:
        param = torch.nn.Parameter(torch.eye(3))
        # Different RNG seeds exercise rank0's authoritative broadcast.
        opt = make_optimizer(param, probability_seed=rank, start_fraction=0.0)
        choices = []
        for _ in range(20):
            param.grad = torch.arange(9, dtype=torch.float32).reshape(3, 3) / 10
            opt.step()
            choices.append(opt.last_is_spectral)
            copies = [torch.empty_like(param) for _ in range(2)]
            dist.all_gather(copies, param)
            torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
        decisions = torch.tensor(choices, dtype=torch.int32)
        copies = [torch.empty_like(decisions) for _ in range(2)]
        dist.all_gather(copies, decisions)
        assert torch.equal(copies[0], copies[1])
    finally:
        dist.destroy_process_group()


class PowerLionMuonTest(unittest.TestCase):
    def test_power_definitions(self):
        matrix = torch.tensor([[4., -3., 0.], [2., 5., -1.]], dtype=torch.float64)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        for a in (0., 0.25, 0.5, 1., 1.5):
            torch.testing.assert_close(spectral_power(matrix, a), u @ torch.diag(s.pow(a)) @ vh)
            torch.testing.assert_close(elementwise_power(matrix, a), matrix.sign() * matrix.abs().pow(a))
        torch.testing.assert_close(spectral_power(matrix, 0), u @ vh)
        torch.testing.assert_close(elementwise_power(matrix, 0), matrix.sign())

    def test_zero_and_rank_deficient_inputs(self):
        for a in (0., 0.5, 1., 2.):
            for matrix in (torch.zeros(2, 3), torch.diag(torch.tensor([4., 0.]))):
                spectral = spectral_power(matrix, a)
                elementwise = elementwise_power(matrix, a)
                self.assertTrue(torch.isfinite(spectral).all())
                torch.testing.assert_close(spectral, elementwise)

    def test_schedule(self):
        values = [power_exponent(i, 101, 0.9) for i in range(101)]
        self.assertTrue(all(a == 0 for a in values[:91]))
        self.assertEqual(values[-1], 1)
        self.assertTrue(all(a <= b for a, b in pairwise(values)))
        self.assertAlmostEqual(power_exponent(50, 101, 0, gamma=2), 0.25)
        self.assertEqual(power_exponent(100, 101, 0, start=0.2, end=1.5), 1.5)
        for args in ((0, 1), (101, 101), (-1, 101)):
            with self.assertRaises(ValueError):
                power_exponent(*args)

    def test_invalid_configuration(self):
        for kwargs in ({'probability': -0.1}, {'probability': float('nan')},
                       {'momentum': 1}, {'start_fraction': 1}, {'exponent_start': -1},
                       {'exponent_end': -1}, {'exponent_gamma': 0}, {'lr': -1},
                       {'exponent_end': float('inf')}, {'weight_decay': float('nan')},
                       {'ns_steps': 0}, {'polynomial_terms': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_optimizer(torch.nn.Parameter(torch.ones(2, 2)), **kwargs)

    def test_exact_updated_momentum_once(self):
        for probability in (0., 1.):
            param = torch.nn.Parameter(torch.eye(2, dtype=torch.float64))
            opt = make_optimizer(param, probability=probability, momentum=0.75,
                                 exponent_start=0.5, exponent_end=0.5, lr=0.01, weight_decay=0.1)
            momentum = torch.zeros_like(param)
            for step in range(3):
                grad = torch.tensor([[2., -1.], [3., 4.]], dtype=param.dtype) + step
                param.grad = grad
                before = param.detach().clone()
                momentum = 0.75 * momentum + 0.25 * grad
                transform = spectral_power if probability else elementwise_power
                expected = before * 0.999 - 0.01 * transform(momentum, 0.5)
                opt.step()
                torch.testing.assert_close(param, expected)
                torch.testing.assert_close(opt.state[param]['momentum_buffer'], momentum)

    def test_a_one_identical_branches(self):
        params = [torch.nn.Parameter(torch.eye(2)) for _ in range(2)]
        opts = [make_optimizer(p, probability=probability, exponent_start=1, exponent_end=1)
                for p, probability in zip(params, (0., 1.))]
        for step in range(20):
            for p, opt in zip(params, opts):
                p.grad = torch.full_like(p, step + 1.)
                opt.step()
            torch.testing.assert_close(params[0], params[1], rtol=0, atol=0)

    def test_autocast_does_not_reduce_iteration_precision(self):
        matrix = torch.tensor([[1., 2.], [-3., 4.]])
        reference = spectral_power(matrix, 0.5)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = spectral_power(matrix, 0.5)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_spectral_accuracy_over_conditioned_spectra(self):
        generator = torch.Generator().manual_seed(7)
        for dtype in (torch.float64, torch.float32):
            u = torch.linalg.qr(torch.randn(16, 8, dtype=dtype, generator=generator)).Q
            v = torch.linalg.qr(torch.randn(8, 8, dtype=dtype, generator=generator)).Q
            for condition in (1e2, 1e4):
                singular_values = torch.logspace(0, -math.log10(condition), 8, dtype=dtype)
                matrix = (u * singular_values) @ v.mT
                for a in (0., 0.1, 0.5, 0.9, 1., 1.5, 2.):
                    reference = (u * singular_values.pow(a)) @ v.mT
                    for candidate, expected in ((matrix, reference), (matrix.mT, reference.mT)):
                        actual = spectral_power(candidate, a)
                        error = (actual - expected).norm() / expected.norm()
                        self.assertLess(error.item(), 2e-4, (dtype, condition, a))

    def test_iteration_and_polynomial_accuracy_controls(self):
        singular_values = torch.tensor([1., 1e-3, 1e-6], dtype=torch.float64)
        matrix = torch.diag(singular_values)
        reference = torch.eye(3, dtype=matrix.dtype)
        low = (spectral_power(matrix, 0., steps=10) - reference).norm()
        high = (spectral_power(matrix, 0., steps=45) - reference).norm()
        self.assertLess(high.item(), 1e-10)
        self.assertGreater(low.item(), 0.5)
        reference = torch.diag(singular_values.sqrt())
        low = (spectral_power(matrix, 0.5, steps=45, polynomial_terms=2) - reference).norm()
        high = (spectral_power(matrix, 0.5, steps=45, polynomial_terms=20) - reference).norm()
        self.assertLess(high.item(), 1e-8)
        self.assertLess(high.item(), low.item() / 100)

    def test_scale_restoration_and_zero_safe(self):
        matrix = torch.tensor([[1., 2.], [-3., 4.]])
        for a in (0., 0.5, 1.):
            reference = spectral_power(matrix, a)
            for factor in (1e-20, 1e-5, 1e5, 1e20):
                actual = spectral_power(matrix * factor, a)
                torch.testing.assert_close(actual, reference * factor ** a, rtol=2e-5, atol=0)
        torch.testing.assert_close(spectral_power(torch.zeros(2, 3), 0), torch.zeros(2, 3))

    def test_working_step_uses_no_decompositions_or_solves(self):
        matrix = torch.nn.Parameter(torch.tensor([[1., 2.], [-3., 4.]]))
        opt = make_optimizer(matrix, probability=1., start_fraction=0.)
        with mock.patch('torch.linalg.svd', side_effect=AssertionError('SVD forbidden')), \
             mock.patch('torch.linalg.svdvals', side_effect=AssertionError('SVD forbidden')), \
             mock.patch('torch.linalg.eigh', side_effect=AssertionError('eigh forbidden')), \
             mock.patch('torch.linalg.inv', side_effect=AssertionError('inverse forbidden')), \
             mock.patch('torch.linalg.solve', side_effect=AssertionError('solve forbidden')):
            for _ in range(20):
                matrix.grad = torch.tensor([[0.5, 0.2], [-0.1, 0.8]])
                opt.step()
        self.assertEqual(opt.completed_steps, 20)

    def test_missing_and_sparse_gradients(self):
        param = torch.nn.Parameter(torch.ones(2, 2))
        opt = make_optimizer(param)
        opt.step()
        self.assertNotIn('momentum_buffer', opt.state[param])
        torch.testing.assert_close(param, torch.ones_like(param))
        param.grad = torch.eye(2).to_sparse()
        with self.assertRaises(RuntimeError):
            opt.step()
        self.assertEqual(opt.completed_steps, 1)

    def test_adamw_auxiliary_matches_torch(self):
        param = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
        other = torch.nn.Parameter(param.detach().clone())
        opt = PowerLionMuon([{'params': [param], 'use_power': False}], total_steps=20,
                            lr=0.01, weight_decay=0.1, betas=(0.9, 0.99), eps=1e-8)
        reference = torch.optim.AdamW([other], lr=0.01, weight_decay=0.1,
                                     betas=(0.9, 0.99), eps=1e-8)
        for step in range(20):
            param.grad = torch.tensor([0.1 + step, -0.2], dtype=param.dtype)
            other.grad = param.grad.clone()
            opt.step()
            reference.step()
            torch.testing.assert_close(param, other, rtol=1e-12, atol=1e-12)

    def test_rng_isolated_and_exact_resume(self):
        for dtype in (torch.float32, torch.bfloat16):
            param = torch.nn.Parameter(torch.ones(3, 2, dtype=dtype))
            opt = make_optimizer(param, start_fraction=0, probability=0.5)
            rng = torch.get_rng_state().clone()
            for step in range(7):
                param.grad = torch.full_like(param, 0.1234567 + step)
                opt.step()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            saved = copy.deepcopy(opt.state_dict())
            other = torch.nn.Parameter(param.detach().clone())
            restored = make_optimizer(other, start_fraction=0, probability=0.5)
            restored.load_state_dict(saved)
            torch.testing.assert_close(restored.state[other]['momentum_buffer'],
                                       opt.state[param]['momentum_buffer'], rtol=0, atol=0)
            self.assertEqual(restored.state[other]['momentum_buffer'].dtype, torch.float32)
            for step in range(7, 20):
                param.grad = torch.full_like(param, 0.1234567 + step)
                other.grad = param.grad.clone()
                opt.step()
                restored.step()
                torch.testing.assert_close(param, other, rtol=0, atol=0)
                self.assertEqual(opt.get_step_metrics(), restored.get_step_metrics())
            with self.assertRaises(ValueError):
                restored.step()
            mismatch = make_optimizer(torch.nn.Parameter(torch.ones_like(param)), probability=0.9)
            with self.assertRaises(ValueError):
                mismatch.load_state_dict(saved)

    def test_scheduler_and_tiny_training(self):
        torch.manual_seed(42)
        model = torch.nn.Linear(3, 2)
        opt = PowerLionMuon([{'params': [model.weight], 'use_power': True},
                             {'params': [model.bias], 'use_power': False}],
                            total_steps=20, lr=0.1, weight_decay=0, start_fraction=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1 - i / 20)
        x = torch.randn(16, 3)
        target = torch.randn(16, 2)
        initial = (model(x) - target).square().mean().item()
        for _ in range(20):
            opt.zero_grad()
            loss = (model(x) - target).square().mean()
            loss.backward()
            opt.step()
            scheduler.step()
            self.assertTrue(torch.isfinite(loss))
        self.assertLess(loss.item(), initial)
        self.assertEqual(opt.last_exponent, 1)
        self.assertTrue(all(g['lr'] == 0 for g in opt.param_groups))

    def test_parser(self):
        from config.base import parse_args
        args = parse_args(argparse.ArgumentParser(), ['--experiment-name', 'power-test',
                          '--opt', 'power_lion_muon', '--power-lion-muon-probability', '0.4',
                          '--power-lion-muon-exponent-end', '1.5'], None)
        self.assertEqual(args.opt, 'power_lion_muon')
        self.assertEqual(args.power_lion_muon_probability, 0.4)
        self.assertEqual(args.power_lion_muon_exponent_end, 1.5)

    @unittest.skipUnless(dist.is_gloo_available(), 'Gloo backend unavailable')
    def test_two_rank_branch_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_worker, args=((Path(directory) / 'rendezvous').as_uri(),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()

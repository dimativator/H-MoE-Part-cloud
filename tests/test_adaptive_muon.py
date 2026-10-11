from copy import deepcopy
from argparse import ArgumentParser
from types import SimpleNamespace
from unittest import TestCase

import torch

from optim.adaptive_muon import Muon2, NorMuon, build_adaptive_muon
from config.base import parse_args


def _ns(matrix, steps):
    x = matrix.bfloat16()
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a = x @ x.T
        b = -4.7750 * a + 2.0315 * a @ a
        x = 3.4445 * x + b @ x
    return x.T if transpose else x


class AdaptiveMuonTest(TestCase):
    def test_cli_choices(self):
        for name in ("muon2", "normuon"):
            args = parse_args(ArgumentParser(), ["--experiment-name", "smoke", "--opt", name,
                                               "--muon2-ns-steps", "4"], None)
            self.assertEqual(args.opt, name)
            self.assertEqual(args.muon2_ns_steps, 4)

    def test_muon2_matches_algorithm_1_multistep(self):
        torch.manual_seed(7)
        for shape in ((4, 7), (7, 4), (5, 5)):
            p = torch.nn.Parameter(torch.randn(shape))
            expected = p.detach().clone()
            m, v = torch.zeros_like(p), torch.zeros_like(p)
            opt = Muon2([p], lr=0.03, momentum=0.9, beta2=0.95, weight_decay=0.1)
            for _ in range(4):
                grad = torch.randn_like(p)
                p.grad = grad.clone()
                m = 0.9 * m + 0.1 * grad
                v = 0.95 * v + 0.05 * grad.square()
                update = _ns(m / (v.sqrt() + 1e-8), 5).float()
                expected.mul_(1 - 0.03 * 0.1)
                expected.add_(update * (shape[1] / shape[0]) ** 0.5, alpha=-0.03)
                opt.step()
                torch.testing.assert_close(p, expected)
                torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
                torch.testing.assert_close(opt.state[p]["variance"], v)

    def test_normuon_matches_official_rule(self):
        torch.manual_seed(3)
        for shape in ((4, 7), (7, 4)):
            for nesterov in (False, True):
                p = torch.nn.Parameter(torch.randn(shape))
                expected = p.detach().clone()
                m = torch.zeros_like(p)
                v = torch.zeros_like(p[:, :1])
                opt = NorMuon([p], lr=0.02, nesterov=nesterov)
                for _ in range(3):
                    grad = torch.randn_like(p)
                    p.grad = grad.clone()
                    m.lerp_(grad, 0.05)
                    direction = grad.lerp(m, 0.95) if nesterov else m
                    update = _ns(direction, 5).float()
                    norm = update.norm()
                    v.lerp_(update.square().mean(dim=-1, keepdim=True), 0.05)
                    update.mul_(1 / (v.sqrt() + 1e-10))
                    update.mul_(norm / (update.norm() + 1e-10))
                    update.mul_(max(1, shape[0] / shape[1]) ** 0.5)
                    expected.add_(update, alpha=-0.02)
                    opt.step()
                    torch.testing.assert_close(p, expected)
                    torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
                self.assertEqual(opt.state[p]["variance"].shape, (shape[0], 1))

    def test_auxiliary_adamw_and_scheduler(self):
        for cls in (Muon2, NorMuon):
            p = torch.nn.Parameter(torch.randn(7))
            ref = torch.nn.Parameter(p.detach().clone())
            opt = cls([dict(params=[p], is_proj_params=False)], lr=0.03,
                      weight_decay=0.1, adamw_lr_scale=0.1)
            adam = torch.optim.AdamW([ref], lr=0.003, betas=(0.9, 0.95), weight_decay=0.1)
            schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
            ref_schedule = torch.optim.lr_scheduler.StepLR(adam, step_size=1, gamma=0.5)
            for _ in range(4):
                grad = torch.randn_like(p)
                p.grad, ref.grad = grad.clone(), grad.clone()
                opt.step()
                adam.step()
                torch.testing.assert_close(p, ref)
                schedule.step()
                ref_schedule.step()

    def test_zero_missing_gradients_and_closure(self):
        for cls in (Muon2, NorMuon):
            p = torch.nn.Parameter(torch.ones(4, 7))
            opt = cls([p], weight_decay=0.1)
            opt.step()
            self.assertFalse(opt.state)
            torch.testing.assert_close(p, torch.ones_like(p))
            p.grad = torch.zeros_like(p)
            opt.step()
            self.assertTrue(torch.isfinite(p).all())
            before = p.detach().clone()
            opt.zero_grad()
            opt.step()
            torch.testing.assert_close(p, before, rtol=0, atol=0)
            self.assertEqual(opt.step(lambda: torch.tensor(2.0)).item(), 2.0)

    def test_checkpoint_resume(self):
        for cls in (Muon2, NorMuon):
            for dtype in (torch.float32, torch.bfloat16):
                p = torch.nn.Parameter(torch.randn(4, 7, dtype=dtype))
                opt = cls([p])
                p.grad = torch.randn_like(p)
                opt.step()
                q = torch.nn.Parameter(p.detach().clone())
                resumed = cls([q])
                resumed.load_state_dict(deepcopy(opt.state_dict()))
                self.assertEqual(resumed.state[q]["variance"].dtype, torch.float32)
                grad = torch.randn_like(p)
                p.grad, q.grad = grad.clone(), grad.clone()
                opt.step()
                resumed.step()
                torch.testing.assert_close(p, q, rtol=0, atol=0)
                torch.testing.assert_close(opt.state[p]["variance"], resumed.state[q]["variance"], rtol=0, atol=0)

    def test_builder_and_invalid_options(self):
        args = SimpleNamespace(opt="muon2", fp8_optim=False, lr=0.01,
                               momentum=0.95, muon_beta2=0.95, muon_ns_steps=5,
                               muon2_ns_steps=5, eps=1e-8, weight_decay=0.1,
                               nesterov=True, beta1=0.9, beta2=0.95,
                               muon_adamw_lr_scale=1.0)
        groups = [dict(params=[torch.nn.Parameter(torch.ones(4, 7))], is_proj_params=True),
                  dict(params=[torch.nn.Parameter(torch.ones(7))], is_proj_params=False, weight_decay=0)]
        self.assertIsInstance(build_adaptive_muon(groups, args), Muon2)
        args.opt = "normuon"
        self.assertIsInstance(build_adaptive_muon(groups, args), NorMuon)
        args.fp8_optim = True
        with self.assertRaises(ValueError):
            build_adaptive_muon(groups, args)
        for kwargs in (dict(momentum=1), dict(beta2=-1), dict(ns_steps=0), dict(eps=0)):
            with self.assertRaises(ValueError):
                Muon2([torch.nn.Parameter(torch.ones(4, 7))], **kwargs)
        with self.assertRaises(ValueError):
            NorMuon([torch.nn.Parameter(torch.ones(7))])

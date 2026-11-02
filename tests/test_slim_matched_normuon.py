"""Independent compressed update, factory, coverage, and launch checks."""
from argparse import ArgumentParser
import importlib.util
import os
from pathlib import Path
import sys
from unittest import TestCase
from unittest.mock import patch

import torch

from config.base import parse_args
from optim.adaptive_muon import build_adaptive_muon
from optim.matched_muon import MatchedMuon
from optim.slim_normuon import SlimMatchedNorMuon
from test_adaptive_muon import _ns


class SlimMatchedNorMuonTest(TestCase):
    def test_reference_both_axes_and_row_equivalence(self):
        torch.manual_seed(27)
        for shape in ((5, 9), (9, 5)):
            for axis in (0, 1):
                p = torch.nn.Parameter(torch.randn(shape))
                q = torch.nn.Parameter(p.detach().clone())
                opt = SlimMatchedNorMuon([dict(params=[p], compress_dims=(axis,))], lr=0.0005, weight_decay=0.1)
                matched = MatchedMuon([q], lr=0.0005, weight_decay=0.1, row_normalization=True)
                expected = p.detach().clone()
                momentum = torch.zeros_like(p)
                variance = torch.zeros_like(p.mean(axis, keepdim=True))
                for _ in range(5):
                    grad = torch.randn_like(p)
                    p.grad, q.grad = grad.clone(), grad.clone()
                    momentum.lerp_(grad, 0.05)
                    u = _ns(grad.lerp(momentum, 0.95), 5).float()
                    variance.lerp_(u.square().mean(axis, keepdim=True), 0.05)
                    u = u / (variance.sqrt() + 1e-10)
                    u = u * (0.2 * u.numel() ** 0.5 / u.norm())
                    expected.mul_(1 - 0.0005 * 0.1).add_(u, alpha=-0.0005)
                    opt.step()
                    matched.step()
                    torch.testing.assert_close(p, expected)
                    torch.testing.assert_close(opt.state[p]['variance'], variance)
                    torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
                    if axis == 1:
                        torch.testing.assert_close(p, q, rtol=0, atol=0)
                p.grad = torch.zeros_like(p)
                # A fresh zero state must not produce NaNs.
                z = torch.nn.Parameter(torch.zeros(shape))
                zero = SlimMatchedNorMuon([dict(params=[z], compress_dims=(axis,))])
                z.grad = torch.zeros_like(z)
                zero.step()
                self.assertTrue(torch.isfinite(z).all())

    def test_factory_compresses_auxiliary_and_fixes_its_lr(self):
        args = parse_args(ArgumentParser(), ['--experiment-name', 'test', '--opt', 'slim_normuon_matched',
                          '--lr', '0.0005', '--muon_adamw_lr_scale', '2'], None)
        matrix = torch.nn.Parameter(torch.randn(5, 9))
        embed = torch.nn.Parameter(torch.randn(11, 9))
        norm = torch.nn.Parameter(torch.ones(9))
        groups = [dict(params=[matrix], is_proj_params=True),
                  dict(params=[embed, norm], is_proj_params=False, weight_decay=0)]
        named = {'transformer.h.0.attn.o_proj.weight': matrix,
                 'transformer.wte.weight': embed, 'transformer.ln_f.weight': norm}
        opt = build_adaptive_muon(groups, args, named)
        self.assertIsInstance(opt, SlimMatchedNorMuon)
        for p in named.values():
            p.grad = torch.randn_like(p)
        opt.step()
        self.assertEqual(opt.state[matrix]['variance'].shape, (1, 9))
        self.assertEqual(opt.state[embed]['variance'].shape, (11, 1))
        self.assertEqual(opt.state[norm]['variance'].shape, (9,))
        self.assertEqual([g['lr'] for g in opt.param_groups], [0.0005, 0.001, 0.001])

    def test_cloud_smoke_then_full_no_other_optimizer(self):
        cloud = Path(__file__).resolve().parents[1] / 'scripts/cloud'
        with patch.dict(os.environ, {'MODE': 'matched', 'LR': '5e-4'}), patch.object(sys, 'path', [str(cloud), *sys.path]):
            spec = importlib.util.spec_from_file_location('slim_matched_campaign', cloud / 'run_slim_matched_normuon.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with patch.object(module, 'preflight'), patch.object(module.campaign, 'train') as train:
                module.main()
            self.assertEqual([x.args for x in train.call_args_list], [
                ('slim_normuon_matched', '5e-4', 257, 'matched_smoke', 39250),
                ('slim_normuon_matched', '5e-4', 257, '1xC', 39250)])
            self.assertTrue(all(x.kwargs == {'weight_decay': '1e-1'} for x in train.call_args_list))

    def test_cloud_grid_keeps_all_other_settings_fixed(self):
        cloud = Path(__file__).resolve().parents[1] / 'scripts/cloud'
        with patch.object(sys, 'path', [str(cloud), *sys.path]):
            spec = importlib.util.spec_from_file_location('slim_matched_grid', cloud / 'run_slim_matched_normuon.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        for lr in ('1e-4', '5e-4', '1e-3', '2e-3'):
            with patch.dict(os.environ, {'LR': lr}), patch.object(module, 'preflight'), patch.object(module.campaign, 'train') as train:
                module.main()
            self.assertEqual([c.args for c in train.call_args_list], [
                ('slim_normuon_matched', lr, 257, 'matched_smoke', 39250),
                ('slim_normuon_matched', lr, 257, '1xC', 39250)])
            self.assertTrue(all(c.kwargs == {'weight_decay': '1e-1'} for c in train.call_args_list))
        with patch.dict(os.environ, {'LR': '0.1'}), patch.object(module, 'preflight'):
            with self.assertRaises(ValueError):
                module.main()

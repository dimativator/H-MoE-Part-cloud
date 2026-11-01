from argparse import ArgumentParser
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from unittest import TestCase
from unittest.mock import patch

import torch

from config.base import parse_args
from optim.adaptive_muon import build_adaptive_muon
from optim.matched_muon import MatchedMuon
from third_party.normuon.kernels import zeropower_via_newtonschulz5


class MatchedMuonTest(TestCase):
    def test_pair_matches_independent_rule_and_common_rms(self):
        torch.manual_seed(19)
        for shape in ((9, 5), (5, 9), (6, 6)):
            for normalized in (False, True):
                p = torch.nn.Parameter(torch.randn(shape))
                opt = MatchedMuon([p], lr=0.001, weight_decay=0.1, row_normalization=normalized)
                expected = p.detach().clone()
                momentum = torch.zeros_like(p)
                variance = torch.zeros(shape[0], 1)
                for _ in range(5):
                    g = torch.randn(shape) * torch.arange(1, shape[0] + 1)[:, None]
                    p.grad = g.clone()
                    momentum.lerp_(g, 0.05)
                    direction = g.lerp(momentum, 0.95)
                    u = zeropower_via_newtonschulz5(direction, 5).float()
                    if normalized:
                        variance.lerp_(u.square().mean(-1, keepdim=True), 0.05)
                        u = u / (variance.sqrt() + 1e-10)
                    u = u * (0.2 * u.numel() ** 0.5 / u.norm().clamp_min(1e-10))
                    torch.testing.assert_close(u.square().mean().sqrt(), torch.tensor(0.2))
                    expected.mul_(1 - 0.001 * 0.1).add_(u, alpha=-0.001)
                    opt.step()
                    torch.testing.assert_close(p, expected)
                    torch.testing.assert_close(p.grad, g, rtol=0, atol=0)
                    torch.testing.assert_close(opt.state[p]['variance'], variance)

    def test_auxiliary_lr_is_fixed_across_matrix_grid_and_matches_adamw(self):
        for lr in (0.0001, 0.0005, 0.001, 0.002):
            args = parse_args(ArgumentParser(), ['--experiment-name', 'test', '--opt', 'normuon_matched',
                              '--lr', str(lr), '--muon_adamw_lr_scale', str(0.001 / lr)], None)
            matrix = torch.nn.Parameter(torch.randn(5, 9))
            aux = torch.nn.Parameter(torch.randn(11, 9))
            reference = torch.nn.Parameter(aux.detach().clone())
            opt = build_adaptive_muon([dict(params=[matrix], is_proj_params=True),
                                      dict(params=[aux], is_proj_params=False, weight_decay=0)], args)
            adam = torch.optim.AdamW([reference], lr=0.001, betas=(0.9, 0.95), eps=args.eps, weight_decay=0)
            self.assertAlmostEqual(opt.param_groups[1]['lr'], 0.001)
            a = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.5)
            b = torch.optim.lr_scheduler.StepLR(adam, 1, gamma=0.5)
            for _ in range(3):
                g = torch.randn_like(aux)
                aux.grad, reference.grad = g.clone(), g.clone()
                opt.step()
                adam.step()
                torch.testing.assert_close(aux, reference)
                a.step()
                b.step()

    def test_zero_direction_is_finite(self):
        for normalized in (False, True):
            p = torch.nn.Parameter(torch.zeros(5, 9))
            opt = MatchedMuon([p], row_normalization=normalized)
            p.grad = torch.zeros_like(p)
            opt.step()
            self.assertTrue(torch.isfinite(p).all())
            self.assertEqual(p.abs().sum(), 0)

    def test_cloud_pair_has_two_smokes_then_two_full_runs(self):
        cloud = Path(__file__).resolve().parents[1] / 'scripts/cloud'
        with patch.dict(os.environ, {'MODE': 'matched', 'MATCHED_STAGE': '257m', 'LR': '5e-4'}):
            with patch.object(sys, 'path', [str(cloud), *sys.path]):
                spec = importlib.util.spec_from_file_location('matched_campaign', cloud / 'run_matched_normuon.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            with patch.object(module, 'preflight'), patch.object(module.campaign, 'train') as train:
                module.main()
            self.assertEqual([x.args for x in train.call_args_list], [
                ('muon_matched', '5e-4', 257, 'matched_smoke', 39250),
                ('normuon_matched', '5e-4', 257, 'matched_smoke', 39250),
                ('muon_matched', '5e-4', 257, '1xC', 39250),
                ('normuon_matched', '5e-4', 257, '1xC', 39250)])
            self.assertTrue(all(x.kwargs == {'weight_decay': '1e-1'} for x in train.call_args_list))

    def test_500m_gate_requires_all_pairs_and_improvement(self):
        from importlib import import_module
        cloud = Path(__file__).resolve().parents[1] / 'scripts/cloud'
        with patch.object(sys, 'path', [str(cloud), *sys.path]):
            module = import_module('run_matched_normuon')
        with tempfile.TemporaryDirectory() as tmp:
            import json
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                module.select_winner(root)
            for lr in module.campaign.LRS:
                for opt in ('muon_matched', 'normuon_matched'):
                    record = dict(optimizer=opt, lr=lr, size=257, phase='1xC', iter=39250,
                                  consumed_tokens=5144576000, weight_decay=0.1, ns_steps=5,
                                  checkpoint_saving=False, update_rms=0.2,
                                  row_normalization=opt == 'normuon_matched', auxiliary_lr=0.001,
                                  **{'final-val/loss': 3.1})
                    path = root / f'lr{lr}' / 'finals' / f'{opt}_257m_bf16_lr{lr}_wd1e-1_1xC_2gpu.json'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(record))
            with self.assertRaises(RuntimeError):
                module.select_winner(root)
            path = root / 'lr1e-3/finals/normuon_matched_257m_bf16_lr1e-3_wd1e-1_1xC_2gpu.json'
            record = json.loads(path.read_text())
            record['final-val/loss'] = 2.99
            path.write_text(json.dumps(record))
            self.assertEqual(module.select_winner(root)['lr'], '1e-3')

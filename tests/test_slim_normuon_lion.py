"""Independent periodic update, auxiliary math, phase state and cloud contracts."""
from argparse import ArgumentParser
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import TestCase
from unittest.mock import patch

import torch

from config.base import parse_args
from optim.adaptive_muon import build_adaptive_muon
from optim.slim_normuon_lion import SlimNorMuonLion
from test_adaptive_muon import _ns


class SlimNorMuonLionTest(TestCase):
    def test_independent_rule_period_axes_weight_decay_and_frozen_variance(self):
        torch.manual_seed(91)
        for period in (1, 2, 5):
            for shape in ((5, 9), (9, 5)):
                for axis in (0, 1):
                    p = torch.nn.Parameter(torch.randn(shape))
                    expected = p.detach().clone()
                    m = torch.zeros_like(p)
                    v = torch.zeros_like(p.mean(axis, keepdim=True))
                    opt = SlimNorMuonLion([dict(params=[p], compress_dims=(axis,))],
                                         period=period, lr=0.001, weight_decay=0.1)
                    with patch('optim.slim_normuon_lion.zeropower_via_newtonschulz5', wraps=_ns) as ns:
                        for step in range(1, 2 * period + 2):
                            g = torch.randn_like(p)
                            p.grad = g.clone()
                            d = 0.95 * m + 0.05 * g
                            if step % period == 0:
                                u = _ns(d, 5).float()
                                v.lerp_(u.square().mean(axis, keepdim=True), 0.05)
                                u = u / (v.sqrt() + 1e-10)
                                u = u * (0.2 * u.numel() ** 0.5 / u.norm().clamp_min(1e-10))
                                lr = 0.001
                            else:
                                u = d.sign()
                                lr = 0.00001
                            expected.mul_(1 - lr * 0.1).add_(u, alpha=-lr)
                            m.lerp_(g, 0.01)
                            opt.step()
                            torch.testing.assert_close(p, expected)
                            torch.testing.assert_close(opt.state[p]['momentum'], m)
                            torch.testing.assert_close(opt.state[p]['variance'], v)
                            torch.testing.assert_close(p.grad, g, rtol=0, atol=0)
                            self.assertEqual(ns.call_count, step // period)
                            self.assertEqual(opt.get_step_metrics()['slim_normuon_lion/slim_steps'], step // period)

    def test_auxiliary_compressed_adamw_every_step_and_scheduler(self):
        for period in (2, 5):
            p = torch.nn.Parameter(torch.randn(11, 7))
            expected = p.detach().clone()
            m, v = torch.zeros_like(p), torch.zeros(11, 1)
            opt = SlimNorMuonLion([dict(params=[p], compress_dims=(1,), is_proj_params=False, weight_decay=0)],
                                 period=period, lr=0.0005, adamw_lr_scale=2)
            schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
            for step in range(1, 8):
                g = torch.randn_like(p)
                p.grad = g.clone()
                m.lerp_(g, 0.1)
                v.lerp_(g.square().mean(1, keepdim=True), 0.05)
                u = (m / (1 - 0.9 ** step)) / ((v / (1 - 0.95 ** step)).sqrt() + 1e-8)
                expected.add_(u, alpha=-0.001 * 0.9 ** (step - 1))
                opt.step()
                torch.testing.assert_close(p, expected)
                schedule.step()

    def test_roundtrip_global_clock_missing_gradient_and_zero(self):
        for period in (2, 5):
            p, other = torch.nn.Parameter(torch.randn(5, 9)), torch.nn.Parameter(torch.ones(5, 9))
            groups = [dict(params=[p], compress_dims=(1,)), dict(params=[other], compress_dims=(0,))]
            opt = SlimNorMuonLion(groups, period=period, lr=0.001)
            for _ in range(period - 1):
                p.grad = torch.randn_like(p)
                opt.step()
            q, r = torch.nn.Parameter(p.detach().clone()), torch.nn.Parameter(other.detach().clone())
            restored = SlimNorMuonLion([{**groups[0], 'params': [q]}, {**groups[1], 'params': [r]}], period=period)
            restored.load_state_dict(deepcopy(opt.state_dict()))
            p.grad = torch.randn_like(p)
            other.grad = torch.randn_like(other)
            q.grad, r.grad = p.grad.clone(), other.grad.clone()
            opt.step()
            restored.step()
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            torch.testing.assert_close(other, r, rtol=0, atol=0)
            self.assertEqual(opt.get_step_metrics()['slim_normuon_lion/is_slim_step'], 1)
            z = torch.nn.Parameter(torch.zeros(5, 9))
            zero = SlimNorMuonLion([dict(params=[z], compress_dims=(1,))], period=period)
            for _ in range(2 * period):
                z.grad = torch.zeros_like(z)
                zero.step()
                self.assertTrue(torch.isfinite(z).all())
                self.assertEqual(z.abs().sum(), 0)

    def test_factory_grid_fixed_auxiliary_and_full_mapping(self):
        for period in (2, 5):
            for lr in (0.0001, 0.0005, 0.001, 0.002):
                args = parse_args(ArgumentParser(), ['--experiment-name', 'test', '--opt', 'slim_normuon_lion',
                                  '--lr', str(lr), '--muon_adamw_lr_scale', str(0.001 / lr),
                                  '--slim-lion-period', str(period)], None)
                p, aux = torch.nn.Parameter(torch.randn(5, 9)), torch.nn.Parameter(torch.randn(11, 9))
                opt = build_adaptive_muon([dict(params=[p], is_proj_params=True),
                                          dict(params=[aux], is_proj_params=False, weight_decay=0)], args,
                                         {'transformer.h.0.attn.o_proj.weight': p, 'transformer.wte.weight': aux})
                self.assertIsInstance(opt, SlimNorMuonLion)
                self.assertEqual(opt.param_groups[0]['slim_lion_period'], period)
                self.assertAlmostEqual(opt.param_groups[1]['lr'], 0.001)
                p.grad, aux.grad = torch.ones_like(p), torch.ones_like(aux)
                opt.step()
                self.assertEqual(opt.state[p]['variance'].shape, (1, 9))
                self.assertEqual(opt.state[aux]['variance'].shape, (11, 1))
        for bad in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                SlimNorMuonLion([dict(params=[p], compress_dims=(1,))], period=bad)

    def test_cloud_one_gpu_replay_two_cycle_smoke_and_no_checkpoints(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/cloud/run_adaptive_muon.py'
        for period in (2, 5):
            with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                    'MODE': 'matched', 'CLOUD_TRAIN_WORLD_SIZE': '1', 'RESULTS_DIR': folder,
                    'CLOUD_DISABLE_CHECKPOINTS': '1', 'SLIM_LION_PERIOD': str(period),
                    'OMPI_COMM_WORLD_SIZE': '1', 'OMPI_COMM_WORLD_RANK': '0'}):
                spec = importlib.util.spec_from_file_location('lion_cloud_test', path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.final_metric = lambda *args: {'val_loss': 3.0}
                commands = []
                class Process:
                    returncode = 0
                    def poll(self):
                        return 0
                def run(command, **kwargs):
                    commands.append(command)
                    return Process()
                with patch.object(module.subprocess, 'Popen', side_effect=run):
                    for phase in ('matched_smoke', '1xC'):
                        module.train('slim_normuon_lion', '5e-4', 257, phase, 39250)
                for command in commands:
                    for flag, value in [('--batch-size', '32'), ('--acc-steps', '4'),
                                        ('--fineweb-replay-world-size', '2'), ('--muon_adamw_lr_scale', '2.0'),
                                        ('--slim-lion-period', str(period))]:
                        self.assertEqual(command[command.index(flag) + 1], value)
                    self.assertIn('--nproc_per_node=1', command)
                    self.assertIn('--no-local-save', command)
                    self.assertNotIn('--inter-ckpts', command)
                    self.assertNotIn('--resume-from', command)
                    self.assertIn('_1gpu_p' + str(period), command[command.index('--experiment-name') + 1])
                self.assertEqual(commands[0][commands[0].index('--early-stop-iteration')+1], str(2 * period))
                self.assertNotIn('--early-stop-iteration', commands[1])

    def test_closure_once_and_hidden_scheduler_ratio(self):
        p = torch.nn.Parameter(torch.ones(3, 4))
        opt = SlimNorMuonLion([dict(params=[p], compress_dims=(1,))], period=2, lr=0.001, weight_decay=0.1)
        calls = []
        def closure():
            calls.append(torch.is_grad_enabled())
            p.grad = torch.ones_like(p)
            return p.square().sum()
        loss = opt.step(closure)
        self.assertEqual(calls, [True])
        self.assertIsNotNone(loss)
        self.assertAlmostEqual(opt._step_lr(opt.param_groups[0]), 0.00001)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.5)
        scheduler.step()
        opt.step(closure)
        self.assertEqual(calls, [True, True])
        self.assertAlmostEqual(opt._step_lr(opt.param_groups[0]), 0.0005)

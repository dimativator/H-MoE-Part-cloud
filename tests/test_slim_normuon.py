"""Independent update and complete Llama compression coverage checks."""

from argparse import ArgumentParser
from copy import deepcopy
import sys
import types
from unittest import TestCase

import torch

from config.base import parse_args
from optim.adaptive_muon import NorMuon, build_adaptive_muon
from optim.slim_normuon import SlimNorMuon, compression_groups
from test_adaptive_muon import _ns


class SlimNorMuonTest(TestCase):
    def test_hidden_axes_against_reference_and_row_equivalence(self):
        for shape in ((4, 7), (7, 4), (5, 5)):
            for axis in (0, 1):
                for nesterov in (False, True):
                    p = torch.nn.Parameter(torch.randn(shape))
                    expected = p.detach().clone()
                    m = torch.zeros_like(p)
                    v = torch.zeros_like(p.mean(dim=axis, keepdim=True))
                    opt = SlimNorMuon([dict(params=[p], compress_dims=(axis,))], lr=0.001,
                                     weight_decay=0.1, nesterov=nesterov)
                    q = torch.nn.Parameter(p.detach().clone())
                    native = NorMuon([q], lr=0.001, weight_decay=0.1, nesterov=nesterov)
                    for _ in range(5):
                        grad = torch.randn_like(p)
                        p.grad, q.grad = grad.clone(), grad.clone()
                        m.lerp_(grad, 0.05)
                        u = _ns(grad.lerp(m, 0.95) if nesterov else m, 5).float()
                        original_norm = u.norm()
                        v.lerp_(u.square().mean(dim=axis, keepdim=True), 0.05)
                        z = u / (v.sqrt() + 1e-10)
                        update = z * original_norm / (z.norm() + 1e-10)
                        expected.mul_(1 - 0.001 * 0.1)
                        expected.add_(update * max(1, shape[0] / shape[1]) ** 0.5, alpha=-0.001)
                        opt.step()
                        native.step()
                        torch.testing.assert_close(p, expected)
                        torch.testing.assert_close(opt.state[p]['variance'], v)
                        torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
                        if axis == 1:
                            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_auxiliary_slim_adam_reference_and_state_roundtrip(self):
        for shape, dims in (((11, 7), (1,)), ((7,), None)):
            p = torch.nn.Parameter(torch.randn(shape))
            expected = p.detach().clone()
            m = torch.zeros_like(p)
            v = torch.zeros_like(p.mean(dim=dims, keepdim=True) if dims else p)
            group = dict(params=[p], compress_dims=dims, is_proj_params=False, weight_decay=0)
            opt = SlimNorMuon([group], lr=0.001)
            for step in range(1, 5):
                p.grad = torch.randn_like(p)
                m.lerp_(p.grad, 0.1)
                squared = p.grad.square()
                v.lerp_(squared.mean(dim=dims, keepdim=True) if dims else squared, 0.05)
                expected.add_((m / (1 - 0.9 ** step)) / ((v / (1 - 0.95 ** step)).sqrt() + 1e-8), alpha=-0.001)
                opt.step()
                torch.testing.assert_close(p, expected)
            q = torch.nn.Parameter(p.detach().clone())
            resumed = SlimNorMuon([{**group, 'params': [q]}], lr=0.001)
            resumed.load_state_dict(deepcopy(opt.state_dict()))
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
            opt.step()
            resumed.step()
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_real_llama_has_complete_compressed_matrix_coverage(self):
        # Meta construction does not tokenize; tiktoken is optional in CPU tests.
        try:
            import tiktoken
        except ImportError:
            sys.modules.setdefault('tiktoken', types.ModuleType('tiktoken'))
        from models.llama import Llama
        for layers, width, heads in ((12, 1024, 8), (18, 1280, 20)):
            args = parse_args(ArgumentParser(), ['--experiment-name', 'coverage', '--model', 'llama',
                             '--opt', 'slim_normuon', '--n-layer', str(layers), '--n-embd', str(width),
                             '--n-head', str(heads), '--multiple-of', '256'], None)
            with torch.device('meta'):
                model = Llama(args)
            named = dict(model.named_parameters())
            groups = [{**group, 'params': [named[n] for n in group['params']]}
                      for group in model.get_parameter_group_specs()]
            opt = build_adaptive_muon(groups, args, named)
            self.assertIsInstance(opt, SlimNorMuon)
            self.assertEqual(len(opt.param_groups), len(named))
            matrices = [g for g in opt.param_groups if g['params'][0].ndim == 2]
            self.assertEqual(len(matrices), 7 * layers + 2)
            for g in matrices:
                p = g['params'][0]
                self.assertIn(g['compress_dims'], ((0,), (1,)))
                self.assertLess(opt._variance_template(p, g).numel(), p.numel())
            for g in opt.param_groups:
                if g['params'][0].ndim == 1:
                    self.assertIsNone(g['compress_dims'])
            p = torch.nn.Parameter(torch.ones(3, 4))
            with self.assertRaises(ValueError):
                compression_groups([dict(params=[p])], {'unknown.weight': p})

    def test_rules_equal_slim_adam_and_launch_configuration(self):
        import ast
        import importlib.util
        import os
        from pathlib import Path
        from unittest.mock import patch
        from optim.slim_normuon import COMPRESSION_RULES
        repo = Path(__file__).resolve().parents[1]
        tree = ast.parse((repo / 'src/optim/memory_efficient/slim_adam/slim_adam.py').read_text())
        defaults = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_COMPRESSION_RULES' for t in node.targets))
        for kind, dims in COMPRESSION_RULES.items():
            self.assertEqual(dims, defaults[kind])
        spec = importlib.util.spec_from_file_location('campaign', repo / 'scripts/cloud/run_adaptive_muon.py')
        campaign = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {'MODE': '257m_slim_normuon', 'RESULTS_DIR': '/tmp/not-executed', 'LR': '2e-3'}):
            spec.loader.exec_module(campaign)
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            campaign.ROOT = Path(folder)
            campaign.WORLD = 1
            campaign.RANK = 0
            campaign.final_metric = lambda *a: {'val_loss': 3.0}
            commands = []
            def run(command, **kwargs):
                commands.append(command)
            with patch.object(campaign.subprocess, 'run', side_effect=run):
                for phase in ('slim_smoke', '1xC'):
                    campaign.train('slim_normuon', '2e-3', 257, phase, 39250)
            for command in commands:
                for key, expected in (('--opt', 'slim_normuon'), ('--weight-decay', '1e-1'),
                                      ('--iterations', '39250'), ('--muon_ns_steps', '5')):
                    self.assertEqual(command[command.index(key)+1], expected)
                self.assertIn('--no-local-save', command)
                self.assertNotIn('--resume-from', command)
                self.assertNotIn('--normuon-embedding-lr', command)
                self.assertIn('--nproc_per_node=2', command)
            self.assertIn('--early-stop-iteration', commands[0])
            self.assertNotIn('--early-stop-iteration', commands[1])

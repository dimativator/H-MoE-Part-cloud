from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import json

import torch

from config.base import parse_args
from optim.mnd_muon import MNDMuon, build_mnd_muon
from scripts.cloud import run_mnd_muon as campaign
from third_party.matrix_normal.kernels import polar_express


class MNDMuonTest(TestCase):
    def test_matches_algorithm_2_multistep(self):
        torch.manual_seed(17)
        for shape in ((4, 7), (7, 4), (5, 5)):
            p = torch.nn.Parameter(torch.randn(shape))
            expected = p.detach().clone()
            m, n = shape
            row, col = torch.full((m,), 1 / m), torch.full((n,), 1 / n)
            momentum, previous = torch.zeros_like(p), torch.zeros_like(p)
            opt = MNDMuon([p], lr=0.03, weight_decay=0.1)
            for _ in range(5):
                grad = torch.randn_like(p)
                p.grad = grad.clone()
                q = 0.9 * torch.outer(row, col) + 0.1 * grad.square()
                linear = 0.1 * grad * previous
                # Direct positive root of Algorithm 2, independently of the stable kernel.
                a = q @ col.clamp_min(1e-9).reciprocal()
                b = linear @ col.clamp_min(1e-9).rsqrt()
                new_row = ((torch.sqrt(b.square() + 4 * n * a) - b) / (2 * n)).square().clamp_min(1e-9)
                a = q.T @ new_row.reciprocal()
                b = linear.T @ new_row.rsqrt()
                new_col = ((torch.sqrt(b.square() + 4 * m * a) - b) / (2 * m)).square().clamp_min(1e-9)
                scale = torch.sqrt(new_col.sum() / new_row.sum())
                row, col = new_row * scale, new_col / scale
                whitened = grad * row.clamp_min(1e-9).rsqrt()[:, None] * col.clamp_min(1e-9).rsqrt()[None, :]
                momentum = 0.9 * momentum + 0.1 * whitened
                previous = polar_express(momentum, 5)
                expected.mul_(1 - 0.03 * 0.1).add_(previous, alpha=-0.03)
                opt.step()
                torch.testing.assert_close(opt.state[p]['row_factor'], row, rtol=2e-5, atol=1e-7)
                torch.testing.assert_close(opt.state[p]['col_factor'], col, rtol=2e-5, atol=1e-7)
                torch.testing.assert_close(p, expected, rtol=2e-3, atol=2e-3)
                torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)

    def test_auxiliary_adamw_and_group_routing(self):
        args = parse_args(ArgumentParser(), ['--experiment-name', 'test', '--opt', 'mnd_muon',
                         '--lr', '0.03', '--weight-decay', '0.1', '--beta1', '0.95', '--beta2', '0.95'], None)
        matrix = torch.nn.Parameter(torch.randn(4, 7))
        embedding = torch.nn.Parameter(torch.randn(11, 7))
        ref = torch.nn.Parameter(embedding.detach().clone())
        opt = build_mnd_muon([dict(params=[matrix], is_proj_params=True),
                              dict(params=[embedding], is_proj_params=False, weight_decay=0)], args)
        adam = torch.optim.AdamW([ref], lr=0.03, betas=(0.95, 0.95), eps=1e-8, weight_decay=0)
        schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        ref_schedule = torch.optim.lr_scheduler.StepLR(adam, step_size=1, gamma=0.5)
        for _ in range(4):
            grad = torch.randn_like(embedding)
            embedding.grad, ref.grad = grad.clone(), grad.clone()
            matrix.grad = torch.randn_like(matrix)
            opt.step()
            adam.step()
            torch.testing.assert_close(embedding, ref)
            schedule.step()
            ref_schedule.step()
        self.assertNotIn('direction', opt.state[embedding])
        self.assertEqual(opt.state[matrix]['row_factor'].shape, (4,))

    def test_zero_missing_gradients_and_bf16_states(self):
        p = torch.nn.Parameter(torch.ones(4, 7, dtype=torch.bfloat16))
        opt = MNDMuon([p])
        opt.step()
        self.assertFalse(opt.state)
        p.grad = torch.zeros_like(p)
        opt.step()
        self.assertTrue(torch.isfinite(p).all())
        for key in ('momentum', 'direction', 'row_factor', 'col_factor'):
            self.assertEqual(opt.state[p][key].dtype, torch.float32)

    def test_campaign_matches_global_batch_and_disables_all_checkpoints(self):
        for lr in campaign.LRS:
            command, _, _ = campaign.build_command(lr, Path('/tmp/test'), '20261022')
            args = parse_args(ArgumentParser(), command[2:], None)
            self.assertEqual(args.opt, 'mnd_muon')
            self.assertEqual(args.weight_decay, 0.1)
            self.assertEqual(args.batch_size * args.acc_steps, 128)
            self.assertIsNone(args.distributed_backend)
            self.assertEqual(args.fineweb_replay_layout, 'serial')
            self.assertEqual(args.fineweb_replay_world_size, 2)
            self.assertTrue(args.no_local_save)
            self.assertEqual(args.latest_ckpt_interval, 0)
            self.assertEqual(args.permanent_ckpt_interval, 0)
            self.assertEqual(args.beta1, 0.95)
            self.assertEqual(args.beta2, 0.95)

    def test_final_requires_native_metric_and_full_budget(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'metrics.jsonl'
            record = dict(event='validation', iter=campaign.ITERATIONS,
                          consumed_tokens=campaign.ITERATIONS * campaign.TOKENS_PER_STEP)
            record['final-val/loss'] = 3.0
            path.write_text(json.dumps(record) + '\n')
            self.assertEqual(campaign.verify_final(path)['final-val/loss'], 3.0)
            record['consumed_tokens'] -= 1
            path.write_text(json.dumps(record) + '\n')
            with self.assertRaises(RuntimeError):
                campaign.verify_final(path)

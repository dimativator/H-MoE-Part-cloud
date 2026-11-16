import math
import unittest

import torch

from third_party.lion_muon.factorized_momentum import initialize_factors, update_factors


class FactorizedMomentumTest(unittest.TestCase):
    def test_simultaneous_regularized_regression(self):
        torch.manual_seed(4)
        b, a, g = torch.randn(9, 3), torch.randn(3, 7), torch.randn(9, 7)
        old_b, old_a = b.clone(), a.clone()
        gamma, eps = 1 - math.sqrt(0.99), 1e-8
        expected_b = (1-gamma)*b + gamma*g@a.T@torch.linalg.inv(a@a.T+eps*torch.eye(3))
        expected_a = (1-gamma)*a + gamma*torch.linalg.inv(b.T@b+eps*torch.eye(3))@b.T@g
        update_factors(b, a, g, 0.99, eps)
        torch.testing.assert_close(b, expected_b)
        torch.testing.assert_close(a, expected_a)
        self.assertFalse(torch.equal(b, old_b))
        self.assertFalse(torch.equal(a, old_a))

    def test_zero_history_and_no_global_rng_consumption(self):
        g = torch.randn(12, 9)
        rng = torch.get_rng_state().clone()
        b, a = initialize_factors(g, 4, 1729)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertEqual(torch.count_nonzero(b@a).item(), 0)
        for _ in range(200):
            update_factors(b, a, g, 0.99, 1e-8)
        self.assertTrue(torch.isfinite(b).all() and torch.isfinite(a).all())
        self.assertLess(torch.linalg.norm(b@a-g), torch.linalg.norm(g))
        self.assertEqual(b.numel()+a.numel(), 4*(12+9))

    def test_optimizer_branches_and_dense_fallback(self):
        from third_party.lion_muon.lion_muon import LionMuon
        for rank in (0, 4):
            torch.manual_seed(3)
            weight = torch.nn.Parameter(torch.randn(12, 9))
            bias = torch.nn.Parameter(torch.zeros(12))
            opt = LionMuon([weight], adamw_params=[bias], momentum_rank=rank)
            for _ in range(6):
                opt.zero_grad()
                loss = (torch.randn(3, 9) @ weight.T + bias).square().mean()
                loss.backward()
                opt.step()
            state = opt.state[weight]
            self.assertEqual("exp_avg" in state, rank == 0)
            self.assertEqual("momentum_A" in state, rank > 0)
            self.assertTrue(torch.isfinite(weight).all())
            self.assertIn("moment2", opt.state[bias])

    def test_launcher_matches_global_batch_after_ddp_adjustment(self):
        import importlib.util
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "src"))
        from distributed.ddp import DataParallelDistributedBackend
        spec = importlib.util.spec_from_file_location("campaign", root / "scripts/cloud/run_factorized_lionmuon.py")
        campaign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(campaign)
        command = campaign.arguments("test")
        args = SimpleNamespace(batch_size=int(command[command.index("--batch-size")+1]),
            acc_steps=int(command[command.index("--acc-steps")+1]), device="cuda", seed=0, data_seed=1337)
        backend = object.__new__(DataParallelDistributedBackend)
        backend.local_rank = 0
        backend.get_world_size = lambda: 2
        args = backend.get_adjusted_args_for_process(args)
        self.assertEqual((args.batch_size, args.acc_steps), (16, 4))
        self.assertEqual(args.batch_size * args.acc_steps * 2, 128)

    def test_all_campaign_ranks(self):
        g = torch.randn(272, 256) * 1e-4
        for rank in (32, 64, 128, 256):
            b, a = initialize_factors(g, rank, 1729)
            for _ in range(8):
                update_factors(b, a, g, 0.99, 1e-8)
            self.assertTrue(torch.isfinite(b).all() and torch.isfinite(a).all())
            self.assertGreater(torch.linalg.norm(b@a), 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

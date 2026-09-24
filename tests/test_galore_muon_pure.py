from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from src.optim.memory_efficient.frugal import GaloreMuon
from src.optim.multi_optimizer import MultiOptimizer
from src.optim.optimization import get_optimizer


class GaloreMuonPureTest(unittest.TestCase):
    def test_projected_muon_and_adamw_fallback(self) -> None:
        projected = torch.nn.Parameter(
            torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0],
                 [3.0, 4.0, 1.0, 2.0], [4.0, 3.0, 2.0, 1.0]]
            )
        )
        regular = torch.nn.Parameter(torch.ones(4))
        args = SimpleNamespace(
            opt="galore_muon", non_proj_opt="adamw", lr=1e-3,
            weight_decay=0.0, beta1=0.9, beta2=0.99, eps=1e-7,
            proj_params_lr_scale=1.0, update_gap=50, density=0.25,
            reset_statistics=True, inactive_lr_scale=0.0,
            proj_side="std", proj_type="svd", momentum=0.95,
            nesterov=True, muon_ns_steps=5, frugal_muon_adjust_lr=True,
        )
        optimizer = get_optimizer(
            [
                {"params": [projected], "is_proj_params": True},
                {"params": [regular], "is_proj_params": False},
            ],
            args,
        )
        self.assertIsInstance(optimizer, MultiOptimizer)
        self.assertIsInstance(optimizer.proj_opt, GaloreMuon)
        self.assertIsInstance(optimizer.non_proj_opt, torch.optim.AdamW)

        before = projected.detach().clone()
        projected.grad = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1],
             [0.2, 0.4, 0.1, 0.3], [0.3, 0.1, 0.4, 0.2]]
        )
        regular.grad = torch.ones_like(regular)
        with mock.patch(
            "src.optim.memory_efficient.frugal.muon._stateless_muon",
            side_effect=AssertionError("complement update must be skipped"),
        ):
            optimizer.step()

        delta = projected.detach() - before
        projector = optimizer.proj_opt.state[projected]["projector"]
        torch.testing.assert_close(
            delta, projector.project_up(projector.project_down(delta)),
            atol=1e-6, rtol=1e-5,
        )
        self.assertFalse(torch.equal(regular.detach(), torch.ones_like(regular)))


if __name__ == "__main__":
    unittest.main()

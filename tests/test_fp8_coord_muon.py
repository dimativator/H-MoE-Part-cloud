import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optim.memory_efficient.frugal import FP8CoordMuon
from optim.distributed_state_comm import DistributedStateCommunicator
from optim.memory_efficient.frugal.muon import MuonBase
from optim.multi_optimizer import MultiOptimizer
from optim.optimization import get_optimizer


def qargs():
    return SimpleNamespace(
        qgroup_size=8,
        first_order_bit="E4M3",
        second_order_bit="E4M3",
        first_order_expansion="expand",
        second_order_expansion="expand",
        expand_min=16,
    )


def optimizer_args(fp8_optim=False):
    return SimpleNamespace(
        opt="coord_muon",
        non_proj_opt="adamw",
        non_proj_lr=None,
        fp8_optim=fp8_optim,
        proj_params_lr_scale=1.0,
        update_gap=2,
        density=0.5,
        reset_statistics=True,
        inactive_lr_scale=1.0,
        coord_choice="columns",
        lr=2e-3,
        momentum=0.95,
        nesterov=True,
        muon_ns_steps=5,
        frugal_muon_adjust_lr=True,
        eps=1e-8,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.99,
    )


class FP8CoordMuonTest(unittest.TestCase):
    def test_projected_momentum_is_stored_in_fp8(self):
        parameter = torch.nn.Parameter(torch.randn(8, 8))
        optimizer = FP8CoordMuon(
            [{"params": [parameter], "is_proj_params": True}],
            qargs=qargs(),
            density=0.5,
            update_gap=2,
        )
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        state = optimizer.state[parameter]
        self.assertEqual(state["momentum_buffer"].dtype, torch.float8_e4m3fn)
        self.assertIn("scale_momentum_buffer", state)

    def test_coord_muon_uses_adamw_for_non_projection_parameters(self):
        matrix = torch.nn.Parameter(torch.randn(8, 8))
        embedding = torch.nn.Parameter(torch.randn(8))
        groups = [
            {"params": [matrix], "is_proj_params": True},
            {"params": [embedding], "is_proj_params": False, "weight_decay": 0.0},
        ]
        optimizer = get_optimizer(groups, optimizer_args(), qargs=None)
        self.assertIsInstance(optimizer, MultiOptimizer)
        self.assertIsInstance(optimizer.non_proj_opt, torch.optim.AdamW)
        self.assertIs(optimizer.proj_opt.param_groups[0]["params"][0], matrix)
        self.assertIs(optimizer.non_proj_opt.param_groups[0]["params"][0], embedding)

    def test_one_dimensional_muon_state_does_not_enter_matrix_wire_reuse(self):
        parameter = torch.nn.Parameter(torch.randn(8))
        optimizer = MuonBase(
            [parameter],
            qargs=qargs(),
            distributed_state_sharding=True,
            state_wire_dtype="fp8",
        )
        parameter.grad = torch.randn_like(parameter)

        with patch.object(
            DistributedStateCommunicator,
            "reuses_quantized_state_on_wire",
            new_callable=PropertyMock,
            return_value=True,
        ):
            optimizer.step()

        self.assertTrue(torch.isfinite(parameter).all())


if __name__ == "__main__":
    unittest.main()

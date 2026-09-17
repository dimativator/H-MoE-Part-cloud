import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optim.memory_efficient.frugal import FP8CoordMuon
from optim.multi_optimizer import MultiOptimizer
from optim.optimization import get_optimizer
from third_party.lite.muonlite import MuonLite


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
        lite_ns_steps=6,
        lite_muon_theta=0.95,
        iterations=10,
        warmup_steps=0,
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

    def test_coord_muon_can_match_muon_adamw_fallback(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(8, 8, bias=False)
                self.embed_tokens = torch.nn.Embedding(8, 8)

        model = TinyModel()
        groups = [
            {"params": [model.proj.weight], "is_proj_params": True},
            {
                "params": [model.embed_tokens.weight],
                "is_proj_params": False,
                "weight_decay": 0.0,
            },
        ]
        args = optimizer_args()
        args.non_proj_opt = "muon_adamw"
        optimizer = get_optimizer(groups, args, model=model, qargs=None)

        self.assertIsInstance(optimizer, MultiOptimizer)
        self.assertIsInstance(optimizer.non_proj_opt, MuonLite)
        state = optimizer.non_proj_opt.state[model.embed_tokens.weight]
        self.assertEqual(state["name"], "embed_tokens.weight")
        self.assertEqual(state["use_muon"], 0)
        self.assertEqual(state["subspace_ratio"], 0.0)
        self.assertEqual(state["lr_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()

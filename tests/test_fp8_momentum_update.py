from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase, skipUnless

import torch

from optim.fp8_momentum_update import update_fp8_momentum_
from optim.fp8_state import (
    dequantize_fp8_state,
    init_fp8_state,
    quantize_fp8_state_,
)
from third_party.lite.muonlite import MuonLite


def _qargs(expansion: str):
    return SimpleNamespace(
        qgroup_size=128,
        first_order_bit="E4M3",
        first_order_expansion=expansion,
        expand_min=16,
    )


def _reference_update(state, gradient, qargs, momentum, gradient_alpha):
    updated = dequantize_fp8_state(
        state, "momentum", qargs, signed=True
    ).mul(momentum).add(gradient, alpha=gradient_alpha)
    quantize_fp8_state_(state, "momentum", updated, qargs, signed=True)


class FP8MomentumUpdateTest(TestCase):
    def _initialized_state(self, device, expansion):
        qargs = _qargs(expansion)
        generator = torch.Generator(device=device).manual_seed(1234)
        initial = torch.randn(17, 19, device=device, generator=generator)
        state = {}
        init_fp8_state(state, "momentum", initial, qargs, order="first")
        quantize_fp8_state_(state, "momentum", initial, qargs, signed=True)
        gradient = torch.randn(17, 19, device=device, generator=generator)
        return qargs, state, gradient

    def test_cpu_fallback_matches_reference(self):
        qargs, state, gradient = self._initialized_state("cpu", "expand")
        expected = deepcopy(state)
        _reference_update(expected, gradient, qargs, 0.95, 0.05)

        update_fp8_momentum_(
            state,
            "momentum",
            gradient,
            qargs,
            momentum=0.95,
            gradient_alpha=0.05,
        )

        for key in expected:
            torch.testing.assert_close(state[key], expected[key], rtol=0, atol=0)

    @skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_matches_reference_with_expansion(self):
        self._assert_cuda_parity("expand")

    @skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_matches_reference_without_expansion(self):
        self._assert_cuda_parity("false")

    def _assert_cuda_parity(self, expansion):
        qargs, state, gradient = self._initialized_state("cuda", expansion)
        expected = deepcopy(state)
        _reference_update(expected, gradient, qargs, 0.95, 0.05)

        original_values = state["momentum"]
        update_fp8_momentum_(
            state,
            "momentum",
            gradient,
            qargs,
            momentum=0.95,
            gradient_alpha=0.05,
        )
        torch.cuda.synchronize()

        self.assertIs(state["momentum"], original_values)
        actual_dense = dequantize_fp8_state(state, "momentum", qargs, signed=True)
        expected_dense = dequantize_fp8_state(
            expected, "momentum", qargs, signed=True
        )
        torch.testing.assert_close(actual_dense, expected_dense, rtol=0.03, atol=2e-3)
        torch.testing.assert_close(
            state["scale_momentum"], expected["scale_momentum"], rtol=0.02, atol=1e-7
        )
        if expansion == "expand":
            torch.testing.assert_close(
                state["expand_momentum"],
                expected["expand_momentum"],
                rtol=0,
                atol=1 / qargs.expand_min,
            )
            torch.testing.assert_close(
                state["sqrt_minmax_momentum"],
                expected["sqrt_minmax_momentum"],
                rtol=0.02,
                atol=1e-7,
            )


class MuonVanillaAdamWTest(TestCase):
    @skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_adamw_matches_existing_muon_path(self):
        qargs = SimpleNamespace(
            qgroup_size=128,
            first_order_bit="E4M3",
            second_order_bit="E4M3",
            first_order_expansion="expand",
            second_order_expansion="expand",
            expand_min=16,
        )
        base = torch.linspace(-0.5, 0.5, 257, device="cuda", dtype=torch.bfloat16)
        reference_param = torch.nn.Parameter(base.clone())
        fused_param = torch.nn.Parameter(base.clone())
        common = dict(
            muon_params=[],
            lr=1e-3,
            weight_decay=0.1,
            adamw_betas=(0.9, 0.95),
            adamw_eps=1e-8,
            chi=1.0,
            chi_adamw=1.0,
            subspace_ratio=0.0,
            qargs=qargs,
        )
        reference = MuonLite(
            adamw_params=[("lm_head.weight", reference_param)],
            fused_vanilla_adamw=False,
            **common,
        )
        fused = MuonLite(
            adamw_params=[("lm_head.weight", fused_param)],
            fused_vanilla_adamw=True,
            **common,
        )

        for step in range(3):
            gradient = torch.sin(
                torch.arange(257, device="cuda", dtype=torch.float32) + step
            ).to(torch.bfloat16)
            reference_param.grad = gradient.clone()
            fused_param.grad = gradient.clone()
            reference.step()
            fused.step()
            if step == 0:
                first_reference_state = reference.state[reference_param]
                first_fused_state = fused.state[fused_param]
                for prefix in ("moment1", "moment2"):
                    torch.testing.assert_close(
                        first_fused_state[f"scale_{prefix}"],
                        first_reference_state[f"scale_{prefix}"],
                        rtol=0.02,
                        atol=2e-3,
                    )
        torch.cuda.synchronize()

        torch.testing.assert_close(fused_param, reference_param, rtol=0.01, atol=2e-3)
        reference_state = reference.state[reference_param]
        fused_state = fused.state[fused_param]
        for prefix, signed in (("moment1", True), ("moment2", False)):
            reference_dense = dequantize_fp8_state(
                reference_state, prefix, qargs, signed=signed
            )
            fused_dense = dequantize_fp8_state(
                fused_state, prefix, qargs, signed=signed
            )
            torch.testing.assert_close(
                fused_dense, reference_dense, rtol=0.05, atol=3e-2
            )

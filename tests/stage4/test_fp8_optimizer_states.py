import copy

import torch

from stage4.fp8_optimizer_states import (
    FP8StateOptimizerMixin,
    dequantize_fp8_state,
    init_fp8_state,
    make_fp8_adamw,
    make_fp8_soap,
    quantize_fp8_state_,
)
from stage4.fused_fp8_adamw import make_fused_fp8_adamw


def test_quantization_layout_and_error():
    value = torch.linspace(-4, 4, 383, device="cuda")
    state = {}
    init_fp8_state(state, "moment", value, group_size=128)
    quantize_fp8_state_(
        state,
        "moment",
        value,
        signed=True,
        group_size=128,
    )
    restored = dequantize_fp8_state(
        state,
        "moment",
        signed=True,
        group_size=128,
    )

    assert state["moment"].dtype == torch.float8_e4m3fn
    assert state["scale_moment"].dtype == torch.float32
    assert state["scale_moment"].shape == (3,)
    assert state["expand_moment"].shape == (3,)
    assert state["sqrt_minmax_moment"].shape == (3,)
    assert sum(t.numel() * t.element_size() for t in state.values()) == 419
    assert torch.isfinite(restored).all()
    assert (restored - value).abs().mean() / value.abs().mean() < 0.05


def make_optimizer(parameter):
    optimizer_class = make_fp8_adamw(torch.optim.AdamW)
    return optimizer_class(
        [parameter],
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )


def set_gradient(parameter, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    parameter.grad = torch.randn(
        parameter.shape,
        generator=generator,
        device=parameter.device,
    )


def test_adam_update_and_checkpoint_resume():
    initial = torch.randn(257, device="cuda")
    reference_parameter = torch.nn.Parameter(initial.clone())
    quantized_parameter = torch.nn.Parameter(initial.clone())
    reference = torch.optim.AdamW(
        [reference_parameter],
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    quantized = make_optimizer(quantized_parameter)

    set_gradient(reference_parameter, 1)
    quantized_parameter.grad = reference_parameter.grad.clone()
    reference.step()
    quantized.step()

    assert torch.equal(reference_parameter, quantized_parameter)
    state = quantized.state[quantized_parameter]
    assert state["exp_avg"].dtype == torch.float8_e4m3fn
    assert state["exp_avg_sq"].dtype == torch.float8_e4m3fn
    assert torch.all(state["exp_avg_sq"].float() >= 0)

    saved_parameter = quantized_parameter.detach().clone()
    saved_optimizer = copy.deepcopy(quantized.state_dict())

    resumed_parameter = torch.nn.Parameter(saved_parameter.clone())
    resumed = make_optimizer(resumed_parameter)
    resumed.load_state_dict(saved_optimizer)
    resumed_state = resumed.state[resumed_parameter]
    assert resumed_state["exp_avg"].dtype == torch.float8_e4m3fn
    assert resumed_state["scale_exp_avg"].dtype == torch.float32

    set_gradient(quantized_parameter, 2)
    resumed_parameter.grad = quantized_parameter.grad.clone()
    quantized.step()
    resumed.step()

    assert torch.equal(quantized_parameter, resumed_parameter)
    for key, value in quantized.state[quantized_parameter].items():
        other = resumed.state[resumed_parameter][key]
        if torch.is_tensor(value):
            assert torch.equal(value, other)
        else:
            assert value == other


def test_soap_state_dict_round_trip_on_cpu():
    class DummySOAP(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, {})

        def step(self, closure=None):
            return None

    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = make_fp8_soap(DummySOAP)([parameter])
    state = optimizer.state[parameter]
    for key, signed in optimizer.state_specs:
        value = torch.full((2, 2), -1.0 if signed else 1.0).to(torch.float8_e4m3fn)
        state[key] = value
        state[f"scale_{key}"] = torch.ones(1)
        state[f"expand_{key}"] = torch.ones(1)
        state[f"sqrt_minmax_{key}"] = torch.ones(1)
    state["step"] = 1

    resumed_parameter = torch.nn.Parameter(torch.zeros(2, 2))
    resumed = make_fp8_soap(DummySOAP)([resumed_parameter])
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    resumed_state = resumed.state[resumed_parameter]
    for key, _ in optimizer.state_specs:
        assert resumed_state[key].dtype == torch.float8_e4m3fn
        assert torch.equal(resumed_state[key], state[key])


def test_muon_matrix_state_and_update():
    from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon

    class FP8Muon(FP8StateOptimizerMixin, TensorParallelMuon):
        state_specs = (("momentum_buffer", True),)

    initial = torch.randn(32, 24, device="cuda")
    reference_parameter = torch.nn.Parameter(initial.clone())
    quantized_parameter = torch.nn.Parameter(initial.clone())
    kwargs = {
        "lr": 3e-4,
        "momentum": 0.95,
        "weight_decay": 0.1,
        "nesterov": False,
        "split_qkv": False,
        "fp32_matmul_prec": "medium",
        "coefficient_type": "quintic",
        "num_ns_steps": 5,
        "scale_mode": "spectral",
        "extra_scale_factor": 0.2,
        "pg_collection": None,
    }
    reference = TensorParallelMuon([reference_parameter], **kwargs)
    quantized = FP8Muon([quantized_parameter], **kwargs)

    set_gradient(reference_parameter, 3)
    quantized_parameter.grad = reference_parameter.grad.clone()
    reference.step()
    quantized.step()

    assert torch.equal(reference_parameter, quantized_parameter)
    state = quantized.state[quantized_parameter]
    assert state["momentum_buffer"].dtype == torch.float8_e4m3fn
    assert state["scale_momentum_buffer"].shape == (6,)

    resumed_parameter = torch.nn.Parameter(quantized_parameter.detach().clone())
    resumed = FP8Muon([resumed_parameter], **kwargs)
    resumed.load_state_dict(copy.deepcopy(quantized.state_dict()))
    set_gradient(quantized_parameter, 5)
    resumed_parameter.grad = quantized_parameter.grad.clone()
    quantized.step()
    resumed.step()
    assert torch.equal(quantized_parameter, resumed_parameter)
    for key, value in quantized.state[quantized_parameter].items():
        assert torch.equal(value, resumed.state[resumed_parameter][key])


def test_transformer_engine_fused_adam_wrapper():
    from transformer_engine.pytorch.optimizers import FusedAdam

    initial = torch.randn(257, device="cuda")
    reference_parameter = torch.nn.Parameter(initial.clone())
    quantized_parameter = torch.nn.Parameter(initial.clone())
    kwargs = {
        "lr": 1e-3,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "adam_w_mode": True,
    }
    reference = FusedAdam([reference_parameter], **kwargs)
    quantized = make_fp8_adamw(FusedAdam)([quantized_parameter], **kwargs)

    set_gradient(reference_parameter, 4)
    quantized_parameter.grad = reference_parameter.grad.clone()
    reference.step()
    quantized.step()

    assert torch.equal(reference_parameter, quantized_parameter)
    state = quantized.state[quantized_parameter]
    assert state["exp_avg"].dtype == torch.float8_e4m3fn
    assert state["exp_avg_sq"].dtype == torch.float8_e4m3fn

    resumed_parameter = torch.nn.Parameter(quantized_parameter.detach().clone())
    resumed = make_fp8_adamw(FusedAdam)([resumed_parameter], **kwargs)
    resumed.load_state_dict(copy.deepcopy(quantized.state_dict()))
    assert resumed.state[resumed_parameter]["exp_avg"].dtype == torch.float8_e4m3fn
    set_gradient(quantized_parameter, 6)
    resumed_parameter.grad = quantized_parameter.grad.clone()
    quantized.step()
    resumed.step()
    assert torch.equal(quantized_parameter, resumed_parameter)
    for key, value in quantized.state[quantized_parameter].items():
        if torch.is_tensor(value):
            assert torch.equal(value, resumed.state[resumed_parameter][key])


def test_fused_fp8_adamw_matches_storage_only_reference():
    class AdamWBase(torch.optim.Optimizer):
        def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            *,
            bias_correction=True,
            adam_w_mode=True,
            **kwargs,
        ):
            del kwargs
            assert adam_w_mode
            super().__init__(
                params,
                {
                    "lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                    "bias_correction": bias_correction,
                },
            )

    initial = torch.randn(4099, device="cuda")
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = make_fused_fp8_adamw(AdamWBase)(
        [parameter],
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        adam_w_mode=True,
    )

    reference_parameter = initial.clone()
    reference_m = torch.zeros_like(initial)
    reference_v = torch.zeros_like(initial)
    for step in (1, 2):
        set_gradient(parameter, 20 + step)
        gradient = parameter.grad.clone()
        reference_m.mul_(0.9).add_(gradient, alpha=0.1)
        reference_v.mul_(0.95).addcmul_(gradient, gradient, value=0.05)
        update = reference_m.div(1.0 - 0.9**step).div(
            reference_v.div(1.0 - 0.95**step).sqrt().add(1e-8)
        )
        reference_parameter.add_(
            update.add(reference_parameter, alpha=0.1), alpha=-1e-3
        )
        optimizer.step()

        state = optimizer.state[parameter]
        assert state["exp_avg"].dtype == torch.float8_e4m3fn
        assert state["exp_avg_sq"].dtype == torch.float8_e4m3fn
        assert torch.allclose(parameter, reference_parameter, atol=2e-6, rtol=2e-6)

        reference_m = state["exp_avg"].float() * state["scale_exp_avg"]
        reference_v = state["exp_avg_sq"].float() * state["scale_exp_avg_sq"]


def test_load_state_dict_does_not_alias_cuda_input():
    parameter = torch.nn.Parameter(torch.randn(257, device="cuda"))
    optimizer = make_optimizer(parameter)
    set_gradient(parameter, 7)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    saved_state = next(iter(saved["state"].values()))
    snapshot = {key: value.clone() for key, value in saved_state.items() if torch.is_tensor(value)}

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = make_optimizer(resumed_parameter)
    resumed.load_state_dict(saved)
    resumed_state = resumed.state[resumed_parameter]
    for key, value in snapshot.items():
        assert resumed_state[key].data_ptr() != saved_state[key].data_ptr()
        assert torch.equal(saved_state[key], value)

    set_gradient(resumed_parameter, 8)
    resumed.step()
    for key, value in snapshot.items():
        assert torch.equal(saved_state[key], value)


def test_noncontiguous_input_uses_contiguous_fp8_storage():
    value = torch.randn(32, 24, device="cuda").t()
    assert not value.is_contiguous()
    state = {}
    init_fp8_state(state, "moment", value, group_size=128)
    quantize_fp8_state_(state, "moment", value, signed=True, group_size=128)
    restored = dequantize_fp8_state(state, "moment", signed=True, group_size=128)

    assert state["moment"].is_contiguous()
    assert restored.shape == value.shape
    assert torch.isfinite(restored).all()
    assert (restored - value).abs().mean() / value.abs().mean() < 0.05


QUANT_EPS = 1e-30


def reference_quantize(value, *, signed, group_size=128, fp8_max=448.0):
    """The audited op-chain the Triton kernels replace, kept as a fixed oracle."""
    flat = value.reshape(-1)
    padding = -flat.numel() % group_size
    if padding:
        flat = torch.cat((flat, flat.new_zeros(padding)))
    groups = flat.view(-1, group_size)

    source = groups if signed else groups.clamp_min(0)
    magnitude = source.abs() if signed else source
    nonzero = magnitude > 0
    absmax = magnitude.max(dim=1).values.clamp_min(QUANT_EPS)
    inf = torch.full_like(magnitude, torch.finfo(torch.float32).max)
    absmin = torch.where(nonzero, magnitude, inf).min(dim=1).values
    absmin = torch.where(
        nonzero.any(dim=1),
        absmin.clamp_min(QUANT_EPS),
        torch.full_like(absmin, QUANT_EPS),
    )

    ratio = (absmax / absmin).clamp_min(1.0 + QUANT_EPS)
    ratio_upper = torch.full_like(ratio, fp8_max * fp8_max / 2.0)
    raw_expansion = (
        torch.floor(
            torch.log2(ratio_upper) / torch.log2(ratio).clamp_min(QUANT_EPS) * 16
        )
        / 16
    )
    expansion = torch.where(
        ratio <= 1.0 + QUANT_EPS,
        torch.ones_like(raw_expansion),
        torch.maximum(raw_expansion, torch.full_like(raw_expansion, 1.0 / 16)),
    )

    sqrt_minmax = (absmax.sqrt() * absmin.sqrt()).clamp_min(QUANT_EPS)
    base = (magnitude / sqrt_minmax.view(-1, 1)).clamp_min(QUANT_EPS)
    normalized = torch.pow(base, expansion.view(-1, 1))
    normalized = torch.where(nonzero, normalized, torch.zeros_like(normalized))
    if signed:
        normalized = torch.sign(source) * normalized
    scale = (
        torch.pow((absmax / sqrt_minmax).clamp_min(QUANT_EPS), expansion) / fp8_max
    ).clamp_min(QUANT_EPS)
    codes = (normalized / scale.view(-1, 1)).reshape(-1)[: value.numel()]
    return codes.reshape(value.shape).to(torch.float8_e4m3fn), scale, expansion, sqrt_minmax


def test_triton_kernels_match_reference_format():
    generator = torch.Generator(device="cuda").manual_seed(20260731)
    values = {
        "moment": torch.randn(4096, generator=generator, device="cuda") * 1e-3,
        "matrix": torch.randn(1536, 1536, generator=generator, device="cuda") * 3e-2,
        "wide_range": torch.exp(
            torch.randn(8192, generator=generator, device="cuda") * 12.0
        ),
        "ragged": torch.randn(383, generator=generator, device="cuda"),
        "sparse": torch.zeros(512, device="cuda").index_fill_(
            0, torch.tensor([5], device="cuda"), 2.5
        ),
    }
    for name, value in values.items():
        for signed in (True, False):
            state = {}
            init_fp8_state(state, name, value, group_size=128)
            quantize_fp8_state_(state, name, value, signed=signed, group_size=128)
            codes, scale, expansion, sqrt_minmax = reference_quantize(
                value, signed=signed
            )

            # The discrete part of the format must be reproduced exactly.
            assert torch.equal(state[f"expand_{name}"], expansion)
            assert state[name].shape == value.shape
            assert state[f"scale_{name}"].shape == expansion.shape

            # CUDA powf differs in its last bit between PyTorch and Triton, so the
            # continuous metadata and the codes it feeds may move by one step.
            for produced, oracle in (
                (state[f"scale_{name}"], scale),
                (state[f"sqrt_minmax_{name}"], sqrt_minmax),
            ):
                assert torch.allclose(produced, oracle, rtol=1e-6, atol=0.0)
            drift = state[name].view(torch.uint8).int() - codes.view(torch.uint8).int()
            assert drift.abs().max() <= 1
            assert (drift != 0).sum() <= max(1, value.numel() // 10000)

            restored = dequantize_fp8_state(
                state, name, signed=signed, group_size=128
            )
            expected = value if signed else value.clamp_min(0)
            span = expected.abs().max().clamp_min(QUANT_EPS)
            assert torch.isfinite(restored).all()
            assert ((restored - expected).norm() / expected.norm().clamp_min(QUANT_EPS)) < 0.05
            assert (restored.abs().max() - span).abs() / span < 0.1


def test_triton_tail_masks_cover_zero_and_singleton_groups():
    cases = {
        "all_zero_127": torch.zeros(127, device="cuda"),
        "all_zero_1025": torch.zeros(1025, device="cuda"),
        "single_tail_129": torch.zeros(129, device="cuda").index_fill_(
            0, torch.tensor([128], device="cuda"), -2.5
        ),
        "single_tail_1025": torch.zeros(1025, device="cuda").index_fill_(
            0, torch.tensor([1024], device="cuda"), 3.0
        ),
    }
    for name, value in cases.items():
        for signed in (True, False):
            state = {}
            init_fp8_state(state, name, value, group_size=128)
            quantize_fp8_state_(state, name, value, signed=signed, group_size=128)
            codes, scale, expand, sqrt_minmax = reference_quantize(value, signed=signed)

            assert torch.equal(state[name], codes)
            assert torch.equal(state[f"expand_{name}"], expand)
            assert torch.allclose(state[f"scale_{name}"], scale, rtol=1e-6, atol=0.0)
            assert torch.allclose(
                state[f"sqrt_minmax_{name}"], sqrt_minmax, rtol=1e-6, atol=0.0
            )
            restored = dequantize_fp8_state(state, name, signed=signed, group_size=128)
            assert torch.isfinite(restored).all()

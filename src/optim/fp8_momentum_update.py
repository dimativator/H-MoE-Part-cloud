"""Fused in-place updates for persistent FP8 first-order optimizer state."""

from __future__ import annotations

from typing import Any, Dict

import torch

from .fp8_state import dequantize_fp8_state, quantize_fp8_state_


try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:  # pragma: no cover - exercised only in CPU-only environments
    triton = None
    tl = None
    libdevice = None


_QUANT_EPS = 1e-30


if triton is not None:

    @triton.jit
    def _fp8_momentum_update_kernel(
        values_ptr,
        scales_ptr,
        gradient_ptr,
        momentum,
        gradient_alpha,
        fp8_max,
        numel,
        GROUP_SIZE: tl.constexpr,
        QUANT_EPS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offsets < numel

        raw = tl.load(values_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scales_ptr + pid).to(tl.float32)
        old = raw * scale
        gradient = tl.load(gradient_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        updated = momentum * old + gradient_alpha * gradient

        absmax = tl.max(tl.where(mask, tl.abs(updated), 0.0), axis=0)
        new_scale = tl.maximum((absmax + QUANT_EPS) / fp8_max, QUANT_EPS)
        quantized = updated / new_scale

        tl.store(values_ptr + offsets, quantized, mask=mask)
        tl.store(scales_ptr + pid, new_scale)


    @triton.jit
    def _fp8_momentum_expand_update_kernel(
        values_ptr,
        scales_ptr,
        expand_ptr,
        sqrt_minmax_ptr,
        gradient_ptr,
        momentum,
        gradient_alpha,
        fp8_max,
        expand_min,
        numel,
        GROUP_SIZE: tl.constexpr,
        QUANT_EPS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offsets < numel
        large_value = tl.full((GROUP_SIZE,), 3.4028235e38, tl.float32)

        raw = tl.load(values_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scales_ptr + pid).to(tl.float32)
        old_expand = tl.maximum(
            tl.load(expand_ptr + pid).to(tl.float32), QUANT_EPS
        )
        old_sqrt_minmax = tl.maximum(
            tl.load(sqrt_minmax_ptr + pid).to(tl.float32), QUANT_EPS
        )
        scaled = raw * scale
        magnitude = tl.abs(scaled)
        restored_magnitude = libdevice.pow(
            tl.maximum(magnitude, QUANT_EPS), 1.0 / old_expand
        ) * old_sqrt_minmax
        old = tl.where(
            magnitude > 0.0,
            tl.where(scaled < 0.0, -restored_magnitude, restored_magnitude),
            0.0,
        )

        gradient = tl.load(gradient_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        updated = momentum * old + gradient_alpha * gradient

        abs_updated = tl.abs(updated)
        nonzero = mask & (abs_updated > 0.0)
        absmax = tl.max(tl.where(mask, abs_updated, 0.0), axis=0)
        absmin = tl.min(tl.where(nonzero, abs_updated, large_value), axis=0)
        nonzero_count = tl.sum(nonzero.to(tl.int32), axis=0)
        absmax = tl.maximum(absmax, QUANT_EPS)
        absmin = tl.where(
            nonzero_count > 0, tl.maximum(absmin, QUANT_EPS), QUANT_EPS
        )

        ratio = tl.maximum(absmax / absmin, 1.0 + QUANT_EPS)
        ratio_upper = fp8_max * fp8_max / 2.0
        log_ratio = tl.log2(ratio)
        safe_log_ratio = tl.maximum(log_ratio, QUANT_EPS)
        raw_expand = (
            tl.floor((tl.log2(ratio_upper) / safe_log_ratio) * expand_min)
            / expand_min
        )
        min_expand = 1.0 / expand_min
        new_expand = tl.where(
            ratio <= 1.0 + QUANT_EPS,
            1.0,
            tl.maximum(raw_expand, min_expand),
        )

        new_sqrt_minmax = tl.maximum(tl.sqrt(absmax) * tl.sqrt(absmin), QUANT_EPS)
        normalized_base = tl.maximum(abs_updated / new_sqrt_minmax, QUANT_EPS)
        normalized_magnitude = libdevice.pow(normalized_base, new_expand)
        normalized = tl.where(
            nonzero,
            tl.where(updated < 0.0, -normalized_magnitude, normalized_magnitude),
            0.0,
        )
        scale_base = tl.maximum(absmax / new_sqrt_minmax, QUANT_EPS)
        new_scale = tl.maximum(
            libdevice.pow(scale_base, new_expand) / fp8_max, QUANT_EPS
        )
        quantized = normalized / new_scale

        tl.store(values_ptr + offsets, quantized, mask=mask)
        tl.store(scales_ptr + pid, new_scale)
        tl.store(expand_ptr + pid, new_expand)
        tl.store(sqrt_minmax_ptr + pid, new_sqrt_minmax)


def update_fp8_momentum_(
    state: Dict[str, Any],
    prefix: str,
    gradient: torch.Tensor,
    qargs,
    *,
    momentum: float,
    gradient_alpha: float,
) -> None:
    """Apply ``state = momentum * state + gradient_alpha * gradient`` in place.

    CUDA tensors use one Triton program per quantization group. The program
    performs expansion-aware decode, EMA update, and requantization without a
    dense persistent-state tensor. CPU execution intentionally uses the current
    PyTorch codec as a reference fallback.
    """
    values = state[prefix]
    scales = state[f"scale_{prefix}"]
    if values.shape != gradient.shape:
        raise ValueError(
            f"gradient shape {tuple(gradient.shape)} != state shape {tuple(values.shape)}"
        )
    if gradient.is_sparse:
        raise ValueError("Fused FP8 momentum update does not support sparse gradients.")

    if not values.is_cuda or triton is None or qargs.qgroup_size != 128:
        updated = dequantize_fp8_state(
            state, prefix, qargs, signed=True
        ).mul_(momentum).add_(gradient, alpha=gradient_alpha)
        quantize_fp8_state_(state, prefix, updated, qargs, signed=True)
        return

    if not gradient.is_cuda or gradient.device != values.device:
        raise ValueError("FP8 momentum state and gradient must be on the same CUDA device.")
    if values.numel() == 0:
        return

    expected_groups = triton.cdiv(values.numel(), qargs.qgroup_size)
    if scales.numel() != expected_groups:
        raise ValueError("Scale tensor does not match state size and qgroup size.")

    grid = (expected_groups,)
    fp8_max = float(torch.finfo(values.dtype).max)
    expand = state.get(f"expand_{prefix}")
    sqrt_minmax = state.get(f"sqrt_minmax_{prefix}")
    if (expand is None) != (sqrt_minmax is None):
        raise ValueError("FP8 expansion requires both expand and sqrt_minmax metadata.")

    if expand is None:
        _fp8_momentum_update_kernel[grid](
            values.view(-1),
            scales.view(-1),
            gradient.contiguous().view(-1),
            momentum,
            gradient_alpha,
            fp8_max,
            values.numel(),
            GROUP_SIZE=qargs.qgroup_size,
            QUANT_EPS=_QUANT_EPS,
        )
        return

    if qargs.expand_min <= 0:
        raise ValueError(f"expand_min must be > 0, got {qargs.expand_min}.")
    if any(t.numel() != expected_groups for t in (expand, sqrt_minmax)):
        raise ValueError("Expansion metadata does not match state size and qgroup size.")
    _fp8_momentum_expand_update_kernel[grid](
        values.view(-1),
        scales.view(-1),
        expand.view(-1),
        sqrt_minmax.view(-1),
        gradient.contiguous().view(-1),
        momentum,
        gradient_alpha,
        fp8_max,
        float(qargs.expand_min),
        values.numel(),
        GROUP_SIZE=qargs.qgroup_size,
        QUANT_EPS=_QUANT_EPS,
    )

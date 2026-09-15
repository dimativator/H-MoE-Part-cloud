"""Fused update of a persistent FP8 Muon momentum shard."""

from __future__ import annotations

from typing import Any, Dict

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from stage4.fp8_optimizer_states import (
    dequantize_fp8_state,
    quantize_fp8_state_,
)


_QUANT_EPS = tl.constexpr(1e-30)


@triton.jit
def _fp8_momentum_update_kernel(
    values_ptr,
    scales_ptr,
    expand_ptr,
    sqrt_minmax_ptr,
    gradient_ptr,
    momentum,
    gradient_alpha,
    fp8_max,
    numel,
    GROUP_SIZE: tl.constexpr,
):
    group = tl.program_id(0)
    offsets = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = offsets < numel
    large_value = tl.full((GROUP_SIZE,), 3.4028235e38, tl.float32)

    raw = tl.load(values_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scales_ptr + group).to(tl.float32)
    old_expand = tl.maximum(tl.load(expand_ptr + group).to(tl.float32), _QUANT_EPS)
    old_minimum = tl.maximum(
        tl.load(sqrt_minmax_ptr + group).to(tl.float32), _QUANT_EPS
    )
    scaled = raw * scale
    magnitude = tl.abs(scaled)
    restored_magnitude = (
        libdevice.pow(tl.maximum(magnitude, _QUANT_EPS), 1.0 / old_expand) * old_minimum
    )
    old = tl.where(
        magnitude > 0.0,
        tl.where(scaled < 0.0, -restored_magnitude, restored_magnitude),
        0.0,
    )

    gradient = tl.load(gradient_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    updated = momentum * old + gradient_alpha * gradient

    abs_updated = tl.abs(updated)
    nonzero = mask & (abs_updated > 0.0)
    absmax = tl.maximum(tl.max(tl.where(mask, abs_updated, 0.0), axis=0), _QUANT_EPS)
    absmin = tl.min(tl.where(nonzero, abs_updated, large_value), axis=0)
    absmin = tl.where(
        tl.sum(nonzero.to(tl.int32), axis=0) > 0,
        tl.maximum(absmin, _QUANT_EPS),
        _QUANT_EPS,
    )

    ratio = tl.maximum(absmax / absmin, 1.0 + _QUANT_EPS)
    ratio_upper = fp8_max * fp8_max / 2.0
    raw_expand = (
        tl.floor((tl.log2(ratio_upper) / tl.maximum(tl.log2(ratio), _QUANT_EPS)) * 16.0)
        / 16.0
    )
    new_expand = tl.where(
        ratio <= 1.0 + _QUANT_EPS,
        1.0,
        tl.maximum(raw_expand, 1.0 / 16.0),
    )
    new_minimum = tl.maximum(tl.sqrt(absmax) * tl.sqrt(absmin), _QUANT_EPS)
    normalized = libdevice.pow(
        tl.maximum(abs_updated / new_minimum, _QUANT_EPS), new_expand
    )
    normalized = tl.where(
        nonzero,
        tl.where(updated < 0.0, -normalized, normalized),
        0.0,
    )
    new_scale = tl.maximum(
        libdevice.pow(tl.maximum(absmax / new_minimum, _QUANT_EPS), new_expand)
        / fp8_max,
        _QUANT_EPS,
    )

    tl.store(values_ptr + offsets, normalized / new_scale, mask=mask)
    tl.store(scales_ptr + group, new_scale)
    tl.store(expand_ptr + group, new_expand)
    tl.store(sqrt_minmax_ptr + group, new_minimum)


def update_fp8_momentum_(
    state: Dict[str, Any],
    prefix: str,
    gradient: torch.Tensor,
    *,
    momentum: float,
    gradient_alpha: float,
    group_size: int = 128,
) -> None:
    """Compute ``m = momentum * m + gradient_alpha * gradient`` in-place.

    The CUDA path decodes, updates, and requantizes one quantization group in a
    single Triton program. It never materializes a dense persistent momentum.
    """
    values = state[prefix]
    if values.shape != gradient.shape:
        raise ValueError(
            f"gradient shape {tuple(gradient.shape)} != state shape {tuple(values.shape)}"
        )
    if values.numel() == 0:
        return
    if not values.is_cuda or group_size != 128:
        updated = (
            dequantize_fp8_state(state, prefix, signed=True, group_size=group_size)
            .mul_(momentum)
            .add_(gradient, alpha=gradient_alpha)
        )
        quantize_fp8_state_(state, prefix, updated, signed=True, group_size=group_size)
        return

    groups = triton.cdiv(values.numel(), group_size)
    metadata = (
        state[f"scale_{prefix}"],
        state[f"expand_{prefix}"],
        state[f"sqrt_minmax_{prefix}"],
    )
    if any(item.numel() != groups for item in metadata):
        raise ValueError("FP8 metadata does not match the momentum shard")
    _fp8_momentum_update_kernel[(groups,)](
        values.view(-1),
        metadata[0],
        metadata[1],
        metadata[2],
        gradient.contiguous().view(-1),
        momentum,
        gradient_alpha,
        float(torch.finfo(values.dtype).max),
        values.numel(),
        GROUP_SIZE=group_size,
    )

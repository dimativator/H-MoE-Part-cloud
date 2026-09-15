"""Fused reconstruction of Muon inputs from gathered FP8 state shards."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


_QUANT_EPS = tl.constexpr(1e-30)


@triton.jit
def _decode_value(
    offsets,
    mask,
    values_ptr,
    scales_ptr,
    expand_ptr,
    sqrt_minmax_ptr,
    gradient_ptr,
    local_numel,
    values_rank_stride,
    scales_rank_stride,
    metadata_rank_stride,
    gradient_alpha,
    GROUP_SIZE: tl.constexpr,
    HAS_GRADIENT: tl.constexpr,
):
    rank = offsets // local_numel
    local_offsets = offsets - rank * local_numel
    group = local_offsets // GROUP_SIZE
    values = tl.load(
        values_ptr + rank * values_rank_stride + local_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scales = tl.load(
        scales_ptr + rank * scales_rank_stride + group, mask=mask, other=1.0
    ).to(tl.float32)
    expansion = tl.maximum(
        tl.load(
            expand_ptr + rank * metadata_rank_stride + group,
            mask=mask,
            other=1.0,
        ),
        _QUANT_EPS,
    ).to(tl.float32)
    minimum = tl.maximum(
        tl.load(
            sqrt_minmax_ptr + rank * metadata_rank_stride + group,
            mask=mask,
            other=1.0,
        ),
        _QUANT_EPS,
    ).to(tl.float32)
    raw = values * scales
    magnitude = tl.abs(raw)
    restored_abs = (
        tl.exp2(
            tl.log2(tl.maximum(magnitude, _QUANT_EPS))
            / tl.maximum(expansion, _QUANT_EPS)
        )
        * minimum
    )
    restored = tl.where(
        magnitude > 0.0,
        tl.where(raw < 0.0, -restored_abs, restored_abs),
        0.0,
    )
    if HAS_GRADIENT:
        gradient = tl.load(gradient_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        restored += gradient_alpha * gradient
    return restored


@triton.jit
def _decode_and_add_kernel(
    output_ptr,
    values_ptr,
    scales_ptr,
    expand_ptr,
    sqrt_minmax_ptr,
    gradient_ptr,
    output_numel,
    local_numel,
    values_rank_stride,
    scales_rank_stride,
    metadata_rank_stride,
    gradient_alpha,
    GROUP_SIZE: tl.constexpr,
    HAS_GRADIENT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_numel
    restored = _decode_value(
        offsets,
        mask,
        values_ptr,
        scales_ptr,
        expand_ptr,
        sqrt_minmax_ptr,
        gradient_ptr,
        local_numel,
        values_rank_stride,
        scales_rank_stride,
        metadata_rank_stride,
        gradient_alpha,
        GROUP_SIZE,
        HAS_GRADIENT,
    )
    tl.store(output_ptr + offsets, restored, mask=mask)


@triton.jit
def _decode_norm_kernel(
    norm_sq_ptr,
    values_ptr,
    scales_ptr,
    expand_ptr,
    sqrt_minmax_ptr,
    gradient_ptr,
    output_numel,
    local_numel,
    values_rank_stride,
    scales_rank_stride,
    metadata_rank_stride,
    gradient_alpha,
    GROUP_SIZE: tl.constexpr,
    HAS_GRADIENT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_numel
    restored = _decode_value(
        offsets,
        mask,
        values_ptr,
        scales_ptr,
        expand_ptr,
        sqrt_minmax_ptr,
        gradient_ptr,
        local_numel,
        values_rank_stride,
        scales_rank_stride,
        metadata_rank_stride,
        gradient_alpha,
        GROUP_SIZE,
        HAS_GRADIENT,
    )
    tl.atomic_add(norm_sq_ptr, tl.sum(restored * restored, axis=0))


@triton.jit
def _decode_normalize_kernel(
    output_ptr,
    norm_sq_ptr,
    values_ptr,
    scales_ptr,
    expand_ptr,
    sqrt_minmax_ptr,
    gradient_ptr,
    output_numel,
    local_numel,
    values_rank_stride,
    scales_rank_stride,
    metadata_rank_stride,
    gradient_alpha,
    rows,
    columns,
    eps,
    GROUP_SIZE: tl.constexpr,
    HAS_GRADIENT: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_numel
    restored = _decode_value(
        offsets,
        mask,
        values_ptr,
        scales_ptr,
        expand_ptr,
        sqrt_minmax_ptr,
        gradient_ptr,
        local_numel,
        values_rank_stride,
        scales_rank_stride,
        metadata_rank_stride,
        gradient_alpha,
        GROUP_SIZE,
        HAS_GRADIENT,
    )
    restored /= tl.maximum(tl.sqrt(tl.load(norm_sq_ptr)), eps)
    if TRANSPOSE:
        row = offsets // columns
        column = offsets - row * columns
        output_offsets = column * rows + row
    else:
        output_offsets = offsets
    tl.store(output_ptr + output_offsets, restored, mask=mask)


def _torch_decode_and_add(
    values: torch.Tensor,
    scales: torch.Tensor,
    expand: torch.Tensor,
    sqrt_minmax: torch.Tensor,
    group_size: int,
    output_shape: torch.Size,
    gradient: Optional[torch.Tensor],
    gradient_alpha: float,
) -> torch.Tensor:
    world_size, local_numel = values.shape
    padded_numel = scales.shape[1] * group_size
    values_float = values.float()
    if padded_numel != local_numel:
        values_float = torch.cat(
            (
                values_float,
                values_float.new_zeros(world_size, padded_numel - local_numel),
            ),
            dim=1,
        )
    raw = values_float.view(world_size, scales.shape[1], group_size)
    raw *= scales.float().unsqueeze(-1)
    magnitude = raw.abs()
    restored = torch.pow(
        magnitude.clamp_min(1e-30),
        1.0 / expand.float().unsqueeze(-1).clamp_min(1e-30),
    ) * sqrt_minmax.float().unsqueeze(-1).clamp_min(1e-30)
    restored = torch.where(magnitude > 0, raw.sign() * restored, 0.0)
    output_numel = output_shape.numel()
    output = restored.reshape(world_size, padded_numel)[:, :local_numel]
    output = output.reshape(-1)[:output_numel]
    if gradient is not None:
        output = output + gradient.reshape(-1).float() * gradient_alpha
    return output.reshape(output_shape)


def dequantize_fp8_state_and_add(
    values: torch.Tensor,
    scales: torch.Tensor,
    expand: torch.Tensor,
    sqrt_minmax: torch.Tensor,
    *,
    group_size: int,
    output_shape: torch.Size,
    gradient: Optional[torch.Tensor] = None,
    gradient_alpha: float = 0.0,
) -> torch.Tensor:
    """Decode rank-major packets directly into the full FP32 NS input."""
    tensors = (values, scales, expand, sqrt_minmax)
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("FP8 wire fields must be rank-major matrices")
    if len({tensor.shape[0] for tensor in tensors}) != 1:
        raise ValueError("FP8 wire fields have different world sizes")
    if gradient is not None and tuple(gradient.shape) != tuple(output_shape):
        raise ValueError("gradient and decoded output shapes differ")
    if not values.is_cuda:
        return _torch_decode_and_add(
            values,
            scales,
            expand,
            sqrt_minmax,
            group_size,
            output_shape,
            gradient,
            gradient_alpha,
        )

    output = torch.empty(output_shape, dtype=torch.float32, device=values.device)
    gradient_input = output if gradient is None else gradient.contiguous()
    _decode_and_add_kernel[(triton.cdiv(output.numel(), 256),)](
        output,
        values,
        scales,
        expand,
        sqrt_minmax,
        gradient_input,
        output.numel(),
        values.shape[1],
        values.stride(0),
        scales.stride(0),
        expand.stride(0),
        gradient_alpha,
        GROUP_SIZE=group_size,
        HAS_GRADIENT=gradient is not None,
        BLOCK_SIZE=256,
    )
    return output


def prepare_fp8_state_ns_input(
    values: torch.Tensor,
    scales: torch.Tensor,
    expand: torch.Tensor,
    sqrt_minmax: torch.Tensor,
    *,
    group_size: int,
    output_shape: torch.Size,
    gradient: Optional[torch.Tensor] = None,
    gradient_alpha: float = 0.0,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, bool]:
    """Decode, add Nesterov gradient, normalize and orient a BF16 NS input."""
    tensors = (values, scales, expand, sqrt_minmax)
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("FP8 wire fields must be rank-major matrices")
    if len({tensor.shape[0] for tensor in tensors}) != 1:
        raise ValueError("FP8 wire fields have different world sizes")
    if gradient is not None and tuple(gradient.shape) != tuple(output_shape):
        raise ValueError("gradient and decoded output shapes differ")
    transpose = output_shape[0] > output_shape[1]
    if not values.is_cuda:
        restored = _torch_decode_and_add(
            values,
            scales,
            expand,
            sqrt_minmax,
            group_size,
            output_shape,
            gradient,
            gradient_alpha,
        )
        restored = torch.nn.functional.normalize(
            restored, p=2, dim=(-2, -1), eps=eps
        )
        if transpose:
            restored = restored.mT
        return restored.to(torch.bfloat16), transpose

    output_numel = output_shape.numel()
    norm_sq = torch.zeros((), dtype=torch.float32, device=values.device)
    gradient_input = norm_sq if gradient is None else gradient.contiguous()
    grid = (triton.cdiv(output_numel, 1024),)
    common = (
        values,
        scales,
        expand,
        sqrt_minmax,
        gradient_input,
        output_numel,
        values.shape[1],
        values.stride(0),
        scales.stride(0),
        expand.stride(0),
        gradient_alpha,
    )
    _decode_norm_kernel[grid](
        norm_sq,
        *common,
        GROUP_SIZE=group_size,
        HAS_GRADIENT=gradient is not None,
        BLOCK_SIZE=1024,
    )
    prepared_shape = (
        torch.Size((output_shape[1], output_shape[0])) if transpose else output_shape
    )
    output = torch.empty(prepared_shape, dtype=torch.bfloat16, device=values.device)
    _decode_normalize_kernel[grid](
        output,
        norm_sq,
        *common,
        output_shape[0],
        output_shape[1],
        eps,
        GROUP_SIZE=group_size,
        HAS_GRADIENT=gradient is not None,
        TRANSPOSE=transpose,
        BLOCK_SIZE=1024,
    )
    return output, transpose

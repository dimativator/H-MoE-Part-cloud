"""Fused reconstruction of Muon inputs from gathered FP8 state shards."""

from __future__ import annotations

from typing import Optional

import torch


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in CPU-only environments
    triton = None
    tl = None


_QUANT_EPS = 1e-30


if triton is not None:

    @triton.jit
    def _dequantize_fp8_state_and_add_kernel(
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
        QGROUP_SIZE: tl.constexpr,
        QUANT_EPS: tl.constexpr,
        HAS_EXPANSION: tl.constexpr,
        HAS_GRADIENT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < output_numel
        rank = offsets // local_numel
        local_offsets = offsets - rank * local_numel
        group = local_offsets // QGROUP_SIZE

        values = tl.load(
            values_ptr + rank * values_rank_stride + local_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scales = tl.load(
            scales_ptr + rank * scales_rank_stride + group,
            mask=mask,
            other=1.0,
        ).to(tl.float32)
        raw = values * scales

        if HAS_EXPANSION:
            expand = tl.load(
                expand_ptr + rank * metadata_rank_stride + group,
                mask=mask,
                other=1.0,
            ).to(tl.float32)
            sqrt_minmax = tl.load(
                sqrt_minmax_ptr + rank * metadata_rank_stride + group,
                mask=mask,
                other=1.0,
            ).to(tl.float32)
            abs_raw = tl.abs(raw)
            restored_abs = tl.exp2(
                tl.log2(tl.maximum(abs_raw, QUANT_EPS))
                / tl.maximum(expand, QUANT_EPS)
            ) * sqrt_minmax
            restored = tl.where(
                abs_raw > 0.0,
                tl.where(raw < 0.0, -restored_abs, restored_abs),
                0.0,
            )
        else:
            restored = raw

        if HAS_GRADIENT:
            gradient = tl.load(gradient_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            restored += gradient_alpha * gradient

        tl.store(output_ptr + offsets, restored, mask=mask)


def _torch_dequantize_fp8_state_and_add(
    values: torch.Tensor,
    scales: torch.Tensor,
    qgroup_size: int,
    *,
    output_shape: torch.Size,
    expand: Optional[torch.Tensor],
    sqrt_minmax: Optional[torch.Tensor],
    gradient: Optional[torch.Tensor],
    gradient_alpha: float,
) -> torch.Tensor:
    world_size, local_numel = values.shape
    padded_numel = scales.shape[1] * qgroup_size
    values_float = values.to(torch.float32)
    if padded_numel != local_numel:
        padding = values_float.new_zeros((world_size, padded_numel - local_numel))
        values_float = torch.cat((values_float, padding), dim=1)

    groups = values_float.view(world_size, scales.shape[1], qgroup_size)
    raw = groups * scales.to(torch.float32).unsqueeze(-1)
    if expand is not None:
        assert sqrt_minmax is not None
        expansion = expand.to(torch.float32).unsqueeze(-1).clamp_min(_QUANT_EPS)
        magnitude = raw.abs()
        restored = torch.pow(
            magnitude.clamp_min(_QUANT_EPS), 1.0 / expansion
        ) * sqrt_minmax.to(torch.float32).unsqueeze(-1).clamp_min(_QUANT_EPS)
        restored = torch.where(magnitude > 0, restored, torch.zeros_like(restored))
        restored = torch.sign(raw) * restored
    else:
        restored = raw

    output_numel = 1
    for dimension in output_shape:
        output_numel *= dimension
    output = restored.view(world_size, padded_numel)[:, :local_numel]
    output = output.reshape(-1)[:output_numel]
    if gradient is not None:
        output = output + gradient.reshape(-1).to(torch.float32) * gradient_alpha
    return output.reshape(output_shape)


def dequantize_fp8_state_and_add(
    values: torch.Tensor,
    scales: torch.Tensor,
    qgroup_size: int,
    *,
    output_shape: torch.Size,
    expand: Optional[torch.Tensor] = None,
    sqrt_minmax: Optional[torch.Tensor] = None,
    gradient: Optional[torch.Tensor] = None,
    gradient_alpha: float = 0.0,
) -> torch.Tensor:
    """Decode gathered row shards and optionally form a Muon Nesterov input.

    ``values`` and metadata use shape ``[world_size, local_numel_or_groups]``.
    Their first dimension may be strided because each rank's fields can be views
    into one packed NCCL packet.  On CUDA, a single Triton kernel reads those
    strided fields and writes the final dense FP32 matrix directly.
    """
    if values.ndim != 2 or scales.ndim != 2:
        raise ValueError("FP8 wire values and scales must be rank-major matrices")
    if values.shape[0] != scales.shape[0]:
        raise ValueError("FP8 wire values and scales have different world sizes")
    if (expand is None) != (sqrt_minmax is None):
        raise ValueError("FP8 expansion requires both expand and sqrt_minmax")
    if gradient is not None and tuple(gradient.shape) != tuple(output_shape):
        raise ValueError(
            f"gradient shape {tuple(gradient.shape)} != output shape {tuple(output_shape)}"
        )

    if not values.is_cuda or triton is None:
        return _torch_dequantize_fp8_state_and_add(
            values,
            scales,
            qgroup_size,
            output_shape=output_shape,
            expand=expand,
            sqrt_minmax=sqrt_minmax,
            gradient=gradient,
            gradient_alpha=gradient_alpha,
        )

    output = torch.empty(output_shape, dtype=torch.float32, device=values.device)
    gradient_input = output if gradient is None else gradient.contiguous()
    expand_input = scales if expand is None else expand
    sqrt_minmax_input = scales if sqrt_minmax is None else sqrt_minmax
    grid = lambda meta: (triton.cdiv(output.numel(), meta["BLOCK_SIZE"]),)
    _dequantize_fp8_state_and_add_kernel[grid](
        output,
        values,
        scales,
        expand_input,
        sqrt_minmax_input,
        gradient_input,
        output.numel(),
        values.shape[1],
        values.stride(0),
        scales.stride(0),
        expand_input.stride(0),
        gradient_alpha,
        QGROUP_SIZE=qgroup_size,
        QUANT_EPS=_QUANT_EPS,
        HAS_EXPANSION=expand is not None,
        HAS_GRADIENT=gradient is not None,
        BLOCK_SIZE=256,
    )
    return output

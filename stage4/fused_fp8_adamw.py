"""Memory-efficient AdamW with per-tensor E4M3 optimizer states.

The Transformer Engine 2.16 precision-aware Adam path materializes FP32 copies
of both moments before the Adam update and requantizes them afterwards.  This
implementation keeps the same storage semantics, FP32 AdamW arithmetic, and
per-tensor current scaling, but recomputes the inexpensive moment update in a
second pass so no full-size FP32 moment buffers are needed.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from stage4.fp8_optimizer_states import FP8StateDictMixin


FP8_MAX = 448.0
MAX_TENSORS_PER_BATCH = 128
BLOCK_SIZE = 1024
NUM_WARPS = 8


@triton.jit
def _locate_tensor(chunk_id, chunk_ends, num_tensors, BLOCK: tl.constexpr):
    """Map a flattened chunk id to a tensor and an element offset."""
    owner = tl.zeros((), tl.int32)
    for shift in tl.static_range(7):
        increment = 64 // (1 << shift)
        candidate = owner + increment
        valid = candidate < num_tensors
        preceding_end = tl.load(
            chunk_ends + candidate - 1,
            mask=valid,
            other=0,
        )
        owner += tl.where(valid & (chunk_id >= preceding_end), increment, 0)
    first_chunk = tl.load(chunk_ends + owner - 1, mask=owner > 0, other=0)
    local_chunk = chunk_id - first_chunk
    offsets = local_chunk * BLOCK + tl.arange(0, BLOCK)
    return owner, local_chunk, offsets


@triton.jit
def _adamw_update_amax_kernel(
    param_ptrs,
    grad_ptrs,
    exp_avg_ptrs,
    exp_avg_sq_ptrs,
    exp_avg_scale_ptrs,
    exp_avg_sq_scale_ptrs,
    numels,
    chunk_ends,
    amax,
    num_tensors,
    beta1,
    beta2,
    beta1_correction,
    beta2_correction,
    learning_rate,
    epsilon,
    weight_decay,
    BLOCK: tl.constexpr,
):
    chunk_id = tl.program_id(0)
    owner, local_chunk, offsets = _locate_tensor(
        chunk_id, chunk_ends, num_tensors, BLOCK
    )
    numel = tl.load(numels + owner)
    mask = offsets < numel

    param = tl.load(param_ptrs + owner).to(tl.pointer_type(tl.float32))
    grad = tl.load(grad_ptrs + owner).to(tl.pointer_type(tl.float32))
    exp_avg = tl.load(exp_avg_ptrs + owner).to(tl.pointer_type(tl.float8e4nv))
    exp_avg_sq = tl.load(exp_avg_sq_ptrs + owner).to(tl.pointer_type(tl.float8e4nv))
    exp_avg_scale = tl.load(exp_avg_scale_ptrs + owner).to(tl.pointer_type(tl.float32))
    exp_avg_sq_scale = tl.load(exp_avg_sq_scale_ptrs + owner).to(
        tl.pointer_type(tl.float32)
    )

    p = tl.load(param + offsets, mask=mask, other=0.0)
    g = tl.load(grad + offsets, mask=mask, other=0.0)
    m = tl.load(exp_avg + offsets, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(exp_avg_sq + offsets, mask=mask, other=0.0).to(tl.float32)
    old_scale_m = tl.load(exp_avg_scale)
    old_scale_v = tl.load(exp_avg_sq_scale)
    m *= old_scale_m
    v *= old_scale_v

    new_m = beta1 * m + (1.0 - beta1) * g
    new_v = beta2 * v + (1.0 - beta2) * g * g
    unbiased_m = libdevice.div_rn(new_m, beta1_correction)
    unbiased_v = libdevice.div_rn(new_v, beta2_correction)
    denominator = libdevice.sqrt_rn(unbiased_v) + epsilon
    update = libdevice.div_rn(unbiased_m, denominator) + weight_decay * p
    new_p = p - learning_rate * update
    tl.store(param + offsets, new_p, mask=mask)

    block_amax_m = tl.max(tl.where(mask, tl.abs(new_m), 0.0), axis=0)
    block_amax_v = tl.max(tl.where(mask, tl.abs(new_v), 0.0), axis=0)
    tl.atomic_max(amax + owner, block_amax_m)
    tl.atomic_max(amax + num_tensors + owner, block_amax_v)
    tl.store(amax + 2 * num_tensors + owner, old_scale_m, mask=local_chunk == 0)
    tl.store(amax + 3 * num_tensors + owner, old_scale_v, mask=local_chunk == 0)


@triton.jit
def _set_scales_kernel(
    amax,
    exp_avg_scale_ptrs,
    exp_avg_sq_scale_ptrs,
    num_tensors,
    fp8_max,
):
    owner = tl.program_id(0)
    if owner < num_tensors:
        exp_avg_scale = tl.load(exp_avg_scale_ptrs + owner).to(
            tl.pointer_type(tl.float32)
        )
        exp_avg_sq_scale = tl.load(exp_avg_sq_scale_ptrs + owner).to(
            tl.pointer_type(tl.float32)
        )
        tl.store(exp_avg_scale, libdevice.div_rn(tl.load(amax + owner), fp8_max))
        tl.store(
            exp_avg_sq_scale,
            libdevice.div_rn(tl.load(amax + num_tensors + owner), fp8_max),
        )


@triton.jit
def _adamw_encode_states_kernel(
    grad_ptrs,
    exp_avg_ptrs,
    exp_avg_sq_ptrs,
    exp_avg_scale_ptrs,
    exp_avg_sq_scale_ptrs,
    numels,
    chunk_ends,
    old_scales,
    num_tensors,
    beta1,
    beta2,
    fp8_max,
    BLOCK: tl.constexpr,
):
    chunk_id = tl.program_id(0)
    owner, _, offsets = _locate_tensor(chunk_id, chunk_ends, num_tensors, BLOCK)
    numel = tl.load(numels + owner)
    mask = offsets < numel

    grad = tl.load(grad_ptrs + owner).to(tl.pointer_type(tl.float32))
    exp_avg = tl.load(exp_avg_ptrs + owner).to(tl.pointer_type(tl.float8e4nv))
    exp_avg_sq = tl.load(exp_avg_sq_ptrs + owner).to(tl.pointer_type(tl.float8e4nv))
    exp_avg_scale = tl.load(exp_avg_scale_ptrs + owner).to(tl.pointer_type(tl.float32))
    exp_avg_sq_scale = tl.load(exp_avg_sq_scale_ptrs + owner).to(
        tl.pointer_type(tl.float32)
    )

    old_scale_m = tl.load(old_scales + owner)
    old_scale_v = tl.load(old_scales + num_tensors + owner)
    scale_m = tl.load(exp_avg_scale)
    scale_v = tl.load(exp_avg_sq_scale)
    g = tl.load(grad + offsets, mask=mask, other=0.0)
    m = (
        tl.load(exp_avg + offsets, mask=mask, other=0.0).to(tl.float32)
        * old_scale_m
    )
    v = (
        tl.load(exp_avg_sq + offsets, mask=mask, other=0.0).to(tl.float32)
        * old_scale_v
    )
    new_m = beta1 * m + (1.0 - beta1) * g
    new_v = beta2 * v + (1.0 - beta2) * g * g

    quantized_m = tl.where(scale_m > 0.0, libdevice.div_rn(new_m, scale_m), 0.0)
    quantized_v = tl.where(scale_v > 0.0, libdevice.div_rn(new_v, scale_v), 0.0)
    quantized_m = tl.maximum(tl.minimum(quantized_m, fp8_max), -fp8_max)
    quantized_v = tl.maximum(tl.minimum(quantized_v, fp8_max), 0.0)
    tl.store(exp_avg + offsets, quantized_m.to(tl.float8e4nv), mask=mask)
    tl.store(exp_avg_sq + offsets, quantized_v.to(tl.float8e4nv), mask=mask)


def _pointer_table(tensors: Iterable[torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        dtype=torch.int64,
        device=device,
    )


class _RuntimeBatch:
    def __init__(self, entries: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]):
        device = entries[0][0].device
        numels = [parameter.numel() for parameter, _, _ in entries]
        chunks = [triton.cdiv(numel, BLOCK_SIZE) for numel in numels]
        chunk_ends = []
        total_chunks = 0
        for count in chunks:
            total_chunks += count
            chunk_ends.append(total_chunks)

        self.num_tensors = len(entries)
        self.total_chunks = total_chunks
        self.param_ptrs = _pointer_table((entry[0] for entry in entries), device)
        self.grad_ptrs = _pointer_table((entry[1] for entry in entries), device)
        self.exp_avg_ptrs = _pointer_table(
            (entry[2]["exp_avg"] for entry in entries), device
        )
        self.exp_avg_sq_ptrs = _pointer_table(
            (entry[2]["exp_avg_sq"] for entry in entries), device
        )
        self.exp_avg_scale_ptrs = _pointer_table(
            (entry[2]["scale_exp_avg"] for entry in entries), device
        )
        self.exp_avg_sq_scale_ptrs = _pointer_table(
            (entry[2]["scale_exp_avg_sq"] for entry in entries), device
        )
        self.numels = torch.tensor(numels, dtype=torch.int64, device=device)
        self.chunk_ends = torch.tensor(chunk_ends, dtype=torch.int32, device=device)
        # First half stores the two new amax vectors. The second half preserves
        # the old scales for the recomputation pass after state scales change.
        self.amax = torch.empty(4 * self.num_tensors, dtype=torch.float32, device=device)


def _quantize_initial_state(value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    value = value.float()
    absmax = value.abs().amax().reshape(1)
    scale = absmax / FP8_MAX
    normalized = torch.where(scale > 0, value / scale, torch.zeros_like(value))
    data = normalized.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return data, scale


def make_fused_fp8_adamw(base_class):
    """Wrap TE FusedAdam with fused, storage-only FP8 moments."""

    class FusedFP8StateAdamW(FP8StateDictMixin, base_class):
        def __init__(self, *args, **kwargs):
            if kwargs.get("capturable", False):
                raise ValueError("fused FP8 AdamW does not support capturable=True")
            if kwargs.get("master_weights", False):
                raise ValueError(
                    "fused FP8 AdamW expects MCore to own FP32 master parameters"
                )
            if kwargs.get("use_decoupled_grad", False):
                raise ValueError("fused FP8 AdamW does not support decoupled gradients")
            if kwargs.get("adam_w_mode", True) is not True:
                raise ValueError("fused FP8 optimizer-state path supports AdamW only")
            kwargs["exp_avg_dtype"] = torch.float32
            kwargs["exp_avg_sq_dtype"] = torch.float32
            super().__init__(*args, **kwargs)
            self._fp8_runtime_cache = {}

        @staticmethod
        def _ensure_state(parameter: torch.Tensor, state: Dict[str, Any]) -> None:
            for name in ("exp_avg", "exp_avg_sq"):
                scale_name = f"scale_{name}"
                value = state.get(name)
                if value is None:
                    state[name] = torch.zeros_like(
                        parameter, dtype=torch.float8_e4m3fn
                    )
                    state[scale_name] = torch.ones(
                        1, dtype=torch.float32, device=parameter.device
                    )
                    continue
                if value.dtype == torch.float8_e4m3fn:
                    if scale_name not in state:
                        raise RuntimeError(f"missing {scale_name} for FP8 AdamW state")
                    continue
                data, scale = _quantize_initial_state(value)
                state[name] = data
                state[scale_name] = scale

        def initialize_state(self, parameter, store_param_remainders=False):
            if store_param_remainders:
                raise ValueError("parameter remainders are not supported")
            self._ensure_state(parameter, self.state[parameter])

        def get_unscaled_state(self, parameter, state_name, skip_unscale=False):
            if state_name in ("exp_avg", "exp_avg_sq"):
                if skip_unscale:
                    raise ValueError("FP8 moments cannot skip dequantization")
                state = self.state[parameter]
                return state[state_name].float() * state[f"scale_{state_name}"]
            return super().get_unscaled_state(parameter, state_name, skip_unscale)

        def load_state_dict(self, state_dict):
            result = super().load_state_dict(state_dict)
            self._fp8_runtime_cache.clear()
            return result

        def _runtime_batches(self, group_index, entries):
            signature = tuple(
                (
                    parameter.data_ptr(),
                    gradient.data_ptr(),
                    state["exp_avg"].data_ptr(),
                    state["exp_avg_sq"].data_ptr(),
                    state["scale_exp_avg"].data_ptr(),
                    state["scale_exp_avg_sq"].data_ptr(),
                )
                for parameter, gradient, state in entries
            )
            cached = self._fp8_runtime_cache.get(group_index)
            if cached is None or cached[0] != signature:
                batches = [
                    _RuntimeBatch(entries[start : start + MAX_TENSORS_PER_BATCH])
                    for start in range(0, len(entries), MAX_TENSORS_PER_BATCH)
                ]
                cached = (signature, batches)
                self._fp8_runtime_cache[group_index] = cached
            return cached[1]

        @torch.no_grad()
        def step(self, closure=None, grad_scaler=None):
            if grad_scaler is not None:
                raise ValueError("fused FP8 AdamW expects MCore to handle gradient scaling")
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group_index, group in enumerate(self.param_groups):
                if not group["params"]:
                    continue
                group["step"] = int(group.get("step", 0)) + 1
                step = group["step"]
                beta1, beta2 = group["betas"]
                if group.get("bias_correction", True):
                    beta1_correction = 1.0 - beta1**step
                    beta2_correction = 1.0 - beta2**step
                else:
                    beta1_correction = 1.0
                    beta2_correction = 1.0

                entries = []
                for parameter in group["params"]:
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    if gradient.is_sparse:
                        raise RuntimeError("fused FP8 AdamW does not support sparse gradients")
                    if parameter.dtype != torch.float32 or gradient.dtype != torch.float32:
                        raise TypeError(
                            "fused FP8 AdamW requires FP32 MCore master parameters and gradients"
                        )
                    if not parameter.is_contiguous() or not gradient.is_contiguous():
                        raise ValueError("fused FP8 AdamW requires contiguous tensors")
                    state = self.state[parameter]
                    self._ensure_state(parameter, state)
                    entries.append((parameter, gradient, state))

                if not entries:
                    continue

                for batch in self._runtime_batches(group_index, entries):
                    batch.amax.zero_()
                    _adamw_update_amax_kernel[(batch.total_chunks,)](
                        batch.param_ptrs,
                        batch.grad_ptrs,
                        batch.exp_avg_ptrs,
                        batch.exp_avg_sq_ptrs,
                        batch.exp_avg_scale_ptrs,
                        batch.exp_avg_sq_scale_ptrs,
                        batch.numels,
                        batch.chunk_ends,
                        batch.amax,
                        batch.num_tensors,
                        beta1,
                        beta2,
                        beta1_correction,
                        beta2_correction,
                        group["lr"],
                        group["eps"],
                        group["weight_decay"],
                        BLOCK=BLOCK_SIZE,
                        num_warps=NUM_WARPS,
                    )
                    _set_scales_kernel[(batch.num_tensors,)](
                        batch.amax,
                        batch.exp_avg_scale_ptrs,
                        batch.exp_avg_sq_scale_ptrs,
                        batch.num_tensors,
                        FP8_MAX,
                        num_warps=1,
                    )
                    _adamw_encode_states_kernel[(batch.total_chunks,)](
                        batch.grad_ptrs,
                        batch.exp_avg_ptrs,
                        batch.exp_avg_sq_ptrs,
                        batch.exp_avg_scale_ptrs,
                        batch.exp_avg_sq_scale_ptrs,
                        batch.numels,
                        batch.chunk_ends,
                        batch.amax[2 * batch.num_tensors :],
                        batch.num_tensors,
                        beta1,
                        beta2,
                        FP8_MAX,
                        BLOCK=BLOCK_SIZE,
                        num_warps=NUM_WARPS,
                    )
            return loss

    FusedFP8StateAdamW.__name__ = "FusedFP8StateAdamW"
    return FusedFP8StateAdamW

"""Distributed transport for optimizer state-derived Muon payloads.

DDP already synchronizes gradients during backward.  This module implements a
separate, explicit communication path used by state-sharded optimizers: each
rank owns a row shard of the persistent state, computes its local pre-NS
payload, and all-gathers those rows before the replicated Newton-Schulz update.
"""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.distributed as dist

from .fp8_state import (
    init_fp8_state,
    quantize_fp8_state_,
)


@dataclass(frozen=True)
class RowShard:
    tensor: torch.Tensor
    original_rows: int


@dataclass(frozen=True)
class _PackedFP8Part:
    key: str
    dtype: torch.dtype
    offset: int
    numel: int
    nbytes: int


class DistributedStateCommunicator:
    """Shard optimizer state by rows and reconstruct update payloads."""

    def __init__(
        self,
        process_group=None,
        *,
        enabled: bool = False,
        wire_dtype: str = "auto",
        qargs=None,
        profile: bool = False,
    ) -> None:
        if wire_dtype not in {"auto", "bfloat16", "fp8"}:
            raise ValueError(
                "wire_dtype must be one of: auto, bfloat16, fp8; "
                f"got {wire_dtype!r}"
            )
        if wire_dtype == "fp8" and qargs is None:
            raise ValueError("FP8 optimizer-state wire format requires qargs")

        self.process_group = process_group
        self.requested = enabled
        self.wire_dtype = wire_dtype
        self.qargs = qargs
        self.profile_enabled = profile
        self._step_profile: Dict[str, float] = {}
        self._last_profile: Dict[str, float] = {}
        self._pending_cuda_events = []

    @property
    def enabled(self) -> bool:
        return (
            self.requested
            and dist.is_available()
            and dist.is_initialized()
            and self.world_size > 1
        )

    @property
    def world_size(self) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 1
        return dist.get_world_size(self.process_group)

    @property
    def rank(self) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 0
        return dist.get_rank(self.process_group)

    def _use_fp8(self, state_derived: bool) -> bool:
        if not state_derived:
            return False
        if self.wire_dtype == "bfloat16":
            return False
        if self.wire_dtype == "fp8":
            return True
        return self.qargs is not None

    @property
    def reuses_quantized_state_on_wire(self) -> bool:
        """Whether stateful traffic can reuse the persistent FP8 buffers."""
        return self.enabled and self.qargs is not None and self._use_fp8(True)

    def start_step(self) -> None:
        if not self.profile_enabled:
            return
        self._step_profile = {
            "optimizer_state_comm_ms": 0.0,
            "optimizer_state_encode_ms": 0.0,
            "optimizer_state_decode_ms": 0.0,
            "optimizer_state_payload_bytes": 0.0,
            "optimizer_state_wire_bytes": 0.0,
            "optimizer_state_rx_bytes": 0.0,
            "optimizer_stateless_wire_bytes": 0.0,
            "optimizer_stateful_wire_bytes": 0.0,
            "optimizer_state_collectives": 0.0,
        }
        self._pending_cuda_events = []

    @contextmanager
    def phase(self, name: str, tensor: torch.Tensor) -> Iterator[None]:
        if not self.profile_enabled:
            with nullcontext():
                yield
            return

        if tensor.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(name)
            start.record()
            try:
                yield
            finally:
                end.record()
                torch.cuda.nvtx.range_pop()
                self._pending_cuda_events.append((name, start, end))
        else:
            start_time = time.perf_counter()
            try:
                yield
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1e3
                key = f"optimizer_state_{name}_ms"
                self._step_profile[key] = self._step_profile.get(key, 0.0) + elapsed_ms

    def finish_step(self) -> None:
        if not self.profile_enabled:
            return
        if self._pending_cuda_events:
            torch.cuda.synchronize()
            for name, start, end in self._pending_cuda_events:
                key = f"optimizer_state_{name}_ms"
                self._step_profile[key] = (
                    self._step_profile.get(key, 0.0) + start.elapsed_time(end)
                )
        self._last_profile = dict(self._step_profile)

    def get_last_profile(self) -> Dict[str, float]:
        return dict(self._last_profile)

    def local_rows(self, tensor: torch.Tensor) -> RowShard:
        """Return this rank's equally-sized row shard, padding the last rows."""
        if tensor.ndim < 2:
            raise ValueError(f"row sharding requires a matrix, got shape {tensor.shape}")
        if not self.enabled:
            return RowShard(tensor=tensor, original_rows=tensor.shape[0])

        rows = tensor.shape[0]
        shard_rows = (rows + self.world_size - 1) // self.world_size
        padded_rows = shard_rows * self.world_size
        if padded_rows != rows:
            padding = tensor.new_zeros((padded_rows - rows, *tensor.shape[1:]))
            tensor = torch.cat((tensor, padding), dim=0)
        start = self.rank * shard_rows
        return RowShard(
            tensor=tensor.narrow(0, start, shard_rows).contiguous(),
            original_rows=rows,
        )

    def gather_rows(
        self,
        shard: RowShard,
        *,
        state_derived: bool,
    ) -> torch.Tensor:
        """All-gather a row shard, optionally using an FP8 wire representation."""
        if not self.enabled:
            return shard.tensor

        local = shard.tensor
        # Distributed Muon's uncompressed reference wire format is BF16 even
        # when the local momentum computation is performed in FP32.
        payload_bytes = local.numel() * 2 * self.world_size
        if self._use_fp8(state_derived):
            full, local_wire_bytes = self._gather_fp8(local)
        else:
            full, local_wire_bytes = self._gather_bfloat16(local)

        self._record_transfer(payload_bytes, local_wire_bytes, state_derived)

        return full.narrow(0, 0, shard.original_rows).to(dtype=local.dtype)

    def gather_quantized_state_rows(
        self,
        state: Dict[str, Any],
        prefix: str,
        *,
        original_rows: int,
        gradient: torch.Tensor | None = None,
        gradient_alpha: float = 0.0,
    ) -> torch.Tensor:
        """All-gather persistent FP8 state and reconstruct the full Muon input.

        The FP8 values and scales in ``state`` are transmitted as-is.  Only
        packet assembly remains in the encode phase; no second quantization is
        performed.  Decode and the optional Nesterov add are fused on CUDA.
        """
        if not self.reuses_quantized_state_on_wire:
            raise RuntimeError(
                "quantized-state wire reuse requires distributed FP8 state traffic"
            )
        local = state[prefix]
        if local.ndim < 2:
            raise ValueError(f"row sharding requires a matrix, got shape {local.shape}")
        output_shape = torch.Size((original_rows, *local.shape[1:]))
        if gradient is not None and tuple(gradient.shape) != tuple(output_shape):
            raise ValueError(
                f"gradient shape {tuple(gradient.shape)} != gathered shape {tuple(output_shape)}"
            )

        with self.phase("encode", local):
            packed, part_specs = self._pack_fp8_state(state, prefix)
        gathered = self._all_gather_flat(packed)
        part_views = self._unpack_fp8_views(gathered, packed.numel(), part_specs)

        from .fp8_state_wire import dequantize_fp8_state_and_add

        with self.phase("decode", local):
            full = dequantize_fp8_state_and_add(
                part_views[prefix],
                part_views[f"scale_{prefix}"],
                self.qargs.qgroup_size,
                output_shape=output_shape,
                expand=part_views.get(f"expand_{prefix}"),
                sqrt_minmax=part_views.get(f"sqrt_minmax_{prefix}"),
                gradient=gradient,
                gradient_alpha=gradient_alpha,
            )

        payload_bytes = local.numel() * 2 * self.world_size
        self._record_transfer(payload_bytes, packed.numel(), state_derived=True)
        return full

    def _record_transfer(
        self,
        payload_bytes: int,
        local_wire_bytes: int,
        state_derived: bool,
    ) -> None:
        if not self.profile_enabled:
            return
        global_wire_bytes = local_wire_bytes * self.world_size
        self._step_profile["optimizer_state_payload_bytes"] += payload_bytes
        self._step_profile["optimizer_state_wire_bytes"] += global_wire_bytes
        self._step_profile["optimizer_state_rx_bytes"] += (
            local_wire_bytes * (self.world_size - 1)
        )
        kind = "stateful" if state_derived else "stateless"
        self._step_profile[f"optimizer_{kind}_wire_bytes"] += global_wire_bytes
        self._step_profile["optimizer_state_collectives"] += 1

    def _all_gather_flat(self, local: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            local.numel() * self.world_size,
            dtype=local.dtype,
            device=local.device,
        )
        with self.phase("comm", local):
            dist.all_gather_into_tensor(output, local.reshape(-1), group=self.process_group)
        return output

    def _gather_bfloat16(self, local: torch.Tensor) -> Tuple[torch.Tensor, int]:
        wire = local.to(torch.bfloat16).contiguous()
        gathered = self._all_gather_flat(wire)
        full_shape = (local.shape[0] * self.world_size, *local.shape[1:])
        return gathered.reshape(full_shape), wire.numel() * wire.element_size()

    def _gather_fp8(self, local: torch.Tensor) -> Tuple[torch.Tensor, int]:
        fp8_state: Dict[str, torch.Tensor] = {}
        with self.phase("encode", local):
            init_fp8_state(fp8_state, "wire", local, self.qargs, order="first")
            quantize_fp8_state_(
                fp8_state,
                "wire",
                local,
                self.qargs,
                signed=True,
            )
            packed, part_specs = self._pack_fp8_state(fp8_state, "wire")

        gathered = self._all_gather_flat(packed)
        part_views = self._unpack_fp8_views(gathered, packed.numel(), part_specs)

        from .fp8_state_wire import dequantize_fp8_state_and_add

        full_shape = torch.Size((local.shape[0] * self.world_size, *local.shape[1:]))
        with self.phase("decode", local):
            full = dequantize_fp8_state_and_add(
                part_views["wire"],
                part_views["scale_wire"],
                self.qargs.qgroup_size,
                output_shape=full_shape,
                expand=part_views.get("expand_wire"),
                sqrt_minmax=part_views.get("sqrt_minmax_wire"),
            ).to(local.dtype)
        return full, packed.numel()

    @staticmethod
    def _pack_fp8_state(
        state: Dict[str, Any],
        prefix: str,
    ) -> Tuple[torch.Tensor, Tuple[_PackedFP8Part, ...]]:
        names = [prefix, f"scale_{prefix}"]
        if f"expand_{prefix}" in state:
            names.extend((f"expand_{prefix}", f"sqrt_minmax_{prefix}"))

        specs = []
        offset = 0
        max_alignment = 1
        for name in names:
            part = state[name]
            element_size = part.element_size()
            max_alignment = max(max_alignment, element_size)
            offset = _align_up(offset, element_size)
            nbytes = part.numel() * element_size
            specs.append(
                _PackedFP8Part(
                    key=name,
                    dtype=part.dtype,
                    offset=offset,
                    numel=part.numel(),
                    nbytes=nbytes,
                )
            )
            offset += nbytes
        packet_bytes = _align_up(offset, max_alignment)
        reference = state[prefix]
        packed = torch.empty(packet_bytes, dtype=torch.uint8, device=reference.device)
        for spec in specs:
            source = state[spec.key].contiguous().view(torch.uint8).reshape(-1)
            packed.narrow(0, spec.offset, spec.nbytes).copy_(source)
        return packed, tuple(specs)

    def _unpack_fp8_views(
        self,
        gathered: torch.Tensor,
        packet_bytes: int,
        specs: Tuple[_PackedFP8Part, ...],
    ) -> Dict[str, torch.Tensor]:
        packets = gathered.view(self.world_size, packet_bytes)
        views = {}
        for spec in specs:
            part_bytes = packets.narrow(1, spec.offset, spec.nbytes)
            views[spec.key] = part_bytes.view(spec.dtype).reshape(
                self.world_size, spec.numel
            )
        return views

def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment

"""Row-sharded optimizer-state transport for the Megatron benchmarks."""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Sequence, Tuple

import torch
import torch.distributed as dist

from stage4.fp8_optimizer_states import init_fp8_state, quantize_fp8_state_
from stage4.fp8_state_wire import (
    dequantize_fp8_state_and_add,
    prepare_fp8_state_ns_input,
)


@dataclass(frozen=True)
class RowShard:
    tensor: torch.Tensor
    original_rows: int


@dataclass(frozen=True)
class _PackedPart:
    key: str
    dtype: torch.dtype
    offset: int
    numel: int
    nbytes: int


@dataclass(frozen=True)
class FP8GatherRequest:
    state: Dict[str, Any]
    prefix: str
    original_rows: int
    gradient: torch.Tensor | None
    gradient_alpha: float
    prepare_for_ns: bool = False


@dataclass(frozen=True)
class GatheredFP8Input:
    tensor: torch.Tensor
    normalized: bool = False
    transposed: bool = False


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class DistributedStateCommunicator:
    """Shard matrix state across DP ranks and reconstruct full NS inputs."""

    def __init__(
        self,
        process_group=None,
        *,
        enabled: bool,
        wire_dtype: str,
        profile: bool,
        group_size: int = 128,
        fp8_bucket_bytes: int = 0,
    ) -> None:
        if wire_dtype not in {"bfloat16", "fp8"}:
            raise ValueError("wire_dtype must be 'bfloat16' or 'fp8'")
        self.process_group = process_group
        self.requested = enabled
        self.wire_dtype = wire_dtype
        self.profile_enabled = profile
        self.group_size = group_size
        self.fp8_bucket_bytes = fp8_bucket_bytes
        self._profile: Dict[str, float] = {}
        self._last_profile: Dict[str, float] = {}
        self._events = []

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

    @property
    def enabled(self) -> bool:
        return self.requested and self.world_size > 1

    @property
    def reuses_quantized_state_on_wire(self) -> bool:
        return self.enabled and self.wire_dtype == "fp8"

    def start_step(self) -> None:
        if not self.profile_enabled:
            return
        self._profile = {
            "optimizer_ms": 0.0,
            "newton_schulz_ms": 0.0,
            "wire_encode_ms": 0.0,
            "state_all_gather_ms": 0.0,
            "wire_decode_ms": 0.0,
            "stateful_wire_bytes": 0.0,
            "stateless_wire_bytes": 0.0,
            "received_wire_bytes": 0.0,
            "collectives": 0.0,
        }
        self._events = []

    @contextmanager
    def phase(self, name: str, tensor: torch.Tensor) -> Iterator[None]:
        if not self.profile_enabled:
            with nullcontext():
                yield
            return
        if tensor.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(f"optimizer_state/{name}")
            start.record()
            try:
                yield
            finally:
                end.record()
                torch.cuda.nvtx.range_pop()
                self._events.append((name, start, end))
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self._profile[f"{name}_ms"] += (time.perf_counter() - started) * 1e3

    def finish_step(self) -> Dict[str, float]:
        if not self.profile_enabled:
            return {}
        if self._events:
            torch.cuda.synchronize()
            for name, start, end in self._events:
                self._profile[f"{name}_ms"] += start.elapsed_time(end)
        self._events = []
        measured = (
            self._profile["newton_schulz_ms"]
            + self._profile["wire_encode_ms"]
            + self._profile["state_all_gather_ms"]
            + self._profile["wire_decode_ms"]
        )
        self._profile["other_ms"] = max(0.0, self._profile["optimizer_ms"] - measured)
        self._last_profile = dict(self._profile)
        return dict(self._last_profile)

    def get_last_profile(self) -> Dict[str, float]:
        return dict(self._last_profile)

    def local_rows(self, tensor: torch.Tensor) -> RowShard:
        if tensor.ndim != 2:
            raise ValueError(
                f"row sharding requires a matrix, got {tuple(tensor.shape)}"
            )
        if not self.enabled:
            return RowShard(tensor.contiguous(), tensor.shape[0])
        rows = tensor.shape[0]
        shard_rows = (rows + self.world_size - 1) // self.world_size
        padded_rows = shard_rows * self.world_size
        if padded_rows != rows:
            tensor = torch.cat(
                (tensor, tensor.new_zeros((padded_rows - rows, tensor.shape[1]))), dim=0
            )
        start = self.rank * shard_rows
        return RowShard(tensor.narrow(0, start, shard_rows).contiguous(), rows)

    def gather_bfloat16(self, shard: RowShard, *, state_derived: bool) -> torch.Tensor:
        if not self.enabled:
            return shard.tensor.float()
        wire = shard.tensor.to(torch.bfloat16).contiguous()
        gathered = self._all_gather(wire)
        self._record_bytes(wire.numel() * wire.element_size(), state_derived)
        full = gathered.reshape(self.world_size * wire.shape[0], wire.shape[1])
        return full[: shard.original_rows].float()

    def gather_fp8_state(
        self,
        state: Dict[str, Any],
        prefix: str,
        *,
        original_rows: int,
        gradient: torch.Tensor | None = None,
        gradient_alpha: float = 0.0,
    ) -> torch.Tensor:
        if not self.reuses_quantized_state_on_wire:
            raise RuntimeError("FP8 state gather requires at least two DP ranks")
        with self.phase("wire_encode", state[prefix]):
            packet, specs = self._pack_fp8_state(state, prefix)
        gathered = self._all_gather(packet)
        fields = self._unpack(gathered, packet.numel(), specs)
        output_shape = torch.Size((original_rows, state[prefix].shape[1]))
        with self.phase("wire_decode", state[prefix]):
            output = dequantize_fp8_state_and_add(
                fields[prefix],
                fields[f"scale_{prefix}"],
                fields[f"expand_{prefix}"],
                fields[f"sqrt_minmax_{prefix}"],
                group_size=self.group_size,
                output_shape=output_shape,
                gradient=gradient,
                gradient_alpha=gradient_alpha,
            )
        self._record_bytes(packet.numel(), state_derived=True)
        return output

    def gather_fp8_states(
        self, requests: Sequence[FP8GatherRequest]
    ) -> Iterator[tuple[int, GatheredFP8Input]]:
        """Gather persistent FP8 states in bounded contiguous communication buckets."""
        if not self.reuses_quantized_state_on_wire:
            raise RuntimeError("FP8 state gather requires at least two DP ranks")
        layouts = []
        for index, request in enumerate(requests):
            specs, packet_bytes = self._fp8_layout(request.state, request.prefix)
            layouts.append((index, request, specs, packet_bytes))

        buckets = []
        current = []
        current_bytes = 0
        limit = self.fp8_bucket_bytes
        for layout in layouts:
            packet_bytes = layout[3]
            aligned_start = _align_up(current_bytes, 4)
            if current and limit > 0 and aligned_start + packet_bytes > limit:
                buckets.append((current, current_bytes))
                current = []
                current_bytes = 0
                aligned_start = 0
            current.append((*layout, aligned_start))
            current_bytes = aligned_start + packet_bytes
        if current:
            buckets.append((current, current_bytes))

        for bucket, bucket_bytes in buckets:
            with self.phase("wire_encode", bucket[0][1].state[bucket[0][1].prefix]):
                local = torch.empty(
                    _align_up(bucket_bytes, 4),
                    dtype=torch.uint8,
                    device=bucket[0][1].state[bucket[0][1].prefix].device,
                )
                for _, request, specs, _, packet_offset in bucket:
                    self._copy_fp8_packet(local, packet_offset, request.state, specs)
            gathered = self._all_gather(local)
            packets = gathered.view(self.world_size, local.numel())
            self._record_bytes(local.numel(), state_derived=True)
            for index, request, specs, _, packet_offset in bucket:
                fields = self._unpack_packets(packets, packet_offset, specs)
                output_shape = torch.Size(
                    (request.original_rows, request.state[request.prefix].shape[1])
                )
                with self.phase("wire_decode", request.state[request.prefix]):
                    if request.prepare_for_ns:
                        output, transposed = prepare_fp8_state_ns_input(
                            fields[request.prefix],
                            fields[f"scale_{request.prefix}"],
                            fields[f"expand_{request.prefix}"],
                            fields[f"sqrt_minmax_{request.prefix}"],
                            group_size=self.group_size,
                            output_shape=output_shape,
                            gradient=request.gradient,
                            gradient_alpha=request.gradient_alpha,
                        )
                        result = GatheredFP8Input(
                            output, normalized=True, transposed=transposed
                        )
                    else:
                        output = dequantize_fp8_state_and_add(
                            fields[request.prefix],
                            fields[f"scale_{request.prefix}"],
                            fields[f"expand_{request.prefix}"],
                            fields[f"sqrt_minmax_{request.prefix}"],
                            group_size=self.group_size,
                            output_shape=output_shape,
                            gradient=request.gradient,
                            gradient_alpha=request.gradient_alpha,
                        )
                        result = GatheredFP8Input(output)
                yield index, result

    def gather_ephemeral_fp8(
        self, shard: RowShard, *, state_derived: bool
    ) -> torch.Tensor:
        """Quantize a non-persistent tensor before communication.

        This is retained for controlled ablations. FRUGAL's stateless complement
        deliberately uses BF16, while persistent momentum uses ``gather_fp8_state``.
        """
        state: Dict[str, Any] = {}
        with self.phase("wire_encode", shard.tensor):
            init_fp8_state(state, "wire", shard.tensor, group_size=self.group_size)
            quantize_fp8_state_(
                state, "wire", shard.tensor, signed=True, group_size=self.group_size
            )
            packet, specs = self._pack_fp8_state(state, "wire")
        gathered = self._all_gather(packet)
        fields = self._unpack(gathered, packet.numel(), specs)
        output_shape = torch.Size((shard.original_rows, shard.tensor.shape[1]))
        with self.phase("wire_decode", shard.tensor):
            output = dequantize_fp8_state_and_add(
                fields["wire"],
                fields["scale_wire"],
                fields["expand_wire"],
                fields["sqrt_minmax_wire"],
                group_size=self.group_size,
                output_shape=output_shape,
            )
        self._record_bytes(packet.numel(), state_derived)
        return output

    def _all_gather(self, local: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            local.numel() * self.world_size, dtype=local.dtype, device=local.device
        )
        with self.phase("state_all_gather", local):
            dist.all_gather_into_tensor(
                output, local.reshape(-1), group=self.process_group
            )
        return output

    def _record_bytes(self, local_wire_bytes: int, state_derived: bool) -> None:
        if not self.profile_enabled:
            return
        key = "stateful_wire_bytes" if state_derived else "stateless_wire_bytes"
        self._profile[key] += local_wire_bytes * self.world_size
        self._profile["received_wire_bytes"] += local_wire_bytes * (self.world_size - 1)
        self._profile["collectives"] += 1

    @staticmethod
    def _pack_fp8_state(
        state: Dict[str, Any], prefix: str
    ) -> Tuple[torch.Tensor, Tuple[_PackedPart, ...]]:
        specs, packet_bytes = DistributedStateCommunicator._fp8_layout(state, prefix)
        packet = torch.empty(
            packet_bytes,
            dtype=torch.uint8,
            device=state[prefix].device,
        )
        DistributedStateCommunicator._copy_fp8_packet(packet, 0, state, specs)
        return packet, specs

    @staticmethod
    def _fp8_layout(
        state: Dict[str, Any], prefix: str
    ) -> Tuple[Tuple[_PackedPart, ...], int]:
        names = (
            prefix,
            f"scale_{prefix}",
            f"expand_{prefix}",
            f"sqrt_minmax_{prefix}",
        )
        specs = []
        offset = 0
        max_alignment = 1
        for name in names:
            part = state[name]
            alignment = part.element_size()
            max_alignment = max(max_alignment, alignment)
            offset = _align_up(offset, alignment)
            nbytes = part.numel() * alignment
            specs.append(_PackedPart(name, part.dtype, offset, part.numel(), nbytes))
            offset += nbytes
        return tuple(specs), _align_up(offset, max_alignment)

    @staticmethod
    def _copy_fp8_packet(
        packet: torch.Tensor,
        packet_offset: int,
        state: Dict[str, Any],
        specs: Tuple[_PackedPart, ...],
    ) -> None:
        for spec in specs:
            source = state[spec.key].contiguous().view(torch.uint8).reshape(-1)
            packet.narrow(0, packet_offset + spec.offset, spec.nbytes).copy_(source)

    @staticmethod
    def _unpack_packets(
        packets: torch.Tensor,
        packet_offset: int,
        specs: Tuple[_PackedPart, ...],
    ) -> Dict[str, torch.Tensor]:
        return {
            spec.key: packets.narrow(
                1, packet_offset + spec.offset, spec.nbytes
            )
            .view(spec.dtype)
            .reshape(packets.shape[0], spec.numel)
            for spec in specs
        }

    def _unpack(
        self,
        gathered: torch.Tensor,
        packet_bytes: int,
        specs: Tuple[_PackedPart, ...],
    ) -> Dict[str, torch.Tensor]:
        packets = gathered.view(self.world_size, packet_bytes)
        return {
            spec.key: packets.narrow(1, spec.offset, spec.nbytes)
            .view(spec.dtype)
            .reshape(self.world_size, spec.numel)
            for spec in specs
        }

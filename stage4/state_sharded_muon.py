"""Muon variants with DP-sharded momentum and explicit state communication."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

import torch
import torch.distributed as dist

from emerging_optimizers import utils
from emerging_optimizers.orthogonalized_optimizers import muon_utils
from emerging_optimizers.orthogonalized_optimizers.muon import get_muon_scale_factor
from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon
from megatron.core.utils import get_pg_size

from stage4.distributed_state_comm import (
    DistributedStateCommunicator,
    FP8GatherRequest,
    GatheredFP8Input,
    RowShard,
)
from stage4.fp8_momentum import update_fp8_momentum_
from stage4.fp8_optimizer_states import (
    FP8StateDictMixin,
    dequantize_fp8_state,
    init_fp8_state,
)


PROFILE_PREFIX = "[OPTIMIZER STATE COMM PROFILE] "


@dataclass
class _PendingFP8Matrix:
    parameter: torch.Tensor
    grad: torch.Tensor
    state: Dict[str, Any]
    lr: float
    active_indices: torch.Tensor | None


def _group_root(process_group) -> int:
    if hasattr(dist, "get_global_rank"):
        return dist.get_global_rank(process_group, 0)
    ranks = [None for _ in range(dist.get_world_size(process_group))]
    dist.all_gather_object(ranks, dist.get_rank(), group=process_group)
    return int(ranks[0])


class StateShardedMuon(FP8StateDictMixin, TensorParallelMuon):
    """Muon whose matrix momentum is row-sharded across the DP group.

    Model parameters remain local to their Megatron PP/TP stage. DDP still
    produces replicated gradients inside each DP group. Each DP rank updates
    only its momentum rows and all-gathers the pre-NS input before applying the
    same parameter update on every replica.
    """

    is_frugal = False
    group_size = 128

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection=None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
        use_syrk: bool = False,
        batched_newton_schulz: bool = False,
        state_precision: Literal["bfloat16", "fp8"] = "bfloat16",
        distributed_state_sharding: bool = True,
        profile_state_communication: bool = False,
        fp8_bucket_bytes: int = 0,
        fused_fp8_ns_input: bool = False,
        frugal_density: float = 0.25,
        frugal_update_gap: int = 50,
        frugal_coord_choice: str = "columns",
        frugal_inactive_lr_scale: float = 1.0,
    ) -> None:
        if state_precision not in {"bfloat16", "fp8"}:
            raise ValueError("state_precision must be 'bfloat16' or 'fp8'")
        if batched_newton_schulz:
            raise ValueError(
                "batched Newton-Schulz is not compatible with per-tensor DP state gathers"
            )
        if self.is_frugal:
            if not 0.0 < frugal_density <= 1.0:
                raise ValueError("frugal_density must be in (0, 1]")
            if frugal_update_gap < 1:
                raise ValueError("frugal_update_gap must be positive")
            if frugal_coord_choice != "columns":
                raise ValueError(
                    "the Megatron FRUGAL path currently supports columns only"
                )
            if pg_collection is not None and get_pg_size(pg_collection.tp) > 1:
                raise ValueError(
                    "FRUGAL Muon-Muon currently requires tensor-model-parallel-size=1; "
                    "use pipeline parallelism to shard model weights"
                )

        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            split_qkv=split_qkv,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=qkv_split_shapes,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
            use_syrk=use_syrk,
            batched_newton_schulz=False,
        )
        self.state_precision = state_precision
        self.frugal_density = frugal_density
        self.frugal_update_gap = frugal_update_gap
        self.frugal_coord_choice = frugal_coord_choice
        self.frugal_inactive_lr_scale = frugal_inactive_lr_scale
        self.coefficient_type = coefficient_type
        self.num_ns_steps = num_ns_steps
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor
        self.fused_fp8_ns_input = fused_fp8_ns_input
        dp_group = None if pg_collection is None else pg_collection.dp_cp
        self.state_comm = DistributedStateCommunicator(
            dp_group,
            enabled=distributed_state_sharding,
            wire_dtype="fp8" if state_precision == "fp8" else "bfloat16",
            profile=profile_state_communication,
            group_size=self.group_size,
            fp8_bucket_bytes=fp8_bucket_bytes,
        )
        self._profile_step = 0
        self._defer_profile_emit = False
        self._pending_profile = None

    def _is_frugal_parameter(self, parameter: torch.Tensor) -> bool:
        return (
            self.is_frugal
            and parameter.ndim == 2
            and not getattr(parameter, "is_embedding_or_output_parameter", False)
        )

    def _new_indices(self, parameter: torch.Tensor) -> torch.Tensor:
        count = max(1, int(parameter.shape[1] * self.frugal_density))
        if not self.state_comm.enabled or self.state_comm.rank == 0:
            indices = torch.randperm(parameter.shape[1], device=parameter.device)[
                :count
            ]
        else:
            indices = torch.empty(count, dtype=torch.long, device=parameter.device)
        if self.state_comm.enabled:
            dist.broadcast(
                indices,
                src=_group_root(self.state_comm.process_group),
                group=self.state_comm.process_group,
            )
        return indices

    def _state_reference(
        self, parameter: torch.Tensor, state: Dict[str, Any]
    ) -> torch.Tensor:
        reference = parameter
        if self._is_frugal_parameter(parameter):
            reference = parameter[:, state["active_indices"]]
        if reference.ndim == 2:
            reference = self.state_comm.local_rows(reference).tensor
        return reference

    def _clear_momentum(self, state: Dict[str, Any]) -> None:
        for key in (
            "momentum_buffer",
            "scale_momentum_buffer",
            "expand_momentum_buffer",
            "sqrt_minmax_momentum_buffer",
        ):
            state.pop(key, None)

    def _init_momentum(self, reference: torch.Tensor, state: Dict[str, Any]) -> None:
        self._clear_momentum(state)
        if self.state_precision == "fp8":
            init_fp8_state(
                state, "momentum_buffer", reference, group_size=self.group_size
            )
        else:
            state["momentum_buffer"] = torch.zeros_like(reference, dtype=torch.bfloat16)

    @torch.no_grad()
    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        for parameter in group["params"]:
            if skip_non_grad_params and parameter.grad is None:
                continue
            state = self.state[parameter]
            if "state_world_size" in state:
                if state["state_world_size"] != self.state_comm.world_size:
                    raise ValueError(
                        "optimizer state was saved with a different DP size; "
                        "checkpoint resharding is not implemented"
                    )
                continue
            state["state_world_size"] = self.state_comm.world_size
            state["projection_step"] = 0
            if self._is_frugal_parameter(parameter):
                state["active_indices"] = self._new_indices(parameter)
            self._init_momentum(self._state_reference(parameter, state), state)

    def _refresh_projection(
        self, parameter: torch.Tensor, state: Dict[str, Any]
    ) -> None:
        step = int(state["projection_step"])
        if step > 0 and step % self.frugal_update_gap == 0:
            state["active_indices"] = self._new_indices(parameter)
            self._init_momentum(self._state_reference(parameter, state), state)
        state["projection_step"] = step + 1

    def _update_fp8_state(
        self, state: Dict[str, Any], local_grad: torch.Tensor, momentum: float
    ) -> None:
        update_fp8_momentum_(
            state,
            "momentum_buffer",
            local_grad,
            momentum=momentum,
            gradient_alpha=1.0 - momentum,
            group_size=self.group_size,
        )

    def _stateful_ns_input(
        self,
        grad: torch.Tensor,
        state: Dict[str, Any],
        momentum: float,
    ) -> torch.Tensor:
        shard = self.state_comm.local_rows(grad)
        local_grad = shard.tensor.float()
        nesterov_alpha = (1.0 - momentum) / momentum if self.nesterov else 0.0

        if self.state_precision == "fp8":
            self._update_fp8_state(state, local_grad, momentum)
            if self.state_comm.enabled:
                return self.state_comm.gather_fp8_state(
                    state,
                    "momentum_buffer",
                    original_rows=shard.original_rows,
                    gradient=grad if self.nesterov else None,
                    gradient_alpha=nesterov_alpha,
                )
            decoded = dequantize_fp8_state(
                state,
                "momentum_buffer",
                signed=True,
                group_size=self.group_size,
            )
            return (
                decoded.add(grad.float(), alpha=nesterov_alpha)
                if self.nesterov
                else decoded
            )

        momentum_buffer = state["momentum_buffer"]
        updated = (
            momentum_buffer.float()
            .mul_(momentum)
            .add_(local_grad, alpha=1.0 - momentum)
        )
        momentum_buffer.copy_(updated)
        local_input = (
            updated.add(local_grad, alpha=nesterov_alpha) if self.nesterov else updated
        )
        return self.state_comm.gather_bfloat16(
            RowShard(local_input, shard.original_rows), state_derived=True
        )

    def _stateless_ns_input(self, grad: torch.Tensor) -> torch.Tensor:
        shard = self.state_comm.local_rows(grad)
        return self.state_comm.gather_bfloat16(shard, state_derived=False)

    def _orthogonalize_profiled(
        self, parameter: torch.Tensor, ns_input: torch.Tensor
    ) -> torch.Tensor:
        phase = self.state_comm.phase("newton_schulz", ns_input)
        with phase, utils.fp32_matmul_precision(self.fp32_matmul_prec):
            return self.orthogonalize(parameter, ns_input)

    def _can_fuse_ns_input(self, parameter: torch.Tensor) -> bool:
        return self.fused_fp8_ns_input and not (
            self.split_qkv and self.is_qkv_fn(parameter)
        )

    def _orthogonalize_prepared_profiled(
        self,
        parameter: torch.Tensor,
        prepared: GatheredFP8Input,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        if not prepared.normalized:
            return self._orthogonalize_profiled(parameter, prepared.tensor)
        phase = self.state_comm.phase("newton_schulz", prepared.tensor)
        with phase, utils.fp32_matmul_precision(self.fp32_matmul_prec):
            coefficients = muon_utils._COEFFICIENT_SETS[self.coefficient_type]
            mode = "repeat_last" if self.coefficient_type == "polar_express" else "cycle"
            coeff_iter = muon_utils.get_coefficient_iterator(
                self.num_ns_steps, coefficients, mode=mode
            )
            output = prepared.tensor
            for a, b, c in coeff_iter:
                output = muon_utils.newton_schulz_step(
                    output, a, b, c, tp_group=None
                )
            output = output.float()
            if prepared.transposed:
                output = output.mT
            scale = get_muon_scale_factor(
                original_shape[0], original_shape[1], mode=self.scale_mode
            )
            return output * scale * self.extra_scale_factor

    def _apply_update(
        self, parameter: torch.Tensor, update: torch.Tensor, lr: float
    ) -> None:
        self.pre_weight_update_fn_inplace(parameter, update)
        parameter.add_(update, alpha=-lr)
        self.post_weight_update_fn_inplace(parameter)

    def _step_fp8_bucketed(self, group: dict) -> None:
        pending: list[_PendingFP8Matrix] = []
        requests: list[FP8GatherRequest] = []
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            grad = parameter.grad
            state = self.state[parameter]
            self._apply_weight_decay_inplace(
                parameter, grad, group["lr"], group["weight_decay"]
            )
            if grad.ndim != 2:
                self._apply_update(
                    parameter,
                    self._vector_update(grad, state, group["momentum"]),
                    group["lr"],
                )
                continue

            active_indices = None
            stateful_grad = grad
            if self._is_frugal_parameter(parameter):
                self._refresh_projection(parameter, state)
                active_indices = state["active_indices"]
                stateful_grad = grad[:, active_indices]
            shard = self.state_comm.local_rows(stateful_grad)
            self._update_fp8_state(state, shard.tensor.float(), group["momentum"])
            nesterov_alpha = (
                (1.0 - group["momentum"]) / group["momentum"]
                if self.nesterov
                else 0.0
            )
            requests.append(
                FP8GatherRequest(
                    state=state,
                    prefix="momentum_buffer",
                    original_rows=shard.original_rows,
                    gradient=stateful_grad if self.nesterov else None,
                    gradient_alpha=nesterov_alpha,
                    prepare_for_ns=self._can_fuse_ns_input(parameter),
                )
            )
            pending.append(
                _PendingFP8Matrix(
                    parameter, grad, state, group["lr"], active_indices
                )
            )

        for index, gathered in self.state_comm.gather_fp8_states(requests):
            item = pending[index]
            stateful_shape = (
                item.grad.shape
                if item.active_indices is None
                else torch.Size((item.grad.shape[0], item.active_indices.numel()))
            )
            stateful_update = self._orthogonalize_prepared_profiled(
                item.parameter, gathered, stateful_shape
            )
            if item.active_indices is None:
                update = stateful_update
            else:
                mask = torch.ones(
                    item.grad.shape[1], dtype=torch.bool, device=item.grad.device
                )
                mask[item.active_indices] = False
                inactive_indices = torch.where(mask)[0]
                update = torch.zeros_like(item.grad)
                update[:, item.active_indices] = stateful_update.to(update.dtype)
                if inactive_indices.numel() > 0:
                    inactive_grad = item.grad[:, inactive_indices]
                    inactive_update = self._orthogonalize_profiled(
                        item.parameter, self._stateless_ns_input(inactive_grad)
                    )
                    update[:, inactive_indices] = (
                        inactive_update * self.frugal_inactive_lr_scale
                    ).to(update.dtype)
            self._apply_update(item.parameter, update, item.lr)

    def _matrix_update(
        self,
        parameter: torch.Tensor,
        grad: torch.Tensor,
        state: Dict[str, Any],
        momentum: float,
    ) -> torch.Tensor:
        if not self._is_frugal_parameter(parameter):
            return self._orthogonalize_profiled(
                parameter, self._stateful_ns_input(grad, state, momentum)
            )

        self._refresh_projection(parameter, state)
        active_indices = state["active_indices"]
        active_grad = grad[:, active_indices]
        active_update = self._orthogonalize_profiled(
            parameter, self._stateful_ns_input(active_grad, state, momentum)
        )

        mask = torch.ones(grad.shape[1], dtype=torch.bool, device=grad.device)
        mask[active_indices] = False
        inactive_indices = torch.where(mask)[0]
        update = torch.zeros_like(grad)
        update[:, active_indices] = active_update.to(update.dtype)
        if inactive_indices.numel() > 0:
            inactive_grad = grad[:, inactive_indices]
            inactive_update = self._orthogonalize_profiled(
                parameter, self._stateless_ns_input(inactive_grad)
            )
            update[:, inactive_indices] = (
                inactive_update * self.frugal_inactive_lr_scale
            ).to(update.dtype)
        return update

    def _vector_update(
        self, grad: torch.Tensor, state: Dict[str, Any], momentum: float
    ) -> torch.Tensor:
        if self.state_precision == "fp8":
            self._update_fp8_state(state, grad.float(), momentum)
            value = dequantize_fp8_state(
                state,
                "momentum_buffer",
                signed=True,
                group_size=self.group_size,
            )
        else:
            value = (
                state["momentum_buffer"]
                .float()
                .mul_(momentum)
                .add_(grad.float(), alpha=1.0 - momentum)
            )
            state["momentum_buffer"].copy_(value)
        if self.nesterov:
            value = value.add(grad.float(), alpha=(1.0 - momentum) / momentum)
        return value.sign()

    def _emit_profile(self, profile: Dict[str, float]) -> None:
        if not profile:
            return
        profile.update(
            {
                "step": self._profile_step,
                "rank": dist.get_rank() if dist.is_initialized() else 0,
                "dp_rank": self.state_comm.rank,
                "dp_size": self.state_comm.world_size,
                "method": "frugal_muon_muon" if self.is_frugal else "muon",
                "states": "FP8" if self.state_precision == "fp8" else "BF16",
            }
        )
        if self._defer_profile_emit:
            self._pending_profile = profile
            return
        print(PROFILE_PREFIX + json.dumps(profile, sort_keys=True), flush=True)

    def flush_state_comm_profile(self, chained_optimizer_ms: float) -> None:
        """Emit a deferred profile after all optimizers in a chain have run."""
        profile = self._pending_profile
        if profile is None:
            return
        measured = sum(
            profile[key]
            for key in (
                "newton_schulz_ms",
                "wire_encode_ms",
                "state_all_gather_ms",
                "wire_decode_ms",
            )
        )
        profile["optimizer_ms"] = chained_optimizer_ms
        profile["other_ms"] = max(0.0, chained_optimizer_ms - measured)
        print(PROFILE_PREFIX + json.dumps(profile, sort_keys=True), flush=True)
        self._pending_profile = None

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        self.state_comm.start_step()
        reference = next(
            (
                parameter
                for group in self.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ),
            None,
        )
        optimizer_phase = (
            self.state_comm.phase("optimizer", reference)
            if reference is not None
            else nullcontext()
        )
        with optimizer_phase:
            for group in self.param_groups:
                self._init_group(group)
                if (
                    self.state_precision == "fp8"
                    and self.state_comm.enabled
                    and self.state_comm.fp8_bucket_bytes > 0
                ):
                    self._step_fp8_bucketed(group)
                    continue
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    grad = parameter.grad
                    state = self.state[parameter]
                    self._apply_weight_decay_inplace(
                        parameter, grad, group["lr"], group["weight_decay"]
                    )
                    if grad.ndim == 2:
                        update = self._matrix_update(
                            parameter, grad, state, group["momentum"]
                        )
                    else:
                        update = self._vector_update(grad, state, group["momentum"])
                    self._apply_update(parameter, update, group["lr"])
        profile = self.state_comm.finish_step()
        self._emit_profile(profile)
        self._profile_step += 1
        return loss


class FrugalMuonMuon(StateShardedMuon):
    """Coordinate FRUGAL: stateful Muon on columns, stateless Muon elsewhere."""

    is_frugal = True

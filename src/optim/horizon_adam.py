"""HorizonAdam: AdamW with a finite-horizon codec for joint moment state.

Large tensors keep bit-packed coordinates of (m, log(v)), local block means
and scales, and sparse FP32 reference probes. A completed probe window supplies
the next codec's utility metric. Updates decode bounded chunks, never a full
shadow copy of the moments. Small tensors use FP32 AdamW state.

Decoded momentum is projected onto the exact Adam moment constraint
m_t^2 <= rho_t * v_t. This prevents quantization noise in low-variance
coordinates from creating an arbitrarily large preconditioned update.

This is an unfused research implementation. The learned metric holds future
gradients fixed and approximates update distortion, not closed-loop loss.
"""

from argparse import Namespace
from collections.abc import Callable, Iterable
from copy import deepcopy
import math

import torch
from torch import Tensor
from torch.optim import Optimizer

from .horizon_codec import decode_blocks, encode_blocks, log_variance_bounds, update_probes


class HorizonAdam(Optimizer):
    def __init__(
        self, params: Iterable, lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
        weight_decay: float = 1e-2, horizon: int = 32, total_bits: int = 8,
        block_size: int = 256, chunk_size: int = 65536, probe_size: int = 64,
        min_size: int = 4096, v_floor: float = 1e-30, ridge: float = 1e-4,
        stochastic_rounding: bool = True, rounding_seed: int = 0, codec_mode: str = "learned",
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        horizon=horizon, total_bits=total_bits, block_size=block_size,
                        chunk_size=chunk_size, probe_size=probe_size, min_size=min_size,
                        v_floor=v_floor, ridge=ridge, stochastic_rounding=stochastic_rounding,
                        rounding_seed=rounding_seed, codec_mode=codec_mode)
        self._validate(defaults)
        super().__init__(params, defaults)

    @staticmethod
    def _validate(group: dict) -> None:
        if group.get("codec_mode", "learned") not in ("learned", "identity"):
            raise ValueError("HorizonAdam codec_mode must be learned or identity")
        for name in ("lr", "weight_decay", "eps", "v_floor", "ridge"):
            value = group[name]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid HorizonAdam {name}: {value}")
        if group["eps"] == 0 or not 1e-38 <= group["v_floor"] < 1 or group["ridge"] == 0:
            raise ValueError("HorizonAdam requires eps > 0, ridge > 0, and 1e-38 <= v_floor < 1")
        if len(group["betas"]) != 2 or not all(0 <= beta < 1 for beta in group["betas"]):
            raise ValueError("HorizonAdam betas must be in [0, 1)")
        for name in ("horizon", "total_bits", "block_size", "chunk_size", "probe_size", "min_size", "rounding_seed"):
            if not isinstance(group[name], int):
                raise ValueError(f"HorizonAdam {name} must be an integer")
        if group["horizon"] < 1 or group["probe_size"] < 2 or group["min_size"] < 0:
            raise ValueError("HorizonAdam requires horizon >= 1, probe_size >= 2, min_size >= 0")
        if not 4 <= group["total_bits"] <= 16:
            raise ValueError("HorizonAdam total_bits must be between 4 and 16")
        if group["block_size"] < 8 or group["block_size"] % 8:
            raise ValueError("HorizonAdam block_size must be a positive multiple of 8")
        if group["chunk_size"] < group["block_size"] or group["chunk_size"] % group["block_size"]:
            raise ValueError("HorizonAdam chunk_size must be a positive multiple of block_size")

    def add_param_group(self, param_group: dict) -> None:
        self._validate({**self.defaults, **param_group})
        super().add_param_group(param_group)

    def _initialize(self, parameter: Tensor, group: dict, index: int) -> dict:
        state = self.state[parameter]
        state.update(step=0, codec_version=1, compressed=parameter.numel() >= max(2, group["min_size"]))
        if not state["compressed"]:
            state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
            return state

        block_size = group["block_size"]
        blocks = (parameter.numel() + block_size - 1) // block_size
        first_bits = group["total_bits"] // 2
        state["bits"] = (first_bits, group["total_bits"] - first_bits)
        state["encoder"] = torch.eye(2, device=parameter.device, dtype=torch.float32)
        state["decoder"] = torch.eye(2, device=parameter.device, dtype=torch.float32)
        state["spectrum"] = torch.zeros(2, device=parameter.device, dtype=torch.float32)
        state["packed"] = torch.empty(blocks, block_size * group["total_bits"] // 8,
                                      device=parameter.device, dtype=torch.uint8)
        state["centers"] = torch.empty(blocks, 2, device=parameter.device, dtype=torch.float32)
        state["scales"] = torch.empty_like(state["centers"])
        count = min(group["probe_size"], parameter.numel())
        # Deterministic, distinct probes across the entire tensor, identical on DDP ranks.
        state["probe_indices"] = torch.arange(count, device=parameter.device) * parameter.numel() // count
        state["probe_m"] = torch.zeros(count, device=parameter.device, dtype=torch.float32)
        state["probe_v"] = torch.zeros_like(state["probe_m"])
        state["round_seed"] = (group["rounding_seed"] + index * 0x9E3779B9) & 0xFFFFFFFF
        state["codec_updates"] = 0
        state["moment_bound"] = 0.0
        return state

    @staticmethod
    def _update_chunk(parameter: Tensor, gradient: Tensor, m: Tensor, v: Tensor,
                      group: dict, step: int) -> None:
        beta1, beta2 = group["betas"]
        m.lerp_(gradient, 1 - beta1)
        v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2).clamp_min_(group["v_floor"])
        denominator = (v / (1 - beta2 ** step)).sqrt().add_(group["eps"])
        parameter.mul_(1 - group["lr"] * group["weight_decay"])
        parameter.add_((m / denominator) * (-group["lr"] / (1 - beta1 ** step)))

    def _compressed_step(self, parameter: Tensor, gradient: Tensor, state: dict, group: dict) -> None:
        flat_parameter, flat_gradient = parameter.view(-1), gradient.view(-1)
        new_codec = None
        if group.get("codec_mode", "learned") == "learned":
            new_codec = update_probes(state, flat_gradient[state["probe_indices"]].float(), group)
        encoder = state["encoder"] if new_codec is None else new_codec["encoder"]
        bits = state["bits"] if new_codec is None else new_codec["bits"]
        block_size = group["block_size"]
        log_min, log_max = log_variance_bounds(group["v_floor"])
        for start in range(0, parameter.numel(), group["chunk_size"]):
            end = min(start + group["chunk_size"], parameter.numel())
            block_start = start // block_size
            block_end = (end + block_size - 1) // block_size
            block_slice = slice(block_start, block_end)
            if state["step"] == 1:
                values = torch.zeros(block_end - block_start, block_size, 2, device=parameter.device, dtype=torch.float32)
                values[..., 1] = log_min
            else:
                values = decode_blocks(state["packed"][block_slice], state["centers"][block_slice],
                                       state["scales"][block_slice], state["decoder"], state["bits"], block_size)
            pairs = values.view(-1, 2)
            m = pairs[:end - start, 0]
            v = pairs[:end - start, 1].clamp(log_min, log_max).exp()
            if state["step"] == 1:
                v.zero_()
            elif math.isfinite(state["moment_bound"]):
                # Weighted Cauchy-Schwarz: exact Adam moments already satisfy
                # this constraint. Only inconsistent decoded pairs are changed.
                bound = v.sqrt() * math.sqrt(state["moment_bound"])
                m.clamp_(min=-bound, max=bound)
            self._update_chunk(flat_parameter[start:end], flat_gradient[start:end].float(), m, v, group, state["step"])
            pairs[:end - start, 1] = v.log()
            # Repeat the final valid pair so padding cannot create an artificial outlier.
            if end - start < pairs.shape[0]:
                pairs[end - start:] = pairs[end - start - 1].clone()
            packed, centers, scales = encode_blocks(values, encoder, bits, group["stochastic_rounding"],
                                                    start, (state["round_seed"] + state["step"] * 0x9E3779B9) & 0xFFFFFFFF)
            state["packed"][block_slice] = packed
            state["centers"][block_slice] = centers
            state["scales"][block_slice] = scales
        # Every chunk was decoded using the old codec and encoded using the new one.
        if new_codec is not None:
            state.update(new_codec)
            state["codec_updates"] += 1
        beta1, beta2 = group["betas"]
        if beta2 > 0:
            state["moment_bound"] = beta1 ** 2 / beta2 * state["moment_bound"] + (1 - beta1) ** 2 / (1 - beta2)
        else:
            # With beta2=0 and momentum, past gradients have no variance weight,
            # so no finite universal moment bound exists.
            state["moment_bound"] = 1.0 if beta1 == 0 else math.inf

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        index = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                index += 1
                gradient = parameter.grad
                if gradient is None or parameter.numel() == 0:
                    continue
                if gradient.is_sparse or not parameter.is_floating_point():
                    raise RuntimeError("HorizonAdam supports only dense, real floating-point gradients")
                if not parameter.is_contiguous() or not gradient.is_contiguous():
                    raise RuntimeError("HorizonAdam requires contiguous parameters and gradients")
                state = self.state[parameter] or self._initialize(parameter, group, index)
                state["step"] += 1
                if state["compressed"]:
                    self._compressed_step(parameter, gradient, state, group)
                else:
                    self._update_chunk(parameter, gradient.float(), state["exp_avg"], state["exp_avg_sq"], group, state["step"])
        return loss

    def load_state_dict(self, state_dict: dict) -> None:
        """Preserve packed integers and FP32 metadata, even for BF16 parameters."""
        saved = state_dict["state"]
        for state in saved.values():
            if state and state.get("codec_version") != 1:
                raise ValueError("Expected a HorizonAdam checkpoint (codec_version=1)")
        for group in state_dict["param_groups"]:
            self._validate(group)
        # Optimizer's default loader casts *all* parameter state tensors to the
        # parameter dtype, which would corrupt packed bytes and probe indices.
        super().load_state_dict({**state_dict, "state": {}})
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups):
            for key, parameter in zip(saved_group["params"], group["params"]):
                if key in saved:
                    self.state[parameter] = {
                        name: value.to(device=parameter.device).clone() if isinstance(value, Tensor) else deepcopy(value)
                        for name, value in saved[key].items()
                    }

    def state_size_bytes(self) -> int:
        """Actual tensor storage, including block metadata and reference probes."""
        return sum(value.numel() * value.element_size() for state in self.state.values()
                   for value in state.values() if isinstance(value, Tensor))


def build_horizon_adam(param_groups: Iterable, args: Namespace) -> HorizonAdam:
    """Constructor shared by the training factory and the data-free smoke run."""
    if getattr(args, "fp8_optim", False):
        raise ValueError("HorizonAdam uses its own packed state codec; omit --fp8-optim.")
    return HorizonAdam(
        param_groups, lr=args.lr, betas=(args.beta1, args.beta2),
        eps=args.eps, weight_decay=args.weight_decay,
        horizon=args.horizon_adam_horizon, total_bits=args.horizon_adam_bits,
        block_size=args.horizon_adam_block_size, chunk_size=args.horizon_adam_chunk_size,
        probe_size=args.horizon_adam_probes, min_size=args.horizon_adam_min_size,
        v_floor=args.horizon_adam_v_floor, ridge=args.horizon_adam_ridge,
        stochastic_rounding=not args.horizon_adam_nearest_rounding,
        rounding_seed=args.horizon_adam_seed,
        codec_mode=getattr(args, "horizon_adam_codec", "learned"),
    )

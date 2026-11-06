import math

import torch

from optim.fp8_state import (
    FP8StateDictMixin,
    dequantize_fp8_state,
    init_fp8_state,
    quantize_fp8_state_,
)

from .apollo import APOLLOAdamW
from .frugal import GaloreAdamW


class _FP8ProjectedAdamMixin(FP8StateDictMixin):
    """Store Adam moments in FP8 while leaving projection state unchanged."""

    def __init__(self, *args, qargs, **kwargs):
        self.qargs = qargs
        super().__init__(*args, **kwargs)

    def _is_state_empty(self, state):
        return any(key not in state for key in ("step", "exp_avg", "exp_avg_sq"))

    @torch.no_grad()
    def _init_state(self, example=None, state=None):
        if state is None:
            state = {}
        if example is None:
            if "exp_avg" in state and "exp_avg_sq" in state:
                state["step"] = 0
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()
                state["scale_exp_avg"].zero_()
                state["scale_exp_avg_sq"].zero_()
                return state
            example = state.get("exp_avg")
        if example is None:
            raise ValueError("FP8 Adam state initialization requires an example tensor")

        state["step"] = 0
        init_fp8_state(state, "exp_avg", example, self.qargs, order="first")
        init_fp8_state(state, "exp_avg_sq", example, self.qargs, order="second")
        return state

    @torch.no_grad()
    def _compute_update(self, grad, state, lr, betas, eps, correct_bias, **kwargs):
        grad_fp32 = grad.to(torch.float32)
        exp_avg = dequantize_fp8_state(
            state, "exp_avg", self.qargs, signed=True
        )
        exp_avg_sq = dequantize_fp8_state(
            state, "exp_avg_sq", self.qargs, signed=False
        )
        beta1, beta2 = betas
        state["step"] += 1

        exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)
        denom = exp_avg_sq.sqrt()

        step_size = lr
        if correct_bias:
            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            step_size = lr / bias_correction1
            denom.div_(math.sqrt(bias_correction2))
        denom.add_(eps)

        update = exp_avg / denom * (-step_size)
        quantize_fp8_state_(
            state, "exp_avg", exp_avg, self.qargs, signed=True
        )
        quantize_fp8_state_(
            state, "exp_avg_sq", exp_avg_sq, self.qargs, signed=False
        )
        return update.to(grad.dtype)


class FP8GaloreAdamW(_FP8ProjectedAdamMixin, GaloreAdamW):
    pass


class FP8APOLLOAdamW(FP8StateDictMixin, APOLLOAdamW):
    def __init__(self, params, qargs, **kwargs):
        super().__init__(params, **kwargs)
        self.qargs = qargs

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[param]
                if "step" not in state:
                    state["step"] = 0
                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = self._initialize_projector(group, state)
                    grad = state["projector"].project(grad, state["step"])

                if "exp_avg" not in state:
                    init_fp8_state(state, "exp_avg", grad, self.qargs, order="first")
                    init_fp8_state(state, "exp_avg_sq", grad, self.qargs, order="second")

                grad_fp32 = grad.to(torch.float32)
                exp_avg = dequantize_fp8_state(
                    state, "exp_avg", self.qargs, signed=True
                )
                exp_avg_sq = dequantize_fp8_state(
                    state, "exp_avg_sq", self.qargs, signed=False
                )
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad_fp32, grad_fp32, value=1.0 - beta2
                )
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size *= math.sqrt(bias_correction2) / bias_correction1
                update = exp_avg / denom
                if "rank" in group:
                    if group["scale_type"] == "channel":
                        norm_dim = 0 if update.shape[0] < update.shape[1] else 1
                        grad_scaling_factor = torch.norm(update, dim=norm_dim) / (
                            torch.norm(grad, dim=norm_dim) + 1e-8
                        )
                        if norm_dim == 1:
                            grad_scaling_factor = grad_scaling_factor.unsqueeze(1)
                    elif group["scale_type"] == "tensor":
                        grad_scaling_factor = torch.norm(update) / (
                            torch.norm(grad) + 1e-8
                        )
                    else:
                        raise ValueError(
                            f"Invalid APOLLO scale type: {group['scale_type']}"
                        )

                    scaled_grad = param.grad * grad_scaling_factor
                    if self.scale_front:
                        scaled_grad *= math.sqrt(group["scale"])
                    if not self.disable_nl:
                        if "scaled_grad" in state:
                            scaled_grad_norm = torch.norm(scaled_grad)
                            limiter = max(
                                scaled_grad_norm / (state["scaled_grad"] + 1e-8),
                                1.01,
                            ) / 1.01
                            scaled_grad = scaled_grad / limiter
                            state["scaled_grad"] = scaled_grad_norm / limiter
                        else:
                            state["scaled_grad"] = torch.norm(scaled_grad)
                    update = scaled_grad
                    if not self.scale_front:
                        update *= math.sqrt(group["scale"])
                param.add_(update.to(param.dtype), alpha=-step_size)
                if group["weight_decay"] > 0.0:
                    param.add_(param, alpha=-group["lr"] * group["weight_decay"])

                quantize_fp8_state_(
                    state, "exp_avg", exp_avg, self.qargs, signed=True
                )
                quantize_fp8_state_(
                    state, "exp_avg_sq", exp_avg_sq, self.qargs, signed=False
                )

        return loss

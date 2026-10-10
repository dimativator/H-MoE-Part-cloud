import torch

from optim.fp8_state import (
    FP8StateDictMixin,
    dequantize_fp8_state,
    init_fp8_state,
    quantize_fp8_state_,
)

from .ademamix import linear_hl_warmup_scheduler, linear_warmup_scheduler
from .ademamix_sign import AdEMAMixSign


class FP8AdEMAMixSign(FP8StateDictMixin, AdEMAMixSign):
    """Sign-AdEMAMix with two stored FP8 EMAs and FP32 update arithmetic."""

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
            lr = group["lr"]
            beta1, beta3_final = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AdEMAMixSign does not support sparse gradients.")
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    if beta1 != 0.0:
                        init_fp8_state(state, "exp_avg_fast", param, self.qargs, order="first")
                    else:
                        state["exp_avg_fast"] = None
                    init_fp8_state(state, "exp_avg_slow", param, self.qargs, order="first")

                grad = param.grad.to(torch.float32)
                fast = dequantize_fp8_state(state, "exp_avg_fast", self.qargs, signed=True) if beta1 else grad
                slow = dequantize_fp8_state(state, "exp_avg_slow", self.qargs, signed=True)
                state["step"] += 1
                alpha = linear_warmup_scheduler(
                    state["step"], alpha_end=group["alpha"], alpha_start=0,
                    warmup=group["alpha_warmup"],
                ) if group["alpha_warmup"] is not None else group["alpha"]
                beta3 = linear_hl_warmup_scheduler(
                    state["step"], beta_end=beta3_final, beta_start=beta1,
                    warmup=group["beta3_warmup"],
                ) if group["beta3_warmup"] is not None else beta3_final
                if beta1:
                    fast.mul_(beta1).add_(grad, alpha=1 - beta1)
                slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                update = fast.add(slow, alpha=alpha).sign_()
                param.mul_(1 - lr * group["weight_decay"])
                param.add_(update.to(param.dtype), alpha=-lr)
                if beta1:
                    quantize_fp8_state_(state, "exp_avg_fast", fast, self.qargs, signed=True)
                quantize_fp8_state_(state, "exp_avg_slow", slow, self.qargs, signed=True)
        return loss

import torch

from optim.fp8_state import (
    FP8StateDictMixin,
    dequantize_fp8_state,
    init_fp8_state,
    quantize_fp8_state_,
)

from .slim_adam import SlimAdamW


class FP8SlimAdamW(FP8StateDictMixin, SlimAdamW):
    """SlimAdam with persistent first and compressed second moments in FP8."""

    def __init__(self, named_parameters, qargs, **kwargs):
        super().__init__(named_parameters, **kwargs)
        self.qargs = qargs

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            weight_decay = group["weight_decay"]
            compress_dims = group["compress_dims"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    init_fp8_state(state, "mu", param, self.qargs, order="first")
                    second_moment = self.compress_grad_squared(param, compress_dims)
                    init_fp8_state(
                        state, "nu", second_moment, self.qargs, order="second"
                    )

                grad_fp32 = param.grad.to(torch.float32)
                mu = dequantize_fp8_state(state, "mu", self.qargs, signed=True)
                nu = dequantize_fp8_state(state, "nu", self.qargs, signed=False)
                mu.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                grad_squared = self.compress_grad_squared(grad_fp32, compress_dims)
                nu.mul_(beta2).add_(grad_squared, alpha=1.0 - beta2)

                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                state["step"] += 1
                mu_hat = mu / (1.0 - beta1 ** state["step"])
                nu_hat = nu / (1.0 - beta2 ** state["step"])
                update = lr * mu_hat / (nu_hat.sqrt() + eps)
                param.add_(-update.to(param.dtype))

                quantize_fp8_state_(state, "mu", mu, self.qargs, signed=True)
                quantize_fp8_state_(state, "nu", nu, self.qargs, signed=False)

        return loss

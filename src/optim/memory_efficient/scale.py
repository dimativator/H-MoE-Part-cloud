"""SCALE optimizer adapted from OptimAI-Lab/Minimalist_LLM_Pretraining.

Upstream revision 94712d9, mem_eff_pt/pt_scale/scale_optimizer.py.  The
parameter partition follows its train_utils.py: attention/MLP/embedding
matrices use stateless column normalization, the output head additionally
uses momentum, and one-dimensional parameters use AdamW.
"""

from collections.abc import Iterable

import torch


class SCALE(torch.optim.Optimizer):
    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        lr: float,
        weight_decay: float,
        momentum: float = 0.9,
        adam_lr: float | None = None,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError("SCALE requires a positive learning rate")
        if not 0 <= momentum < 1:
            raise ValueError("SCALE momentum must be in [0, 1)")

        main: list[torch.nn.Parameter] = []
        secondary: list[torch.nn.Parameter] = []
        oned: list[torch.nn.Parameter] = []
        types: dict[torch.nn.Parameter, str] = {}
        embedding: set[torch.nn.Parameter] = set()
        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            if param in types:
                continue
            if param.ndim == 1:
                oned.append(param)
                types[param] = "oned_param"
            elif param.ndim >= 2 and (
                ".attn." in name or ".mlp." in name or name.endswith("wte.weight")
            ):
                main.append(param)
                types[param] = "main_param"
                if name.endswith("wte.weight"):
                    embedding.add(param)
            else:
                secondary.append(param)
                types[param] = "secondary_param"

        params = main + secondary + oned
        if not params:
            raise ValueError("SCALE received no trainable parameters")
        super().__init__(
            params,
            dict(
                lr=lr,
                wd=weight_decay,
                momentum=momentum,
                adam_lr=lr if adam_lr is None else adam_lr,
                adamw_betas=adamw_betas,
                adamw_eps=adamw_eps,
            ),
        )
        self.max_lr = lr
        self._embedding = embedding
        for param, kind in types.items():
            self.state[param]["param_type"] = kind

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            beta1 = group["momentum"]
            adam_beta1, adam_beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            adam_lr = group["adam_lr"] * lr / self.max_lr

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                kind = state["param_type"]
                if kind == "oned_param":
                    if "step" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(grad)
                        state["moment2"] = torch.zeros_like(grad)
                    state["step"] += 1
                    state["moment1"].lerp_(grad, 1 - adam_beta1)
                    state["moment2"].lerp_(grad.square(), 1 - adam_beta2)
                    update = state["moment1"] / (eps + state["moment2"].sqrt())
                    correction = (1 - adam_beta1 ** state["step"]) / (
                        1 - adam_beta2 ** state["step"]
                    ) ** 0.5
                    param.mul_(1 - adam_lr * wd)
                    param.add_(update, alpha=-adam_lr / correction)
                    continue

                if grad.ndim > 2:
                    grad = grad.view(grad.size(0), -1)
                if kind == "secondary_param":
                    if "moment1" not in state:
                        state["moment1"] = torch.zeros_like(grad)
                    state["moment1"].lerp_(grad, 1 - beta1)
                    grad = state["moment1"]
                axis = 0 if param in self._embedding else 1
                scale = grad.square().mean(dim=axis, keepdim=True).sqrt().clamp_min_(1e-8)
                update = grad / scale
                param.mul_(1 - lr * wd)
                param.add_(update.view_as(param), alpha=-lr)
        return loss

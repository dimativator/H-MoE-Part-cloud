"""MND-Muon (arXiv:2610.24839, Algorithm 2) with the paper's update scale."""

import torch
from torch.distributed.tensor import DTensor

from third_party.matrix_normal.kernels import mnd_muon_momentum, polar_express


class MNDMuon(torch.optim.Optimizer):
    """FP32 diagonal factors and direction-dependent whitening, then Polar Express."""

    def __init__(self, params, lr=0.03, beta=0.9, eps=1e-9, weight_decay=0.1,
                 polar_steps=5, adam_betas=(0.95, 0.95), adam_eps=1e-8):
        from math import isfinite

        if any(not isfinite(x) or x < 0 for x in (lr, weight_decay)):
            raise ValueError("lr and weight_decay must be finite and nonnegative")
        if not 0 < beta < 1 or not isfinite(eps) or eps <= 0:
            raise ValueError("beta must be in (0, 1) and eps must be positive")
        if type(polar_steps) is not int or polar_steps < 1:
            raise ValueError("polar_steps must be a positive integer")
        if len(adam_betas) != 2 or any(not 0 <= b < 1 for b in adam_betas):
            raise ValueError("adam_betas must contain two values in [0, 1)")
        if not isfinite(adam_eps) or adam_eps <= 0:
            raise ValueError("adam_eps must be finite and positive")
        super().__init__(params, dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay,
                                     polar_steps=polar_steps, adam_betas=adam_betas,
                                     adam_eps=adam_eps, is_proj_params=True))
        for group in self.param_groups:
            for p in group["params"]:
                if isinstance(p, DTensor):
                    raise NotImplementedError("MND-Muon requires full parameters (single GPU or DDP)")
                if group["is_proj_params"] and p.ndim != 2:
                    raise ValueError("MND-Muon hidden weights must be two-dimensional")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.layout != torch.strided:
                    raise RuntimeError("MND-Muon requires dense gradients")
                g = p.grad.float()
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(g)
                    if group["is_proj_params"]:
                        m, n = g.shape
                        state["row_factor"] = g.new_full((m,), 1 / m)
                        state["col_factor"] = g.new_full((n,), 1 / n)
                        state["direction"] = torch.zeros_like(g)
                    else:
                        state["variance"] = torch.zeros_like(g)
                state["step"] += 1
                with torch.autocast(device_type=p.device.type, enabled=False):
                    if group["is_proj_params"]:
                        momentum, row, col = mnd_muon_momentum(
                            g, state["momentum"], state["direction"],
                            state["row_factor"], state["col_factor"], group["beta"], group["eps"])
                        update = polar_express(momentum, group["polar_steps"])
                        state["momentum"].copy_(momentum)
                        state["row_factor"].copy_(row)
                        state["col_factor"].copy_(col)
                        state["direction"].copy_(update)
                    else:
                        b1, b2 = group["adam_betas"]
                        state["momentum"].lerp_(g, 1 - b1)
                        state["variance"].lerp_(g.square(), 1 - b2)
                        mean = state["momentum"] / (1 - b1 ** state["step"])
                        variance = state["variance"] / (1 - b2 ** state["step"])
                        update = mean / (variance.sqrt() + group["adam_eps"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"])
        return loss


def build_mnd_muon(param_groups: list[dict], args) -> MNDMuon:
    if getattr(args, "fp8_optim", False) or getattr(args, "optimizer_state_sharding", False):
        raise ValueError("MND-Muon requires replicated FP32 optimizer states")
    return MNDMuon(param_groups, lr=args.lr, weight_decay=args.weight_decay,
                   beta=args.mnd_muon_beta, eps=args.mnd_muon_eps,
                   polar_steps=args.mnd_muon_polar_steps,
                   adam_betas=(args.beta1, args.beta2), adam_eps=args.eps)

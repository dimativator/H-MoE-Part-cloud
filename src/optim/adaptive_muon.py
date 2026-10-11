"""Muon² (arXiv:2604.09967, Algorithm 1) and official NorMuon updates.

Uses replicated optimizer states with single-device or DDP training. NorMuon
kernels are vendored under third_party/normuon with their MIT license.
"""

import math

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT

from third_party.normuon.kernels import normuon_update, zeropower_via_newtonschulz5


class _AdaptiveMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.02,
        momentum: float = 0.95,
        beta2: float = 0.95,
        ns_steps: int = 5,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_lr_scale: float = 1.0,
    ):
        if lr < 0 or weight_decay < 0 or eps <= 0 or adamw_lr_scale < 0:
            raise ValueError("lr, weight_decay and adamw_lr_scale must be nonnegative; eps must be positive.")
        if not 0 <= momentum < 1 or not 0 <= beta2 < 1:
            raise ValueError("momentum and beta2 must be in [0, 1).")
        if len(adamw_betas) != 2 or any(not 0 <= b < 1 for b in adamw_betas):
            raise ValueError("adamw_betas must contain two values in [0, 1).")
        if not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError("ns_steps must be a positive integer.")
        defaults = dict(
            lr=lr, momentum=momentum, beta2=beta2, ns_steps=ns_steps,
            eps=eps, weight_decay=weight_decay, nesterov=nesterov,
            adamw_betas=adamw_betas, is_proj_params=True,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            if not group["is_proj_params"]:
                group["lr"] *= adamw_lr_scale
            for p in group["params"]:
                if isinstance(p, DTensor):
                    raise NotImplementedError("Adaptive Muon requires full parameters (single device or DDP).")
                if group["is_proj_params"] and p.ndim != 2:
                    raise ValueError("Muon² and NorMuon require 2D hidden weights.")

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        # Optimizer.load_state_dict casts floating buffers to the parameter dtype.
        # Restore directly from the saved FP32 values to avoid BF16 rounding.
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups):
            for saved_id, p in zip(saved_group["params"], group["params"]):
                saved_state = state_dict["state"].get(saved_id, {})
                for key in ("momentum", "variance"):
                    if key in saved_state:
                        self.state[p][key] = saved_state[key].to(device=p.device, dtype=torch.float32).clone()

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
                if p.grad.is_sparse:
                    raise RuntimeError("Adaptive Muon does not support sparse gradients.")
                # FP32 states keep second moments stable with BF16 model weights.
                g = p.grad.float()
                state = self.state[p]
                if not state:
                    state["momentum"] = torch.zeros_like(g)
                    shape = g[:, :1] if self._row_variance and group["is_proj_params"] else g
                    state["variance"] = torch.zeros_like(shape)
                    state["step"] = 0
                state["step"] += 1
                if group["is_proj_params"]:
                    update = self._matrix_update(g, state, group)
                else:
                    b1, b2 = group["adamw_betas"]
                    state["momentum"].lerp_(g, 1 - b1)
                    state["variance"].lerp_(g.square(), 1 - b2)
                    m = state["momentum"] / (1 - b1 ** state["step"])
                    v = state["variance"] / (1 - b2 ** state["step"])
                    update = m / (v.sqrt() + group["eps"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"])
        return loss


class Muon2(_AdaptiveMuon):
    """Algorithm 1: EMA gradients / sqrt(EMA squared gradients), then NS.

    No bias correction or Nesterov momentum. For W in R^(n x m), the
    orthogonalized update is multiplied by sqrt(m/n). Decoupled weight decay
    and auxiliary AdamW follow the repository's training conventions.
    """

    _row_variance = False

    def __init__(self, params: ParamsT, ns_steps: int = 5, **kwargs):
        super().__init__(params, ns_steps=ns_steps, **kwargs)

    def _matrix_update(self, g: Tensor, state: dict, group: dict) -> Tensor:
        state["momentum"].lerp_(g, 1 - group["momentum"])
        state["variance"].lerp_(g.square(), 1 - group["beta2"])
        preconditioned = state["momentum"] / (state["variance"].sqrt() + group["eps"])
        update = zeropower_via_newtonschulz5(preconditioned, steps=group["ns_steps"])
        return update.float() * math.sqrt(g.size(1) / g.size(0))


class NorMuon(_AdaptiveMuon):
    """Official row normalization and norm-preserving rescaling after NS."""

    _row_variance = True

    def _matrix_update(self, g: Tensor, state: dict, group: dict) -> Tensor:
        # The upstream Nesterov kernel modifies its gradient argument.
        return normuon_update(
            g.clone(), state["momentum"], state["variance"],
            beta=group["momentum"], beta2=group["beta2"],
            ns_steps=group["ns_steps"], nesterov=group["nesterov"],
        )


def build_adaptive_muon(param_groups: list[dict], args) -> _AdaptiveMuon:
    """Build from existing is_proj_params groups, preserving no-decay groups."""
    if getattr(args, "fp8_optim", False):
        raise ValueError("Muon² and NorMuon use FP32 states; --fp8-optim is not supported.")
    if getattr(args, "optimizer_state_sharding", False):
        raise ValueError("Muon² and NorMuon use replicated states; --optimizer-state-sharding is not supported.")
    cls = Muon2 if args.opt.lower() == "muon2" else NorMuon
    ns_steps = getattr(args, "muon2_ns_steps", 5) if cls is Muon2 else args.muon_ns_steps
    return cls(
        param_groups, lr=args.lr, momentum=args.momentum,
        beta2=args.muon_beta2, ns_steps=ns_steps, eps=args.eps,
        weight_decay=args.weight_decay, nesterov=args.nesterov,
        adamw_betas=(args.beta1, args.beta2), adamw_lr_scale=args.muon_adamw_lr_scale,
    )
